"""Extracts the Management's Discussion and Analysis (MD&A) section from a filing's
HTML document and chunks it into paragraph-level text.

This is the fallback claim source in place of earnings-call transcripts (see the
proposal's Week 13 scope safeguard): MD&A is the prose section of a 10-K/10-Q where
companies narrate their own results ("revenue grew 12% YoY...") — the same kind of
checkable claims transcripts would have provided.

Section boundaries are found by matching the *body heading* text, not just the item
number — a bare "Item 7." also appears in the table of contents and in incidental
cross-references elsewhere in the document. The item number and title can render as
one combined text node or as separate adjacent nodes depending on the filer (Phase 6's
integration-at-scale run found this varies: AAPL/MSFT/NVDA combine them, GOOGL/AMZN
render the real heading the same split way a table-of-contents entry does) — so
matching on "combined node" alone doesn't generalize, and can't be used to tell a real
heading from a ToC entry once matching is loosened to allow the split form too.

What does generalize: a real section runs for hundreds of lines of prose before the
next Item heading; a ToC entry or a cross-reference is only ever a few lines from
whatever comes next. Every window where the item number and title appear near each
other is treated as a *candidate*, matched against every candidate end-of-section
marker the same way, and the (start, end) pair with the largest gap between them wins
— which is robust to exactly how a given filer happens to render the heading.
"""

import re
import warnings
from dataclasses import dataclass

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from tools.schema import TextChunk

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

MDNA_SECTION_NAME = "MD&A"

_MIN_CHUNK_LENGTH = 20  # drops bullet glyphs, bare page numbers, sub-headers

# Elements that mark a real paragraph/cell boundary. A "leaf" element from this
# set (one with no descendant also in this set) is one text chunk; an inline
# wrapper (span, b, i, font, a, ...) inside it is glued into that same chunk,
# not split out on its own - see extract_paragraphs's docstring for why this
# matters.
_BLOCK_TAGS = ("div", "p", "li", "td", "th", "tr", "h1", "h2", "h3", "h4", "h5", "h6")


@dataclass(frozen=True)
class _SectionPattern:
    start: re.Pattern
    end_candidates: tuple[re.Pattern, ...]


# The separator between an Item number and its title varies by filer - most use a
# bare period ("Item 7. Management's..."), but some (confirmed against real INTU
# 10-K data) use a dash instead ("ITEM 7 - MANAGEMENT'S..."), with no period at
# all. `\.?` alone only tolerated an optional period; this also tolerates an
# optional dash (hyphen-minus, en dash, or em dash) in its place, with whitespace
# freely around either. Still requires the title text itself to follow, so this
# doesn't loosen the match enough to catch unrelated "Item 7" cross-references.
_ITEM_SEPARATOR = r"[\s\xa0]*[.\-–—]?[\s\xa0]*"

# MD&A lives at a different Item number in a 10-K (Item 7) vs a 10-Q (Item 2).
_SECTION_PATTERNS: dict[str, _SectionPattern] = {
    "10-K": _SectionPattern(
        start=re.compile(r"Item\s*7" + _ITEM_SEPARATOR + r"Management.{0,3}s\s+Discussion", re.IGNORECASE),
        end_candidates=(
            re.compile(r"Item\s*7A" + _ITEM_SEPARATOR + r"Quantitative", re.IGNORECASE),
            re.compile(r"Item\s*8" + _ITEM_SEPARATOR + r"Financial\s+Statements", re.IGNORECASE),
        ),
    ),
    "10-Q": _SectionPattern(
        start=re.compile(r"Item\s*2" + _ITEM_SEPARATOR + r"Management.{0,3}s\s+Discussion", re.IGNORECASE),
        end_candidates=(re.compile(r"Item\s*3" + _ITEM_SEPARATOR + r"Quantitative", re.IGNORECASE),),
    ),
}


class MdnaNotFoundError(RuntimeError):
    """Raised when the MD&A section boundaries can't be located in a filing's HTML.
    Deliberately not swallowed — silently returning an empty/wrong section would
    poison every claim extracted from it downstream."""


# How many consecutive text-node lines to join when checking for a heading match —
# covers a title split across up to this many nodes (item number, title, and one
# more for safety) without being so wide it starts matching unrelated nearby text.
_HEADING_WINDOW = 3


def _find_heading_candidates(lines: list[str], pattern: re.Pattern) -> list[tuple[int, int]]:
    """Every position where up to _HEADING_WINDOW consecutive lines, joined with a
    space, match `pattern` — as (start_idx, end_idx_exclusive) spans. Returns every
    candidate rather than just the first; the caller (which also knows where each
    candidate's *next* heading falls) is what actually tells a real section heading
    apart from a table-of-contents entry or an incidental cross-reference.

    The match must *begin* within the window's first line, not merely appear
    somewhere in the joined text — otherwise a window starting at unrelated
    preceding prose could "absorb" a real heading that actually starts on a later
    line, silently stealing that line from whichever section it belongs to."""
    candidates = []
    for i in range(len(lines)):
        for span in range(1, _HEADING_WINDOW + 1):
            joined = " ".join(lines[i : i + span])
            match = pattern.search(joined)
            if match and match.start() <= len(lines[i]):
                candidates.append((i, i + span))
                break  # shortest matching window for this start line
    return candidates


def extract_paragraphs(html: str) -> list[str]:
    """Flatten a filing's HTML into one string per *leaf* block-level element
    (div/p/li/td/th/tr/h1-h6 with no such element nested inside it), in document
    order.

    Real bug, confirmed against CRM's live 10-K: a single sentence — "For fiscal
    2026, diluted net income per share was $7.80 as compared to diluted net
    income per share of $6.36 from a year ago." — renders as ONE <div> made of
    five sibling <span> elements, each wrapping a differently-styled run (plain
    text, then a bolded/highlighted number, repeated). The previous version of
    this function called `soup.get_text("\\n")` on the whole document, which
    inserts its separator between every distinct text node it walks past -
    including between sibling *inline* elements, not just between real
    paragraph boundaries. That split this one sentence into pieces across
    separate downstream chunks, so extraction saw "$6.36 from a year ago"
    completely divorced from "diluted net income per share" and any current-
    period context - a genuine mid-sentence truncation the module's own
    (previously wrong) docstring claimed couldn't happen.

    Walking BLOCK-level leaf elements and joining each one's own text with a
    plain space (not a further per-child split) keeps a filer's real paragraph
    breaks - `extract_mdna_paragraphs`'s heading detection already tolerates a
    heading rendered across more than one of these leaves (see
    `_find_heading_candidates`'s window) - while no longer manufacturing a
    fake break at every inline styling boundary a filer never intended as one."""
    soup = BeautifulSoup(html, "lxml")
    lines = []
    for tag in soup.find_all(_BLOCK_TAGS):
        if tag.find(_BLOCK_TAGS) is not None:
            continue  # not a leaf - its block-level children are collected separately
        text = tag.get_text(" ", strip=True)
        if text:
            lines.append(text)
    return lines


def extract_mdna_paragraphs(html: str, form: str) -> list[str]:
    """Return the paragraph lines that fall inside the MD&A section for the given
    form type (10-K or 10-Q). Raises MdnaNotFoundError if the section can't be found —
    callers should treat that as "this filing needs a fallback", not as empty data."""
    pattern = _SECTION_PATTERNS.get(form)
    if pattern is None:
        raise MdnaNotFoundError(f"No MD&A section pattern defined for form type '{form}'")

    lines = extract_paragraphs(html)

    start_candidates = _find_heading_candidates(lines, pattern.start)
    if not start_candidates:
        raise MdnaNotFoundError(f"Could not locate MD&A start heading for form '{form}'")

    end_candidates = sorted(
        span for end_pattern in pattern.end_candidates for span in _find_heading_candidates(lines, end_pattern)
    )
    if not end_candidates:
        raise MdnaNotFoundError(f"Could not locate MD&A end heading for form '{form}'")

    # Pair each start candidate with the nearest end candidate strictly after it,
    # then take the pair with the largest gap — see the module docstring for why
    # that's what actually distinguishes the real section from a ToC entry once
    # matching is loosened enough to catch a heading split across text nodes.
    best: tuple[int, int, int] | None = None  # (gap, body_start, body_end)
    for _, s_end in start_candidates:
        e = next(((e_start, e_end) for e_start, e_end in end_candidates if e_start >= s_end), None)
        if e is None:
            continue
        gap = e[0] - s_end
        if best is None or gap > best[0]:
            best = (gap, s_end, e[0])

    if best is None:
        raise MdnaNotFoundError(f"Could not locate MD&A end heading for form '{form}'")

    _, body_start, body_end = best
    return lines[body_start:body_end]


def chunk_mdna(
    html: str,
    ticker: str,
    cik: str,
    accession_number: str,
    form: str,
    filing_date: str,
) -> list[TextChunk]:
    """Extract MD&A and chunk it into TextChunks, dropping trivially short lines
    (bullet glyphs, bare sub-headers, page numbers) that aren't checkable claims."""
    paragraphs = extract_mdna_paragraphs(html, form)
    kept = [p for p in paragraphs if len(p) >= _MIN_CHUNK_LENGTH]

    return [
        TextChunk(
            ticker=ticker.upper(),
            cik=cik,
            accession_number=accession_number,
            form=form,
            filing_date=filing_date,
            section=MDNA_SECTION_NAME,
            chunk_index=i,
            text=text,
        )
        for i, text in enumerate(kept)
    ]
