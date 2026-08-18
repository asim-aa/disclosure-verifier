"""Extracts the Management's Discussion and Analysis (MD&A) section from a filing's
HTML document and chunks it into paragraph-level text.

This is the fallback claim source in place of earnings-call transcripts (see the
proposal's Week 13 scope safeguard): MD&A is the prose section of a 10-K/10-Q where
companies narrate their own results ("revenue grew 12% YoY...") — the same kind of
checkable claims transcripts would have provided.

Section boundaries are found by matching the *body heading* text, not just the item
number: EDGAR filings render their table of contents as separate text nodes per cell
(e.g. "Item 7." on its own line, then the title, then a page number on other lines),
while the real section heading is one text node combining both ("Item 7.
Management's Discussion and Analysis..."). Matching on the combined pattern skips
the TOC automatically without needing to special-case it.
"""

import re
import warnings
from dataclasses import dataclass

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from tools.schema import TextChunk

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

MDNA_SECTION_NAME = "MD&A"

_MIN_CHUNK_LENGTH = 20  # drops bullet glyphs, bare page numbers, sub-headers


@dataclass(frozen=True)
class _SectionPattern:
    start: re.Pattern
    end_candidates: tuple[re.Pattern, ...]


# MD&A lives at a different Item number in a 10-K (Item 7) vs a 10-Q (Item 2).
_SECTION_PATTERNS: dict[str, _SectionPattern] = {
    "10-K": _SectionPattern(
        start=re.compile(r"Item\s*7\.?[\s\xa0]*Management.{0,3}s\s+Discussion", re.IGNORECASE),
        end_candidates=(
            re.compile(r"Item\s*7A\.?[\s\xa0]*Quantitative", re.IGNORECASE),
            re.compile(r"Item\s*8\.?[\s\xa0]*Financial\s+Statements", re.IGNORECASE),
        ),
    ),
    "10-Q": _SectionPattern(
        start=re.compile(r"Item\s*2\.?[\s\xa0]*Management.{0,3}s\s+Discussion", re.IGNORECASE),
        end_candidates=(re.compile(r"Item\s*3\.?[\s\xa0]*Quantitative", re.IGNORECASE),),
    ),
}


class MdnaNotFoundError(RuntimeError):
    """Raised when the MD&A section boundaries can't be located in a filing's HTML.
    Deliberately not swallowed — silently returning an empty/wrong section would
    poison every claim extracted from it downstream."""


def extract_paragraphs(html: str) -> list[str]:
    """Flatten a filing's HTML into one string per block-level text node, in document
    order. Each result corresponds to a single original HTML element, so a chunk can
    never span a mid-sentence tag boundary the filer didn't already treat as a break."""
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text("\n")
    return [line.strip() for line in text.split("\n") if line.strip()]


def extract_mdna_paragraphs(html: str, form: str) -> list[str]:
    """Return the paragraph lines that fall inside the MD&A section for the given
    form type (10-K or 10-Q). Raises MdnaNotFoundError if the section can't be found —
    callers should treat that as "this filing needs a fallback", not as empty data."""
    pattern = _SECTION_PATTERNS.get(form)
    if pattern is None:
        raise MdnaNotFoundError(f"No MD&A section pattern defined for form type '{form}'")

    lines = extract_paragraphs(html)

    start_idx = next((i for i, l in enumerate(lines) if pattern.start.search(l)), None)
    if start_idx is None:
        raise MdnaNotFoundError(f"Could not locate MD&A start heading for form '{form}'")

    end_idx = None
    for end_pattern in pattern.end_candidates:
        match = next(
            (i for i in range(start_idx + 1, len(lines)) if end_pattern.search(lines[i])),
            None,
        )
        if match is not None and (end_idx is None or match < end_idx):
            end_idx = match

    if end_idx is None:
        raise MdnaNotFoundError(f"Could not locate MD&A end heading for form '{form}'")

    return lines[start_idx + 1 : end_idx]


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
