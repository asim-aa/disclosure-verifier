"""Unit tests for MD&A extraction/chunking, against trimmed real 10-K and 10-Q
HTML fixtures (AAPL, sliced around the actual MD&A section boundaries)."""

from pathlib import Path

import pytest

from tools.mdna_parser import MdnaNotFoundError, chunk_mdna, extract_mdna_paragraphs

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def aapl_10k_html():
    return (FIXTURES / "aapl_10k_mdna_sample.htm").read_text()


@pytest.fixture
def aapl_10q_html():
    return (FIXTURES / "aapl_10q_mdna_sample.htm").read_text()


def test_extracts_10k_mdna_section(aapl_10k_html):
    paragraphs = extract_mdna_paragraphs(aapl_10k_html, "10-K")
    assert paragraphs
    # First real paragraph of AAPL's actual FY2025 10-K MD&A section.
    assert "should be read in conjunction with the consolidated financial statements" in paragraphs[0]


def test_10k_mdna_stops_before_item_7a(aapl_10k_html):
    paragraphs = extract_mdna_paragraphs(aapl_10k_html, "10-K")
    joined = " ".join(paragraphs)
    assert "Quantitative and Qualitative Disclosures About Market Risk" not in joined


def test_10k_mdna_excludes_table_of_contents_noise(aapl_10k_html):
    paragraphs = extract_mdna_paragraphs(aapl_10k_html, "10-K")
    # A bare "Item 7." TOC line (no title attached) would mean we grabbed the
    # wrong occurrence of the heading.
    assert "Item 7." not in paragraphs


def test_extracts_10q_mdna_section(aapl_10q_html):
    paragraphs = extract_mdna_paragraphs(aapl_10q_html, "10-Q")
    assert paragraphs
    assert any("forward-looking statements" in p for p in paragraphs[:3])


def test_10q_mdna_stops_before_item_3(aapl_10q_html):
    paragraphs = extract_mdna_paragraphs(aapl_10q_html, "10-Q")
    joined = " ".join(paragraphs)
    assert "Quantitative and Qualitative Disclosures About Market Risk" not in joined


def test_unsupported_form_type_raises():
    with pytest.raises(MdnaNotFoundError):
        extract_mdna_paragraphs("<html></html>", "8-K")


def test_missing_section_raises_not_silently_empty():
    with pytest.raises(MdnaNotFoundError):
        extract_mdna_paragraphs("<html><body><p>nothing relevant here</p></body></html>", "10-K")


def test_chunk_mdna_produces_text_chunks_with_citations(aapl_10k_html):
    chunks = chunk_mdna(
        aapl_10k_html,
        ticker="aapl",
        cik="0000320193",
        accession_number="0000320193-25-000079",
        form="10-K",
        filing_date="2025-10-31",
    )
    assert chunks
    assert all(c.ticker == "AAPL" for c in chunks)
    assert all(c.accession_number == "0000320193-25-000079" for c in chunks)
    assert all(c.section == "MD&A" for c in chunks)
    # chunk_index is a stable, contiguous ordering starting at 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_mdna_drops_trivially_short_lines(aapl_10k_html):
    chunks = chunk_mdna(
        aapl_10k_html,
        ticker="AAPL",
        cik="0000320193",
        accession_number="0000320193-25-000079",
        form="10-K",
        filing_date="2025-10-31",
    )
    # Bullet glyphs / bare sub-headers like "•" or "First Quarter 2025:" shouldn't
    # survive as their own "claim" chunks.
    assert all(len(c.text) >= 20 for c in chunks)


def test_chunk_mdna_no_chunk_is_a_mid_sentence_fragment(aapl_10k_html):
    """Every chunk should read as a complete unit (not truncated mid-word) since
    chunk boundaries come from the filer's own HTML block elements, not our slicing."""
    chunks = chunk_mdna(
        aapl_10k_html,
        ticker="AAPL",
        cik="0000320193",
        accession_number="0000320193-25-000079",
        form="10-K",
        filing_date="2025-10-31",
    )
    for c in chunks:
        stripped = c.text.strip()
        assert stripped == c.text  # no leading/trailing whitespace fragments
        assert stripped[-1] not in ("-",), f"looks mid-word: {stripped!r}"


# ---------- split heading (item number and title in separate text nodes) ----------
#
# Phase 6's integration-at-scale run found real filers (GOOGL, AMZN) render the
# *real* body heading exactly the way AAPL/MSFT/NVDA render only their ToC entry —
# item number and title as separate adjacent nodes, not one combined node. This
# synthetic fixture reproduces that shape (split ToC entry, then a long real
# section with a split heading too) without depending on live EDGAR access.


def _split_heading_html(body_paragraph_count: int = 60) -> str:
    body_paragraphs = "".join(
        f"<p>Real MD&amp;A body sentence number {i} discusses actual results in enough "
        f"length to look like genuine prose rather than a heading or a page number.</p>"
        for i in range(body_paragraph_count)
    )
    return f"""
    <html><body>
      <p>Item&nbsp;7.</p>
      <p>Management's Discussion and Analysis of Financial Condition and Results of Operations</p>
      <p>44</p>
      <p>Item&nbsp;7A.</p>
      <p>Quantitative and Qualitative Disclosures About Market Risk</p>
      <p>72</p>
      <p>Item&nbsp;8.</p>
      <p>Financial Statements and Supplementary Data</p>
      <p>73</p>

      <p>Some unrelated intervening section text that is not part of the ToC or the
      real MD&amp;A, included so the real section isn't merely "whatever comes
      right after the ToC".</p>

      <p>Item&nbsp;7.</p>
      <p>Management's Discussion and Analysis of Financial Condition and Results of Operations</p>
      {body_paragraphs}
      <p>Item&nbsp;7A.</p>
      <p>Quantitative and Qualitative Disclosures About Market Risk</p>
      <p>Real Item 7A content that must not leak into the MD&amp;A section.</p>
    </body></html>
    """


def test_split_heading_finds_the_real_section_not_the_table_of_contents():
    paragraphs = extract_mdna_paragraphs(_split_heading_html(), "10-K")
    assert any("Real MD&A body sentence number 0" in p for p in paragraphs)
    assert any("Real MD&A body sentence number 59" in p for p in paragraphs)


def test_split_heading_excludes_the_toc_page_number_line():
    paragraphs = extract_mdna_paragraphs(_split_heading_html(), "10-K")
    assert "44" not in paragraphs
    assert "72" not in paragraphs


def test_split_heading_stops_before_the_real_item_7a():
    paragraphs = extract_mdna_paragraphs(_split_heading_html(), "10-K")
    joined = " ".join(paragraphs)
    assert "Real Item 7A content" not in joined


def test_split_heading_with_a_short_real_section_still_prefers_it_over_the_toc():
    """Even when the real section is shorter than usual, it should still win over
    the ToC/cross-reference candidates as long as it's the largest gap available —
    guards against the fix accidentally requiring a hardcoded minimum length."""
    paragraphs = extract_mdna_paragraphs(_split_heading_html(body_paragraph_count=3), "10-K")
    assert any("Real MD&A body sentence number 0" in p for p in paragraphs)
    assert "44" not in paragraphs


def _dash_heading_html() -> str:
    """Real heading text confirmed against INTU's live FY2025 10-K: 'ITEM 7 -
    MANAGEMENT'S...' with a dash separator and no period, not the '.'-separated
    form (AAPL/MSFT/etc.) the parser was originally built against."""
    body = "".join(
        f"<p>Real MD&amp;A body sentence number {i} discusses actual results in enough "
        f"length to look like genuine prose rather than a heading or a page number.</p>"
        for i in range(30)
    )
    return f"""
    <html><body>
      <p>ITEM 7 - MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL</p>
      <p>CONDITION AND RESULTS OF OPERATIONS</p>
      {body}
      <p>ITEM 7A - QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET</p>
      <p>RISK</p>
      <p>Real Item 7A content that must not leak into the MD&amp;A section.</p>
    </body></html>
    """


def test_dash_separated_heading_is_found():
    """Real bug, found via research/specificity_check.py's tech-company control
    set: INTU's real 10-K returned zero MD&A chunks because the heading regex
    only tolerated an optional period between the item number and its title, not
    a dash - and INTU's real heading uses a dash with no period at all."""
    paragraphs = extract_mdna_paragraphs(_dash_heading_html(), "10-K")
    assert any("Real MD&A body sentence number 0" in p for p in paragraphs)


def test_dash_separated_heading_stops_before_item_7a():
    paragraphs = extract_mdna_paragraphs(_dash_heading_html(), "10-K")
    joined = " ".join(paragraphs)
    assert "Real Item 7A content" not in joined
