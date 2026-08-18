"""Live checkpoint test for the MD&A Extractor: pulls AAPL's real, current 10-K
from EDGAR and confirms MD&A is found and chunked with correct citations. Marked
`network` — run explicitly with `pytest -m network`."""

import pytest

from tools.mdna_extractor import get_mdna

pytestmark = pytest.mark.network


def test_get_mdna_returns_cited_chunks_from_live_filing():
    chunks = get_mdna("AAPL", form_type="10-K", limit=1)
    assert chunks
    assert all("error" not in c for c in chunks), [c for c in chunks if "error" in c]
    assert all(c["ticker"] == "AAPL" for c in chunks)
    assert all(c["cik"] == "0000320193" for c in chunks)
    assert all(c["section"] == "MD&A" for c in chunks)
    assert all(len(c["text"]) >= 20 for c in chunks)
