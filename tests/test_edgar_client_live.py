"""Integration tests against the real SEC EDGAR API. Marked `network` — skipped by
default in CI-less quick runs; run explicitly with `pytest -m network`."""

import pytest

from tools.edgar_client import EdgarClient
from tools.xbrl_parser import parse_company_facts, parse_filings

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def client():
    return EdgarClient()


def test_resolve_cik_known_ticker(client):
    assert client.resolve_cik("AAPL") == "0000320193"


def test_resolve_cik_unknown_ticker_raises(client):
    from tools.edgar_client import EdgarClientError

    with pytest.raises(EdgarClientError):
        client.resolve_cik("NOTATICKERXYZ")


def test_known_good_filing_reconciles_against_live_edgar(client):
    """The checkpoint test for Phase 1: pull AAPL's real filing history + XBRL facts
    live from EDGAR and confirm the parser extracts the actual reported FY2023 revenue
    ($383,285,000,000, from the FY2023 10-K, accn 0000320193-23-000106)."""
    cik = client.resolve_cik("AAPL")

    submissions = client.get_submissions(cik)
    filings = parse_filings(submissions, "AAPL", form_types=("10-K",))
    assert any(f.accession_number == "0000320193-23-000106" for f in filings)

    facts = client.get_company_facts(cik)
    results = parse_company_facts(
        facts, "AAPL", concepts=["RevenueFromContractWithCustomerExcludingAssessedTax"]
    )
    match = [
        r
        for r in results
        if r.accession_number == "0000320193-23-000106"
        and r.period_start == "2022-09-25"
        and r.period_end == "2023-09-30"
    ]
    assert len(match) == 1
    assert match[0].value == 383_285_000_000
