"""Unit tests for XBRL/submissions parsing, against a trimmed real-data fixture
(a subset of AAPL's actual EDGAR company-facts and submissions payloads)."""

import json
from pathlib import Path

import pytest

from tools.xbrl_parser import parse_company_facts, parse_filings, list_available_concepts

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def aapl_facts():
    return json.loads((FIXTURES / "aapl_facts_sample.json").read_text())


@pytest.fixture
def aapl_submissions():
    return json.loads((FIXTURES / "aapl_submissions_sample.json").read_text())


def test_list_available_concepts(aapl_facts):
    concepts = list_available_concepts(aapl_facts)
    assert "Assets" in concepts
    assert "RevenueFromContractWithCustomerExcludingAssessedTax" in concepts


def test_parse_company_facts_known_revenue(aapl_facts):
    """Known-good check: AAPL's FY2023 10-K (accn 0000320193-23-000106) reported
    full-year revenue of $383,285,000,000 for the period 2022-09-25 to 2023-09-30."""
    results = parse_company_facts(
        aapl_facts, "AAPL", concepts=["RevenueFromContractWithCustomerExcludingAssessedTax"]
    )
    assert results

    match = [
        r
        for r in results
        if r.period_start == "2022-09-25"
        and r.period_end == "2023-09-30"
        and r.form == "10-K"
        and r.fiscal_period == "FY"
        and r.accession_number == "0000320193-23-000106"
    ]
    assert len(match) == 1
    fact = match[0]
    assert fact.value == 383_285_000_000
    assert fact.unit == "USD"
    assert fact.ticker == "AAPL"
    assert fact.cik == "0000320193"


def test_parse_company_facts_filters_by_concept(aapl_facts):
    results = parse_company_facts(aapl_facts, "AAPL", concepts=["Assets"])
    assert all(r.concept == "Assets" for r in results)
    assert results


def test_parse_company_facts_no_filter_returns_all_concepts(aapl_facts):
    results = parse_company_facts(aapl_facts, "AAPL")
    concepts_seen = {r.concept for r in results}
    assert concepts_seen == {"Assets", "RevenueFromContractWithCustomerExcludingAssessedTax"}


def test_parse_filings_defaults_to_10k_10q_8k(aapl_submissions):
    filings = parse_filings(aapl_submissions, "AAPL")
    assert filings
    assert all(f.form in ("10-K", "10-Q", "8-K") for f in filings)
    assert all(f.ticker == "AAPL" for f in filings)
    assert all(f.cik == "0000320193" for f in filings)


def test_parse_filings_filters_to_requested_form_types(aapl_submissions):
    filings = parse_filings(aapl_submissions, "AAPL", form_types=("10-Q",))
    assert filings
    assert all(f.form == "10-Q" for f in filings)


def test_parse_filings_respects_limit(aapl_submissions):
    filings = parse_filings(aapl_submissions, "AAPL", limit=2)
    assert len(filings) <= 2


def test_filing_meta_url_is_well_formed(aapl_submissions):
    filings = parse_filings(aapl_submissions, "AAPL", limit=1)
    url = filings[0].filing_url()
    assert url.startswith("https://www.sec.gov/Archives/edgar/data/320193/")
    assert filings[0].primary_document in url
