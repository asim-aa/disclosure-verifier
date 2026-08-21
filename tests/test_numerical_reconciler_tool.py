"""Offline unit tests for the Numerical Reconciler MCP tool (reconcile_claim) —
no network. Covers the gap found while auditing this tool's docstring for
Pillar 1's tool-contract checklist: as_of was accepted and threaded through by
agents/verification_agent.py, but the MCP tool itself never exposed the
parameter at all, so any direct caller of reconcile_claim (not routed through
the Coordinator) had no bitemporal protection against a later restatement."""

from tools import numerical_reconciler
from tools.schema import FinancialFact

TICKER = "AAPL"
FY23_START = "2022-09-25"
FY23_END = "2023-09-30"


def _fact(value: float, accession_number: str, filed: str) -> FinancialFact:
    return FinancialFact(
        ticker=TICKER,
        cik="0000320193",
        concept="Revenues",
        label="Revenues",
        value=value,
        unit="USD",
        period_start=FY23_START,
        period_end=FY23_END,
        fiscal_year=2023,
        fiscal_period="FY",
        form="10-K",
        filed=filed,
        accession_number=accession_number,
    )


class _FakeClient:
    def resolve_cik(self, ticker: str) -> str:
        return "0000320193"

    def get_company_facts(self, cik: str) -> dict:
        return {}  # unused — parse_company_facts is monkeypatched below


def _install_fake_client(monkeypatch, facts: list[FinancialFact]) -> None:
    monkeypatch.setattr(numerical_reconciler, "get_client", lambda: _FakeClient())
    monkeypatch.setattr(numerical_reconciler, "parse_company_facts", lambda raw, ticker, concepts=None: facts)


def test_as_of_omitted_reads_the_later_restatement(monkeypatch):
    facts = [
        _fact(383_285_000_000, "0000320193-23-000106", filed="2023-11-03"),
        _fact(390_000_000_000, "0000320193-24-999999", filed="2024-03-01"),
    ]
    _install_fake_client(monkeypatch, facts)

    result = numerical_reconciler.reconcile_claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="absolute",
        claimed_value=383_285_000_000,
        period_end=FY23_END,
        period_start=FY23_START,
    )
    assert result["verdict"] == "inconsistent"
    assert result["citations"] == ["0000320193-24-999999"]


def test_as_of_pinned_to_source_filing_protects_the_claim(monkeypatch):
    facts = [
        _fact(383_285_000_000, "0000320193-23-000106", filed="2023-11-03"),
        _fact(390_000_000_000, "0000320193-24-999999", filed="2024-03-01"),
    ]
    _install_fake_client(monkeypatch, facts)

    result = numerical_reconciler.reconcile_claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="absolute",
        claimed_value=383_285_000_000,
        period_end=FY23_END,
        period_start=FY23_START,
        as_of="2023-11-03",
    )
    assert result["verdict"] == "consistent"
    assert result["citations"] == ["0000320193-23-000106"]


def test_result_carries_a_machine_readable_reason_code(monkeypatch):
    _install_fake_client(monkeypatch, [])

    result = numerical_reconciler.reconcile_claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="absolute",
        claimed_value=1,
        period_end=FY23_END,
        period_start=FY23_START,
    )
    assert result["verdict"] == "unverifiable"
    assert result["reason_code"] == "missing_fact"
