"""The scenario traces explicitly required by the capstone brief ("tested with
mock tools and scenario traces"): confirm the coordinator correctly handles a
claim that is consistent, one that is inconsistent, and one that can't be
verified (missing data) — end to end through the real Coordinator, with every
agent mocked so this runs with no network calls and no LLM calls at all.
"""

from agents.coordinator import Coordinator
from agents.extraction_agent import MockExtractionAgent
from agents.retrieval_agent import MockRetrievalAgent
from agents.schema import VerificationOutcome
from agents.verification_agent import MockVerificationAgent, RealVerificationAgent
from eval.schema import ExtractedClaim
from tools.schema import (
    VERDICT_CONSISTENT,
    VERDICT_INCONSISTENT,
    VERDICT_UNVERIFIABLE,
    FinancialFact,
    TextChunk,
)

CHUNK = TextChunk(
    ticker="ACME",
    cik="0000000001",
    accession_number="0000000001-26-000001",
    form="10-K",
    filing_date="2026-01-01",
    section="MD&A",
    chunk_index=0,
    text="Revenue increased 12% to $1.2 billion.",
)

CLAIM = ExtractedClaim(
    metric="Revenue",
    value=12.0,
    value_unit="percent",
    period="",
    comparison_type="growth_pct",
    quote="Revenue increased 12%",
)


def make_coordinator(outcome: VerificationOutcome, checkpoint_dir) -> Coordinator:
    retrieval = MockRetrievalAgent(chunks=[CHUNK], facts=[])
    extraction = MockExtractionAgent(canned={CHUNK.text: [CLAIM]})
    verification = MockVerificationAgent(outcomes=[outcome])
    return Coordinator(retrieval, extraction, verification, checkpoint_dir=checkpoint_dir)


# ---------- the 3 required scenarios ----------


def test_scenario_claim_is_consistent(tmp_path):
    outcome = VerificationOutcome(
        verdict=VERDICT_CONSISTENT, explanation="Matches EDGAR data.", citations=["0000000001-26-000001"]
    )
    report = make_coordinator(outcome, tmp_path).run("ACME")

    assert len(report.verified_claims) == 1
    vc = report.verified_claims[0]
    assert vc.verdict == VERDICT_CONSISTENT
    assert vc.source == CHUNK
    assert vc.extracted == CLAIM
    assert vc.citations == ["0000000001-26-000001"]


def test_scenario_claim_is_inconsistent(tmp_path):
    outcome = VerificationOutcome(
        verdict=VERDICT_INCONSISTENT, explanation="Does not match EDGAR data.", citations=["0000000001-26-000001"]
    )
    report = make_coordinator(outcome, tmp_path).run("ACME")

    assert len(report.verified_claims) == 1
    assert report.verified_claims[0].verdict == VERDICT_INCONSISTENT


def test_scenario_claim_cannot_be_verified_missing_data(tmp_path):
    outcome = VerificationOutcome(
        verdict=VERDICT_UNVERIFIABLE, explanation="No reported data for this period.", citations=[]
    )
    report = make_coordinator(outcome, tmp_path).run("ACME")

    assert len(report.verified_claims) == 1
    vc = report.verified_claims[0]
    assert vc.verdict == VERDICT_UNVERIFIABLE
    assert vc.citations == []


# ---------- end-to-end: the real "compared with" bug, through the real pipeline ----------
#
# Real bug, found via research/specificity_check.py against TXN's actual FY2025
# 10-K. Uses the real RealVerificationAgent (no LLM - extraction is mocked with
# the exact real quote), so this exercises Coordinator.run's actual occurrence-
# numbering logic end to end, not just RealVerificationAgent in isolation.


def _txn_fact(value, period_start, period_end):
    return FinancialFact(
        ticker="TXN", cik="0000097476", concept="NetIncomeLoss", label="Net Income (Loss)",
        value=value, unit="USD", period_start=period_start, period_end=period_end,
        fiscal_year=2025, fiscal_period="FY", form="10-K", filed="2026-02-06", accession_number="a",
    )


def test_real_txn_compared_with_sentence_both_claims_verify_correctly(tmp_path):
    """Real quote from TXN's FY2025 10-K MD&A: 'Net income was $5.00 billion
    compared with $4.80 billion.' Extraction (confirmed against the live
    extractor, not mocked here) correctly produces two absolute claims with the
    right values; before this fix, both resolved against the same current-period
    fact and the true $4.80B prior-year claim came back inconsistent."""
    chunk = TextChunk(
        ticker="TXN", cik="0000097476", accession_number="a", form="10-K",
        filing_date="2026-02-06", section="MD&A", chunk_index=0,
        text="Net income was $5.00 billion compared with $4.80 billion.",
    )
    quote = "Net income was $5.00 billion compared with $4.80 billion."
    claims = [
        ExtractedClaim(metric="net income", value=5_000_000_000.0, value_unit="USD", period="", comparison_type="absolute", quote=quote),
        ExtractedClaim(metric="net income", value=4_800_000_000.0, value_unit="USD", period="", comparison_type="absolute", quote=quote),
    ]
    facts = [_txn_fact(5_000_000_000, "2025-01-01", "2025-12-31"), _txn_fact(4_800_000_000, "2024-01-01", "2024-12-31")]

    retrieval = MockRetrievalAgent(chunks=[chunk], facts=facts)
    extraction = MockExtractionAgent(canned={chunk.text: claims})
    report = Coordinator(retrieval, extraction, RealVerificationAgent(), checkpoint_dir=tmp_path).run("TXN")

    assert len(report.verified_claims) == 2
    assert [vc.verdict for vc in report.verified_claims] == [VERDICT_CONSISTENT, VERDICT_CONSISTENT]


def _operating_income_fact(value, period_start, period_end):
    return FinancialFact(
        ticker="TXN", cik="0000097476", concept="OperatingIncomeLoss", label="Operating Income (Loss)",
        value=value, unit="USD", period_start=period_start, period_end=period_end,
        fiscal_year=2025, fiscal_period="FY", form="10-K", filed="2026-02-06", accession_number="a",
    )


def test_real_txn_interleaved_dollar_and_percent_claims_pair_correctly(tmp_path):
    """Real quote from TXN's FY2025 10-K MD&A: 'Operating profit was $6.02
    billion, or 34.1% of revenue, compared with $5.47 billion, or 34.9% of
    revenue.' Confirmed against the live extractor (not mocked here): this
    produces 4 claims for the SAME metric text, interleaved USD/percent, and
    without different sub-quotes per claim - so occurrence must be numbered per
    (metric, value_unit), not (metric, quote), or the second $ claim keeps
    resolving against the current period instead of the prior one. The two
    percent claims are separately caught by the percent/dollar-concept guard
    (OperatingIncomeLoss is a dollar concept, not a ratio) - unverifiable, not
    wrong, either way."""
    chunk = TextChunk(
        ticker="TXN", cik="0000097476", accession_number="a", form="10-K",
        filing_date="2026-02-06", section="MD&A", chunk_index=0,
        text="Operating profit was $6.02 billion, or 34.1% of revenue, compared with $5.47 billion, or 34.9% of revenue.",
    )
    claims = [
        ExtractedClaim(metric="Operating profit", value=6_020_000_000.0, value_unit="USD", period="", comparison_type="absolute", quote="Operating profit was $6.02 billion, or 34.1% of revenue, compared with $5.47 billion, or 34.9% of revenue."),
        ExtractedClaim(metric="Operating profit", value=5_470_000_000.0, value_unit="USD", period="", comparison_type="absolute", quote="compared with $5.47 billion"),
        ExtractedClaim(metric="Operating profit", value=34.1, value_unit="percent", period="", comparison_type="absolute", quote="or 34.1% of revenue"),
        ExtractedClaim(metric="Operating profit", value=34.9, value_unit="percent", period="", comparison_type="absolute", quote="or 34.9% of revenue"),
    ]
    facts = [
        _operating_income_fact(6_020_000_000, "2025-01-01", "2025-12-31"),
        _operating_income_fact(5_470_000_000, "2024-01-01", "2024-12-31"),
    ]

    retrieval = MockRetrievalAgent(chunks=[chunk], facts=facts)
    extraction = MockExtractionAgent(canned={chunk.text: claims})
    report = Coordinator(retrieval, extraction, RealVerificationAgent(), checkpoint_dir=tmp_path).run("TXN")

    verdicts = [vc.verdict for vc in report.verified_claims]
    assert verdicts == [VERDICT_CONSISTENT, VERDICT_CONSISTENT, VERDICT_UNVERIFIABLE, VERDICT_UNVERIFIABLE]


# ---------- routing / aggregation correctness ----------


def test_chunk_with_no_claims_is_skipped_not_errored(tmp_path):
    empty_chunk = TextChunk(
        ticker="ACME", cik="0000000001", accession_number="a", form="10-K",
        filing_date="2026-01-01", section="MD&A", chunk_index=1, text="Boilerplate with no numbers.",
    )
    retrieval = MockRetrievalAgent(chunks=[empty_chunk], facts=[])
    extraction = MockExtractionAgent(canned={})  # no claims for any paragraph
    verification = MockVerificationAgent(outcomes=[])
    report = Coordinator(retrieval, extraction, verification, checkpoint_dir=tmp_path).run("ACME")

    assert report.verified_claims == []
    decisions = [e for e in report.trace if e.kind == "decision" and e.action == "skip_empty_chunk"]
    assert len(decisions) == 1


def test_multiple_chunks_mixed_verdicts_all_captured(tmp_path):
    chunk_a = TextChunk(
        ticker="ACME", cik="0000000001", accession_number="a", form="10-K",
        filing_date="2026-01-01", section="MD&A", chunk_index=0, text="Revenue was $100.",
    )
    chunk_b = TextChunk(
        ticker="ACME", cik="0000000001", accession_number="a", form="10-K",
        filing_date="2026-01-01", section="MD&A", chunk_index=1, text="Margin was 40%.",
    )
    claim_a = ExtractedClaim(metric="Revenue", value=100, value_unit="USD", period="", comparison_type="absolute", quote="")
    claim_b = ExtractedClaim(metric="Margin", value=40, value_unit="percent", period="", comparison_type="absolute", quote="")

    retrieval = MockRetrievalAgent(chunks=[chunk_a, chunk_b], facts=[])
    extraction = MockExtractionAgent(canned={chunk_a.text: [claim_a], chunk_b.text: [claim_b]})
    verification = MockVerificationAgent(
        outcomes=[
            VerificationOutcome(verdict=VERDICT_CONSISTENT, explanation="", citations=["a"]),
            VerificationOutcome(verdict=VERDICT_UNVERIFIABLE, explanation="no data", citations=[]),
        ]
    )
    report = Coordinator(retrieval, extraction, verification, checkpoint_dir=tmp_path).run("ACME")

    assert len(report.verified_claims) == 2
    verdicts = {vc.verdict for vc in report.verified_claims}
    assert verdicts == {VERDICT_CONSISTENT, VERDICT_UNVERIFIABLE}


def test_report_summary_counts_tool_calls_and_decisions_and_verdicts(tmp_path):
    outcome = VerificationOutcome(verdict=VERDICT_CONSISTENT, explanation="", citations=["x"])
    report = make_coordinator(outcome, tmp_path).run("ACME")
    summary = report.summary()

    assert summary["ticker"] == "ACME"
    assert summary["n_claims"] == 1
    assert summary["by_verdict"] == {VERDICT_CONSISTENT: 1}
    # 2 retrieval calls (chunks, facts) + 1 extraction call + 1 verification call
    assert summary["n_tool_calls"] == 4
    assert summary["tool_calls_by_agent"]["retrieval"] == 2
    assert summary["tool_calls_by_agent"]["extraction"] == 1
    assert summary["tool_calls_by_agent"]["verification"] == 1
    assert summary["time_to_first_action_seconds"] is not None
    assert summary["n_reasoning_steps"] >= 3  # chunks_retrieved, facts_retrieved, claim_verdict
    assert summary["partial"] is False


def test_empty_ticker_no_chunks_produces_empty_report_not_a_crash(tmp_path):
    retrieval = MockRetrievalAgent(chunks=[], facts=[])
    extraction = MockExtractionAgent(canned={})
    verification = MockVerificationAgent(outcomes=[])
    report = Coordinator(retrieval, extraction, verification, checkpoint_dir=tmp_path).run("EMPTY")

    assert report.verified_claims == []
    assert report.summary()["n_claims"] == 0
