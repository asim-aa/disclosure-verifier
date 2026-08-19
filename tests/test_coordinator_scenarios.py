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
from agents.verification_agent import MockVerificationAgent
from eval.schema import ExtractedClaim
from tools.schema import (
    VERDICT_CONSISTENT,
    VERDICT_INCONSISTENT,
    VERDICT_UNVERIFIABLE,
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


def make_coordinator(outcome: VerificationOutcome) -> Coordinator:
    retrieval = MockRetrievalAgent(chunks=[CHUNK], facts=[])
    extraction = MockExtractionAgent(canned={CHUNK.text: [CLAIM]})
    verification = MockVerificationAgent(outcomes=[outcome])
    return Coordinator(retrieval, extraction, verification)


# ---------- the 3 required scenarios ----------


def test_scenario_claim_is_consistent():
    outcome = VerificationOutcome(
        verdict=VERDICT_CONSISTENT, explanation="Matches EDGAR data.", citations=["0000000001-26-000001"]
    )
    report = make_coordinator(outcome).run("ACME")

    assert len(report.verified_claims) == 1
    vc = report.verified_claims[0]
    assert vc.verdict == VERDICT_CONSISTENT
    assert vc.source == CHUNK
    assert vc.extracted == CLAIM
    assert vc.citations == ["0000000001-26-000001"]


def test_scenario_claim_is_inconsistent():
    outcome = VerificationOutcome(
        verdict=VERDICT_INCONSISTENT, explanation="Does not match EDGAR data.", citations=["0000000001-26-000001"]
    )
    report = make_coordinator(outcome).run("ACME")

    assert len(report.verified_claims) == 1
    assert report.verified_claims[0].verdict == VERDICT_INCONSISTENT


def test_scenario_claim_cannot_be_verified_missing_data():
    outcome = VerificationOutcome(
        verdict=VERDICT_UNVERIFIABLE, explanation="No reported data for this period.", citations=[]
    )
    report = make_coordinator(outcome).run("ACME")

    assert len(report.verified_claims) == 1
    vc = report.verified_claims[0]
    assert vc.verdict == VERDICT_UNVERIFIABLE
    assert vc.citations == []


# ---------- routing / aggregation correctness ----------


def test_chunk_with_no_claims_is_skipped_not_errored():
    empty_chunk = TextChunk(
        ticker="ACME", cik="0000000001", accession_number="a", form="10-K",
        filing_date="2026-01-01", section="MD&A", chunk_index=1, text="Boilerplate with no numbers.",
    )
    retrieval = MockRetrievalAgent(chunks=[empty_chunk], facts=[])
    extraction = MockExtractionAgent(canned={})  # no claims for any paragraph
    verification = MockVerificationAgent(outcomes=[])
    report = Coordinator(retrieval, extraction, verification).run("ACME")

    assert report.verified_claims == []
    decisions = [e for e in report.trace if e.kind == "decision" and e.action == "skip_empty_chunk"]
    assert len(decisions) == 1


def test_multiple_chunks_mixed_verdicts_all_captured():
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
    report = Coordinator(retrieval, extraction, verification).run("ACME")

    assert len(report.verified_claims) == 2
    verdicts = {vc.verdict for vc in report.verified_claims}
    assert verdicts == {VERDICT_CONSISTENT, VERDICT_UNVERIFIABLE}


def test_report_summary_counts_tool_calls_and_decisions_and_verdicts():
    outcome = VerificationOutcome(verdict=VERDICT_CONSISTENT, explanation="", citations=["x"])
    report = make_coordinator(outcome).run("ACME")
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


def test_empty_ticker_no_chunks_produces_empty_report_not_a_crash():
    retrieval = MockRetrievalAgent(chunks=[], facts=[])
    extraction = MockExtractionAgent(canned={})
    verification = MockVerificationAgent(outcomes=[])
    report = Coordinator(retrieval, extraction, verification).run("EMPTY")

    assert report.verified_claims == []
    assert report.summary()["n_claims"] == 0
