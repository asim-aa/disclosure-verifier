"""Tests for Coordinator's budget (stop-rule) and checkpoint/resume behavior —
the two "harness" gaps identified when comparing this project against a general
agent-harness checklist: no cap on a run's cost, and no durable state if a long
run crashes or is stopped partway through."""

from agents import checkpoint as checkpoint_store
from agents.coordinator import Coordinator
from agents.extraction_agent import MockExtractionAgent
from agents.retrieval_agent import MockRetrievalAgent
from agents.schema import Budget, VerificationOutcome
from agents.verification_agent import MockVerificationAgent
from eval.schema import ExtractedClaim
from tools.schema import VERDICT_CONSISTENT, TextChunk


def make_chunk(i, accn="a", text=None):
    return TextChunk(
        ticker="ACME", cik="0000000001", accession_number=accn, form="10-K",
        filing_date="2026-01-01", section="MD&A", chunk_index=i, text=text or f"Revenue was ${i}00.",
    )


def make_claim(i):
    return ExtractedClaim(metric="Revenue", value=i * 100, value_unit="USD", period="", comparison_type="absolute", quote="")


class CountingExtractionAgent:
    """Wraps MockExtractionAgent to record how many times extract() was actually
    called — the thing a real budget/checkpoint is protecting (LLM calls)."""

    def __init__(self, canned: dict):
        self._inner = MockExtractionAgent(canned)
        self.call_count = 0

    def extract(self, paragraph: str):
        self.call_count += 1
        return self._inner.extract(paragraph)


def make_pipeline(n_chunks: int, checkpoint_dir):
    chunks = [make_chunk(i) for i in range(n_chunks)]
    canned = {c.text: [make_claim(i)] for i, c in enumerate(chunks)}
    retrieval = MockRetrievalAgent(chunks=chunks, facts=[])
    extraction = CountingExtractionAgent(canned)
    outcomes = [VerificationOutcome(verdict=VERDICT_CONSISTENT, explanation="", citations=["a"]) for _ in range(n_chunks)]
    verification = MockVerificationAgent(outcomes=outcomes)
    coordinator = Coordinator(retrieval, extraction, verification, checkpoint_dir=checkpoint_dir)
    return coordinator, extraction, chunks


# ---------- budget ----------


def test_no_budget_processes_all_chunks(tmp_path):
    coordinator, extraction, _ = make_pipeline(5, tmp_path)
    report = coordinator.run("ACME")

    assert len(report.verified_claims) == 5
    assert extraction.call_count == 5
    assert report.partial is False
    assert report.partial_reason is None


def test_max_chunks_budget_stops_early_and_marks_partial(tmp_path):
    coordinator, extraction, _ = make_pipeline(5, tmp_path)
    report = coordinator.run("ACME", budget=Budget(max_chunks=2))

    assert extraction.call_count == 2
    assert len(report.verified_claims) == 2
    assert report.partial is True
    assert "max_chunks" in report.partial_reason


def test_max_extraction_calls_budget_stops_early(tmp_path):
    coordinator, extraction, _ = make_pipeline(5, tmp_path)
    report = coordinator.run("ACME", budget=Budget(max_extraction_calls=3))

    assert extraction.call_count == 3
    assert report.partial is True
    assert "max_extraction_calls" in report.partial_reason


def test_max_seconds_zero_stops_before_any_chunk(tmp_path):
    coordinator, extraction, _ = make_pipeline(5, tmp_path)
    report = coordinator.run("ACME", budget=Budget(max_seconds=0))

    assert extraction.call_count == 0
    assert report.verified_claims == []
    assert report.partial is True
    assert "max_seconds" in report.partial_reason


def test_budget_exceeded_is_recorded_in_trace(tmp_path):
    coordinator, _, _ = make_pipeline(5, tmp_path)
    report = coordinator.run("ACME", budget=Budget(max_chunks=1))

    budget_events = [e for e in report.trace if e.action == "budget_exceeded"]
    assert len(budget_events) == 1
    assert budget_events[0].kind == "decision"


# ---------- checkpoint / resume ----------


def test_checkpoint_file_created_when_budget_stops_early(tmp_path):
    coordinator, _, _ = make_pipeline(5, tmp_path)
    coordinator.run("ACME", budget=Budget(max_chunks=2))

    assert (tmp_path / "ACME_10-K_1.json").exists()


def test_checkpoint_deleted_after_full_completion(tmp_path):
    coordinator, _, _ = make_pipeline(3, tmp_path)
    coordinator.run("ACME")  # no budget — runs to completion

    assert not (tmp_path / "ACME_10-K_1.json").exists()


def test_resume_continues_without_reprocessing_done_chunks(tmp_path):
    coordinator, extraction, _ = make_pipeline(5, tmp_path)

    first = coordinator.run("ACME", budget=Budget(max_chunks=2))
    assert first.partial is True
    assert extraction.call_count == 2

    second = coordinator.run("ACME")  # no budget this time — finish the rest
    assert second.partial is False
    # only the 3 NOT already processed should have triggered new extract() calls
    assert extraction.call_count == 2 + 3

    assert len(second.verified_claims) == 5
    values = {vc.extracted.value for vc in second.verified_claims}
    assert values == {0, 100, 200, 300, 400}


def test_resume_preserves_trace_step_numbering(tmp_path):
    coordinator, _, _ = make_pipeline(5, tmp_path)
    first = coordinator.run("ACME", budget=Budget(max_chunks=2))
    second = coordinator.run("ACME")

    steps = [e.step for e in second.trace]
    assert steps == sorted(steps)  # strictly increasing, no resets or collisions
    assert len(steps) == len(set(steps))  # no duplicate step numbers
    assert len(second.trace) > len(first.trace)


def test_resume_false_ignores_checkpoint_and_starts_fresh(tmp_path):
    # resume=False re-processes all 5 chunks on the second call, on top of the 2
    # already done in the first — the mock verification agent needs an outcome
    # ready for every one of those 7 total verify() calls across both runs.
    chunks = [make_chunk(i) for i in range(5)]
    canned = {c.text: [make_claim(i)] for i, c in enumerate(chunks)}
    retrieval = MockRetrievalAgent(chunks=chunks, facts=[])
    extraction = CountingExtractionAgent(canned)
    outcomes = [VerificationOutcome(verdict=VERDICT_CONSISTENT, explanation="", citations=["a"]) for _ in range(7)]
    verification = MockVerificationAgent(outcomes=outcomes)
    coordinator = Coordinator(retrieval, extraction, verification, checkpoint_dir=tmp_path)

    coordinator.run("ACME", budget=Budget(max_chunks=2))
    assert extraction.call_count == 2

    report = coordinator.run("ACME", resume=False)
    # started fresh: all 5 chunks re-extracted, not just the remaining 3
    assert extraction.call_count == 2 + 5
    assert len(report.verified_claims) == 5
    assert report.partial is False


def test_stale_checkpoint_discarded_when_accession_numbers_differ(tmp_path):
    coordinator, _extraction, _chunks = make_pipeline(5, tmp_path)
    coordinator.run("ACME", budget=Budget(max_chunks=2))
    assert checkpoint_store._checkpoint_path("ACME", "10-K", 1, tmp_path).exists()

    # a fresh retrieval where the underlying filing has changed (different accession)
    new_chunks = [make_chunk(i, accn="different-filing") for i in range(3)]
    new_canned = {c.text: [make_claim(i)] for i, c in enumerate(new_chunks)}
    new_retrieval = MockRetrievalAgent(chunks=new_chunks, facts=[])
    new_extraction = CountingExtractionAgent(new_canned)
    new_verification = MockVerificationAgent(
        outcomes=[VerificationOutcome(verdict=VERDICT_CONSISTENT, explanation="", citations=["a"]) for _ in range(3)]
    )
    new_coordinator = Coordinator(new_retrieval, new_extraction, new_verification, checkpoint_dir=tmp_path)

    report = new_coordinator.run("ACME")

    # stale checkpoint discarded -> all 3 new chunks processed fresh, not skipped
    assert new_extraction.call_count == 3
    assert len(report.verified_claims) == 3


def test_checkpoint_survives_reconstruction_with_reconciliation_result(tmp_path):
    """Round-trip a VerifiedClaim that actually carries a ReconciliationResult
    (not just a bare VerificationOutcome) through save/load. Uses 2 chunks with
    max_chunks=1 so the run genuinely stops early (a single-chunk run would just
    complete normally within budget and delete its checkpoint on exit)."""
    from agents.verification_agent import RealVerificationAgent
    from tools.schema import FinancialFact

    facts = [
        FinancialFact(
            ticker="ACME", cik="0000000001", concept="Revenues", label="Revenues", value=100,
            unit="USD", period_start="2025-01-01", period_end="2026-01-01", fiscal_year=2026,
            fiscal_period="FY", form="10-K", filed="2026-01-01", accession_number="a",
        )
    ]
    chunk_0 = make_chunk(0, text="Revenue chunk zero.")
    chunk_1 = make_chunk(1, text="Revenue chunk one.")
    claim = ExtractedClaim(metric="revenue", value=100, value_unit="USD", period="", comparison_type="absolute", quote="")
    retrieval = MockRetrievalAgent(chunks=[chunk_0, chunk_1], facts=facts)
    extraction = CountingExtractionAgent({chunk_0.text: [claim], chunk_1.text: [claim]})
    verification = RealVerificationAgent()
    coordinator = Coordinator(retrieval, extraction, verification, checkpoint_dir=tmp_path)

    report = coordinator.run("ACME", budget=Budget(max_chunks=1))
    assert report.partial is True
    assert extraction.call_count == 1

    loaded = checkpoint_store.load("ACME", "10-K", 1, ["a"], checkpoint_dir=tmp_path)
    restored_claim = loaded["verified_claims"][0]
    assert restored_claim.verdict == VERDICT_CONSISTENT
    assert restored_claim.reconciliation is not None
    assert restored_claim.reconciliation.claim.metric == "Revenues"
