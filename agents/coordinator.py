"""The hierarchical coordinator (Pillar 3): routes a ticker through retrieval ->
extraction -> verification and aggregates the result into a Report.

Justification for hierarchical routing over a single ReAct loop (per the
proposal): each stage has a different tool dependency (EDGAR HTTP calls, an LLM
call, arithmetic against already-fetched facts) and a well-defined
input/output contract, so the sequence itself doesn't need to be *decided* by an
LLM — what needs deciding is what to do with each stage's result (skip a chunk
with no claims, mark a claim unresolvable, stop verifying once data runs out).
Those are the coordinator's actual routing decisions, and they're what the trace
records separately from tool calls.

Agents are dependency-injected (not constructed here) specifically so mock
agents can be swapped in for scenario testing without touching this class at
all — see tests/test_coordinator_scenarios.py.

Budget and checkpointing exist for the same reason: a real 10-K can have 100+
MD&A chunks, each needing its own LLM call. Without a budget, one run() call has
no cap on cost or time. Without a checkpoint, a crash (or an intentional budget
stop) at chunk 90 of 126 loses all 90 chunks' work on the next attempt.
"""

import time

from agents import checkpoint as checkpoint_store
from agents.checkpoint import CHECKPOINT_DIR
from agents.extraction_agent import ExtractionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.schema import Budget, Report, Tracer, VerifiedClaim
from agents.verification_agent import VerificationAgent


class Coordinator:
    def __init__(
        self,
        retrieval: RetrievalAgent,
        extraction: ExtractionAgent,
        verification: VerificationAgent,
        checkpoint_dir=CHECKPOINT_DIR,
    ):
        self.retrieval = retrieval
        self.extraction = extraction
        self.verification = verification
        self.checkpoint_dir = checkpoint_dir

    def run(
        self,
        ticker: str,
        form_type: str = "10-K",
        limit: int = 1,
        budget: Budget | None = None,
        resume: bool = True,
    ) -> Report:
        run_start = time.monotonic()

        chunks = self.retrieval.get_mdna_chunks(ticker, form_type=form_type, limit=limit)
        facts = self.retrieval.get_facts(ticker)
        accession_numbers = sorted({c.accession_number for c in chunks})

        checkpointed = None
        if resume:
            checkpointed = checkpoint_store.load(
                ticker, form_type, limit, accession_numbers, checkpoint_dir=self.checkpoint_dir
            )

        if checkpointed:
            processed = checkpointed["processed_chunk_indices"]
            report = Report(ticker=ticker, verified_claims=checkpointed["verified_claims"])
            tracer = Tracer(start_step=len(checkpointed["trace"]), existing_events=checkpointed["trace"])
            tracer.record(
                "decision", "coordinator", "resumed_from_checkpoint",
                f"{len(processed)} of {len(chunks)} chunks already processed",
            )
        else:
            processed: set[int] = set()
            report = Report(ticker=ticker)
            tracer = Tracer()
            tracer.record("tool_call", "retrieval", "get_mdna_chunks", f"{ticker} {form_type} limit={limit}")
            tracer.record("decision", "coordinator", "chunks_retrieved", f"{len(chunks)} MD&A chunks")
            tracer.record("tool_call", "retrieval", "get_facts", ticker)
            tracer.record("decision", "coordinator", "facts_retrieved", f"{len(facts)} XBRL facts")

        n_chunks_this_call = 0
        n_extraction_calls_this_call = 0
        stopped_early = False

        for chunk in chunks:
            if chunk.chunk_index in processed:
                continue

            if budget is not None:
                elapsed = time.monotonic() - run_start
                reason = budget.exceeded_reason(n_chunks_this_call, n_extraction_calls_this_call, elapsed)
                if reason is not None:
                    tracer.record("decision", "coordinator", "budget_exceeded", reason)
                    report.partial = True
                    report.partial_reason = reason
                    stopped_early = True
                    break

            tracer.record("tool_call", "extraction", "extract", f"chunk {chunk.chunk_index}")
            claims = self.extraction.extract(chunk.text)
            n_extraction_calls_this_call += 1

            if not claims:
                tracer.record(
                    "decision", "coordinator", "skip_empty_chunk", f"chunk {chunk.chunk_index} yielded no claims"
                )
            else:
                for claim in claims:
                    tracer.record(
                        "tool_call", "verification", "verify", f"{claim.metric}={claim.value}{claim.value_unit}"
                    )
                    outcome = self.verification.verify(claim, ticker, facts, as_of=chunk.filing_date)
                    report.verified_claims.append(
                        VerifiedClaim(
                            source=chunk,
                            extracted=claim,
                            verdict=outcome.verdict,
                            explanation=outcome.explanation,
                            citations=outcome.citations,
                            reconciliation=outcome.reconciliation,
                        )
                    )
                    tracer.record("decision", "coordinator", "claim_verdict", f"{claim.metric} -> {outcome.verdict}")

            processed.add(chunk.chunk_index)
            n_chunks_this_call += 1
            report.trace = tracer.events
            checkpoint_store.save(
                ticker, form_type, limit, accession_numbers, sorted(processed), report,
                checkpoint_dir=self.checkpoint_dir,
            )

        report.trace = tracer.events

        if not stopped_early:
            checkpoint_store.delete(ticker, form_type, limit, checkpoint_dir=self.checkpoint_dir)

        return report
