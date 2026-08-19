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
"""

from agents.extraction_agent import ExtractionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.schema import Report, Tracer, VerifiedClaim
from agents.verification_agent import VerificationAgent


class Coordinator:
    def __init__(self, retrieval: RetrievalAgent, extraction: ExtractionAgent, verification: VerificationAgent):
        self.retrieval = retrieval
        self.extraction = extraction
        self.verification = verification

    def run(self, ticker: str, form_type: str = "10-K", limit: int = 1) -> Report:
        tracer = Tracer()
        report = Report(ticker=ticker)

        tracer.record("tool_call", "retrieval", "get_mdna_chunks", f"{ticker} {form_type} limit={limit}")
        chunks = self.retrieval.get_mdna_chunks(ticker, form_type=form_type, limit=limit)
        tracer.record("decision", "coordinator", "chunks_retrieved", f"{len(chunks)} MD&A chunks")

        tracer.record("tool_call", "retrieval", "get_facts", ticker)
        facts = self.retrieval.get_facts(ticker)
        tracer.record("decision", "coordinator", "facts_retrieved", f"{len(facts)} XBRL facts")

        for chunk in chunks:
            tracer.record("tool_call", "extraction", "extract", f"chunk {chunk.chunk_index}")
            claims = self.extraction.extract(chunk.text)

            if not claims:
                tracer.record(
                    "decision", "coordinator", "skip_empty_chunk", f"chunk {chunk.chunk_index} yielded no claims"
                )
                continue

            for claim in claims:
                tracer.record("tool_call", "verification", "verify", f"{claim.metric}={claim.value}{claim.value_unit}")
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

        report.trace = tracer.events
        return report
