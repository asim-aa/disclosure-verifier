"""Schema for the orchestration layer: what the coordinator produces (a Report of
VerifiedClaims) and how its routing decisions get recorded (a trace of TraceEvents)
for the tool-call-count / reasoning-step / time-to-first-action reporting the
proposal's Pillar 3 section calls for.
"""

import time
from dataclasses import asdict, dataclass, field

from eval.schema import ExtractedClaim
from tools.schema import ReconciliationResult, TextChunk


@dataclass(frozen=True)
class TraceEvent:
    """One step the coordinator took, in order. `kind` distinguishes an actual
    tool/agent call from a routing decision made about what to do with its result
    (e.g. "skip this chunk, extraction returned nothing" is a decision, not a
    call) — the two are counted separately in the summary."""

    step: int
    kind: str  # "tool_call" | "decision"
    agent: str  # "retrieval" | "extraction" | "verification" | "coordinator"
    action: str
    detail: str
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class VerificationOutcome:
    """What the verification agent produces for one ExtractedClaim — everything
    the coordinator needs except which TextChunk it came from, which only the
    coordinator (not the verification agent) knows."""

    verdict: str  # consistent | inconsistent | unverifiable
    explanation: str
    citations: list[str]
    reconciliation: ReconciliationResult | None = None


@dataclass(frozen=True)
class VerifiedClaim:
    """One extracted claim carried all the way through verification (or as far as
    it could get — `verdict` covers the full range including resolution failures,
    not just the reconciler's three verdicts)."""

    source: TextChunk
    extracted: ExtractedClaim
    verdict: str  # consistent | inconsistent | unverifiable
    explanation: str
    citations: list[str]
    reconciliation: ReconciliationResult | None = None

    def to_dict(self) -> dict:
        return {
            "source": self.source.to_dict(),
            "extracted": self.extracted.model_dump(),
            "verdict": self.verdict,
            "explanation": self.explanation,
            "citations": self.citations,
            "reconciliation": self.reconciliation.to_dict() if self.reconciliation else None,
        }


@dataclass(frozen=True)
class Budget:
    """A cap on how much of a Coordinator.run() is allowed to happen before it
    stops early and returns a partial Report rather than continuing unbounded.
    Every field is optional; unset fields impose no limit."""

    max_chunks: int | None = None  # MD&A chunks processed (extraction attempted)
    max_extraction_calls: int | None = None  # LLM calls specifically — the costly ones
    max_seconds: float | None = None  # wall-clock time for this run() invocation

    def exceeded_reason(self, n_chunks: int, n_extraction_calls: int, elapsed_seconds: float) -> str | None:
        """None if still within budget, else a human-readable reason it's not."""
        if self.max_chunks is not None and n_chunks >= self.max_chunks:
            return f"max_chunks ({self.max_chunks}) reached"
        if self.max_extraction_calls is not None and n_extraction_calls >= self.max_extraction_calls:
            return f"max_extraction_calls ({self.max_extraction_calls}) reached"
        if self.max_seconds is not None and elapsed_seconds >= self.max_seconds:
            return f"max_seconds ({self.max_seconds}) reached"
        return None


@dataclass
class Report:
    """The coordinator's final output for one ticker: every claim it verified (or
    tried to), plus the trace of how it got there. `partial` is True when a
    Budget cut the run short — the report reflects everything processed up to
    that point, not the whole document."""

    ticker: str
    verified_claims: list[VerifiedClaim] = field(default_factory=list)
    trace: list[TraceEvent] = field(default_factory=list)
    partial: bool = False
    partial_reason: str | None = None

    def summary(self) -> dict:
        tool_calls = [e for e in self.trace if e.kind == "tool_call"]
        decisions = [e for e in self.trace if e.kind == "decision"]
        by_verdict: dict[str, int] = {}
        for vc in self.verified_claims:
            by_verdict[vc.verdict] = by_verdict.get(vc.verdict, 0) + 1

        return {
            "ticker": self.ticker,
            "n_claims": len(self.verified_claims),
            "by_verdict": by_verdict,
            "n_tool_calls": len(tool_calls),
            "tool_calls_by_agent": _count_by(tool_calls, lambda e: e.agent),
            "n_reasoning_steps": len(decisions),
            "time_to_first_action_seconds": self.trace[0].elapsed_seconds if self.trace else None,
            "total_elapsed_seconds": self.trace[-1].elapsed_seconds if self.trace else None,
            "partial": self.partial,
            "partial_reason": self.partial_reason,
        }

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "verified_claims": [vc.to_dict() for vc in self.verified_claims],
            "trace": [e.to_dict() for e in self.trace],
            "partial": self.partial,
            "partial_reason": self.partial_reason,
            "summary": self.summary(),
        }


def _count_by(events: list[TraceEvent], key) -> dict:
    counts: dict[str, int] = {}
    for e in events:
        k = key(e)
        counts[k] = counts.get(k, 0) + 1
    return counts


class Tracer:
    """Accumulates TraceEvents with monotonic elapsed time from construction.

    `start_step` and `existing_events` support resuming after a checkpoint load:
    step numbers keep incrementing across the resumed run rather than restarting
    at 0, but elapsed_seconds necessarily restarts from 0 for the new process —
    the original run's wall-clock died with that process. Events from before the
    resume keep their original (now historical) elapsed_seconds unchanged."""

    def __init__(self, start_step: int = 0, existing_events: list[TraceEvent] | None = None):
        self._start = time.monotonic()
        self._events: list[TraceEvent] = list(existing_events) if existing_events else []
        self._next_step = start_step

    def record(self, kind: str, agent: str, action: str, detail: str = "") -> None:
        self._events.append(
            TraceEvent(
                step=self._next_step,
                kind=kind,
                agent=agent,
                action=action,
                detail=detail,
                elapsed_seconds=time.monotonic() - self._start,
            )
        )
        self._next_step += 1

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)
