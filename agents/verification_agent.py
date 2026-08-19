"""Verification agent: resolves an ExtractedClaim's free-text metric/period into
a tools.schema.Claim, then reuses Phase 3's reconciler exactly as-is (the same
logic Phase 7 reuses as the RLVR reward function — nothing about it changes here).
"""

from typing import Protocol

from agents.resolver import resolve_concept, resolve_periods
from agents.schema import VerificationOutcome
from eval.schema import ExtractedClaim
from tools.reconciler import reconcile
from tools.schema import VERDICT_UNVERIFIABLE, Claim, FinancialFact


class VerificationAgent(Protocol):
    def verify(
        self, extracted: ExtractedClaim, ticker: str, facts: list[FinancialFact], as_of: str | None = None
    ) -> VerificationOutcome: ...


class RealVerificationAgent:
    def verify(
        self, extracted: ExtractedClaim, ticker: str, facts: list[FinancialFact], as_of: str | None = None
    ) -> VerificationOutcome:
        """`as_of` should be the filing_date of the filing the claim's source
        paragraph came from — see resolve_periods for why this cutoff matters
        (without it, a claim from an older 10-K can get compared against a
        newer 10-Q's quarterly figures instead of that 10-K's own annual ones)."""
        concept = resolve_concept(extracted.metric, facts, as_of=as_of)
        if concept is None:
            return VerificationOutcome(
                verdict=VERDICT_UNVERIFIABLE,
                explanation=f"Could not resolve metric '{extracted.metric}' to a known XBRL concept.",
                citations=[],
            )

        try:
            current_fact, comparison_fact = resolve_periods(facts, concept, as_of=as_of)
        except ValueError as exc:
            return VerificationOutcome(verdict=VERDICT_UNVERIFIABLE, explanation=str(exc), citations=[])

        needs_comparison = extracted.comparison_type != "absolute"
        if needs_comparison and comparison_fact is None:
            return VerificationOutcome(
                verdict=VERDICT_UNVERIFIABLE,
                explanation=f"No comparison period available for '{concept}' to check a {extracted.comparison_type} claim.",
                citations=[],
            )

        # XBRL doesn't use "percent" as a unit — rate/ratio concepts (e.g. effective
        # tax rate) are tagged "pure" and store a decimal fraction (0.19, not 19).
        # Claims extracted from prose always state percentages the human way (19),
        # so convert to match the fact's actual representation before comparing.
        # Only applies to "absolute" claims: growth_pct/bps_change/absolute_change
        # are always computed and compared in percent/bps/dollars by the reconciler
        # itself, regardless of what unit the underlying fact happens to use.
        claimed_value = extracted.value
        if extracted.comparison_type == "absolute" and current_fact.unit == "pure" and extracted.value_unit == "percent":
            claimed_value = extracted.value / 100.0

        claim = Claim(
            ticker=ticker,
            metric=concept,
            comparison_type=extracted.comparison_type,
            claimed_value=claimed_value,
            period_start=current_fact.period_start,
            period_end=current_fact.period_end,
            comparison_period_start=comparison_fact.period_start if comparison_fact else None,
            comparison_period_end=comparison_fact.period_end if comparison_fact else None,
            unit=current_fact.unit,
        )

        result = reconcile(claim, facts)
        return VerificationOutcome(
            verdict=result.verdict,
            explanation=result.explanation,
            citations=result.citations,
            reconciliation=result,
        )


class MockVerificationAgent:
    """Returns a canned VerificationOutcome per ExtractedClaim (matched by identity
    via a list index, since ExtractedClaim isn't hashable in a stable way across
    equal-but-distinct instances) — set up by the test, not looked up dynamically."""

    def __init__(self, outcomes: list[VerificationOutcome]):
        self._outcomes = outcomes
        self._i = 0

    def verify(
        self, extracted: ExtractedClaim, ticker: str, facts: list[FinancialFact], as_of: str | None = None
    ) -> VerificationOutcome:
        outcome = self._outcomes[self._i]
        self._i += 1
        return outcome
