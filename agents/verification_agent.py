"""Verification agent: resolves an ExtractedClaim's free-text metric/period into
a tools.schema.Claim, then reuses Phase 3's reconciler (the same logic Phase 7
reuses as the RLVR reward function). Passes through the same `as_of` cutoff used
for period/concept resolution so fact-matching inside the reconciler stays
bitemporally consistent too — see tools/reconciler.py's `_find_fact` docstring.
"""

from typing import Protocol

from agents.resolver import is_relative_year_ago_hint, resolve_concept, resolve_periods
from agents.schema import VerificationOutcome
from eval.schema import ExtractedClaim
from tools.reconciler import reconcile
from tools.schema import VERDICT_UNVERIFIABLE, Claim, FinancialFact


class VerificationAgent(Protocol):
    def verify(
        self, extracted: ExtractedClaim, ticker: str, facts: list[FinancialFact], as_of: str | None = None,
        occurrence: int = 0,
    ) -> VerificationOutcome: ...


class RealVerificationAgent:
    def verify(
        self, extracted: ExtractedClaim, ticker: str, facts: list[FinancialFact], as_of: str | None = None,
        occurrence: int = 0,
    ) -> VerificationOutcome:
        """`as_of` should be the filing_date of the filing the claim's source
        paragraph came from — see resolve_periods for why this cutoff matters
        (without it, a claim from an older 10-K can get compared against a
        newer 10-Q's quarterly figures instead of that 10-K's own annual ones).

        `occurrence` numbers repeated (metric, quote) claims within one chunk
        (see Coordinator.run) — real bug, confirmed against TXN's actual MD&A
        (research/specificity_check.py): a sentence like "Net income was $5.00
        billion compared with $4.80 billion." extracts correctly as TWO absolute
        claims, but neither states an explicit period, so both independently
        resolve_periods to the SAME current fact — the true, accurate $4.80B
        prior-year claim then gets checked against the current year's number and
        wrongly flagged inconsistent. When occurrence == 1 (the second same-
        metric/same-quote absolute claim with no stated period) check it against
        the *comparison* fact resolve_periods already found for the first
        occurrence instead. occurrence >= 2 (a third+ repeat, not seen in any
        real chunk this project has inspected) falls back to default behavior —
        resolve_periods only returns one comparison fact, so there's no third
        period to check a third repeat against without deeper resolver changes,
        and guessing wrong would be worse than an honest, unfixed gap."""
        concept = resolve_concept(extracted.metric, facts, as_of=as_of)
        if concept is None:
            return VerificationOutcome(
                verdict=VERDICT_UNVERIFIABLE,
                explanation=f"Could not resolve metric '{extracted.metric}' to a known XBRL concept.",
                citations=[],
            )

        try:
            current_fact, comparison_fact = resolve_periods(facts, concept, as_of=as_of, period_hint=extracted.period)
        except ValueError as exc:
            return VerificationOutcome(verdict=VERDICT_UNVERIFIABLE, explanation=str(exc), citations=[])

        if occurrence == 1 and extracted.comparison_type == "absolute" and not extracted.period:
            if comparison_fact is None:
                return VerificationOutcome(
                    verdict=VERDICT_UNVERIFIABLE,
                    explanation=(
                        f"No prior-period fact available for '{concept}' to check this repeated "
                        "absolute claim (no stated period) against."
                    ),
                    citations=[],
                )
            current_fact, comparison_fact = comparison_fact, None

        # An absolute claim whose ENTIRE value describes the prior period, not a
        # comparison pair - real CRM case, confirmed against the live MD&A: "as
        # compared to diluted net income per share of $6.36 from a year ago." is
        # its own standalone chunk (the current-period sentence lives in a
        # different one), so there's no sibling claim for `occurrence` to pair
        # this with. `resolve_periods` already resolves the correct year-ago
        # fact as `comparison_fact` here - the hint "a year ago" is exactly what
        # its own comparison-period logic matches - so the only thing needed is
        # checking THIS claim's value against that fact instead of `current_fact`.
        # Distinct from the growth_pct/absolute_change case (excluded via the
        # comparison_type check): "revenue grew 14% year-over-year" also
        # contains "year-over-year", but there `current_fact` genuinely IS this
        # period and the hint correctly describes the delta reference, not the
        # claim's own period.
        elif extracted.comparison_type == "absolute" and is_relative_year_ago_hint(extracted.period):
            if comparison_fact is None:
                return VerificationOutcome(
                    verdict=VERDICT_UNVERIFIABLE,
                    explanation=(
                        f"No prior-period fact available for '{concept}' to check this "
                        f"'{extracted.period}' absolute claim against."
                    ),
                    citations=[],
                )
            current_fact, comparison_fact = comparison_fact, None

        needs_comparison = extracted.comparison_type != "absolute"
        if needs_comparison and comparison_fact is None:
            return VerificationOutcome(
                verdict=VERDICT_UNVERIFIABLE,
                explanation=f"No comparison period available for '{concept}' to check a {extracted.comparison_type} claim.",
                citations=[],
            )

        # A metric-text mapping can point at a real concept that's still the wrong
        # *kind* of fact for the claim - e.g. "gross profit margin" (a stated
        # percentage) mapped to GrossProfit, which XBRL reports as a raw dollar
        # figure, never a ratio. That's not a real inconsistency, it's a claim the
        # dictionary can't actually check: no numeric comparison between a percent
        # and a dollar amount can ever be meaningful. Confirmed against real TXN
        # data (research/specificity_check.py): "gross profit decreased to 57.0%
        # from 58.1%" claimed 57 against GrossProfit's real $10.083B, a 100%
        # "difference" that isn't a mismatch at all, just an incompatible unit.
        # A percent claim can only be checked against a "pure" (decimal-fraction)
        # fact - anything else means unverifiable, not automatically wrong.
        if extracted.comparison_type == "absolute" and extracted.value_unit == "percent" and current_fact.unit != "pure":
            return VerificationOutcome(
                verdict=VERDICT_UNVERIFIABLE,
                explanation=(
                    f"'{concept}' is reported in {current_fact.unit!r}, not a percentage/ratio — "
                    "can't check a percent claim against it."
                ),
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

        result = reconcile(claim, facts, as_of=as_of)
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
        self, extracted: ExtractedClaim, ticker: str, facts: list[FinancialFact], as_of: str | None = None,
        occurrence: int = 0,
    ) -> VerificationOutcome:
        outcome = self._outcomes[self._i]
        self._i += 1
        return outcome
