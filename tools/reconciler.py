"""Numerical Reconciler: checks whether a Claim is arithmetically consistent with
retrieved XBRL FinancialFacts.

This is the highest-stakes module in the project — Phase 7 reuses this exact logic
as the RLVR reward function, so correctness and honest failure (returning
"unverifiable" instead of guessing) matter more here than anywhere else in the
codebase.
"""

from tools.schema import (
    COMPARISON_ABSOLUTE,
    COMPARISON_ABSOLUTE_CHANGE,
    COMPARISON_BPS_CHANGE,
    COMPARISON_GROWTH_PCT,
    VERDICT_CONSISTENT,
    VERDICT_INCONSISTENT,
    VERDICT_UNVERIFIABLE,
    Claim,
    FinancialFact,
    ReconciliationResult,
)

# Defaults chosen for the messiness of real filings: absolute dollar claims in prose
# are often rounded, and growth/margin figures are sometimes computed by the company
# on a slightly different (e.g. constant-currency) basis than raw GAAP figures.
DEFAULT_TOLERANCES = {
    COMPARISON_ABSOLUTE: 0.01,  # 1% relative tolerance
    COMPARISON_GROWTH_PCT: 1.0,  # 1.0 percentage point
    COMPARISON_ABSOLUTE_CHANGE: 0.01,  # 1% relative tolerance (of the larger magnitude)
    COMPARISON_BPS_CHANGE: 50.0,  # 50 basis points
}


def _resolve_tolerance(claim: Claim) -> float:
    if claim.tolerance is not None:
        return claim.tolerance
    return DEFAULT_TOLERANCES.get(claim.comparison_type, 0.0)


def _find_fact(
    facts: list[FinancialFact],
    concept: str,
    period_start: str | None,
    period_end: str,
    unit: str,
) -> tuple[FinancialFact | None, str | None]:
    """Find the fact for a (concept, period, unit). Returns (fact, error) — error is
    None on success, or a human-readable reason it couldn't be resolved (not found,
    or ambiguous because multiple period_start values share this period_end and the
    claim didn't disambiguate)."""
    candidates = [f for f in facts if f.concept == concept and f.unit == unit and f.period_end == period_end]

    if period_start is not None:
        candidates = [f for f in candidates if f.period_start == period_start]
    else:
        distinct_starts = {f.period_start for f in candidates}
        if len(distinct_starts) > 1:
            return None, (
                f"Ambiguous period: {len(distinct_starts)} different period_start values "
                f"report '{concept}' ending {period_end} (e.g. a quarter and a fiscal year "
                f"can share an end date) — claim must specify period_start."
            )

    if not candidates:
        return None, f"No reported '{concept}' found for period ending {period_end}."

    # The same period is often re-reported verbatim in later filings' comparative
    # columns (e.g. a 10-K shows 3 years of history) — that's not a restatement, so
    # the most natural citation is the original filing, not the latest one to repeat
    # it. But if the reported *value* actually differs across filings, that's a real
    # restatement/correction, and the most recently filed value is the authoritative
    # one to reconcile against.
    distinct_values = {f.value for f in candidates}
    if len(distinct_values) == 1:
        best = min(candidates, key=lambda f: (f.filed, f.accession_number))
    else:
        best = max(candidates, key=lambda f: (f.filed, f.accession_number))
    return best, None


def reconcile(claim: Claim, facts: list[FinancialFact]) -> ReconciliationResult:
    """Check a Claim against a list of FinancialFacts (typically all facts retrieved
    for one ticker). Never raises on missing data — returns verdict='unverifiable'
    instead, since a reward function must always produce a signal."""
    tolerance = _resolve_tolerance(claim)

    if claim.comparison_type == COMPARISON_ABSOLUTE:
        return _reconcile_absolute(claim, facts, tolerance)
    if claim.comparison_type == COMPARISON_GROWTH_PCT:
        return _reconcile_growth_pct(claim, facts, tolerance)
    if claim.comparison_type == COMPARISON_ABSOLUTE_CHANGE:
        return _reconcile_absolute_change(claim, facts, tolerance)
    if claim.comparison_type == COMPARISON_BPS_CHANGE:
        return _reconcile_bps_change(claim, facts, tolerance)

    return ReconciliationResult(
        verdict=VERDICT_UNVERIFIABLE,
        claim=claim,
        computed_value=None,
        difference=None,
        tolerance=tolerance,
        explanation=f"Unknown comparison_type '{claim.comparison_type}'.",
        citations=[],
    )


def _unverifiable(claim: Claim, tolerance: float, reason: str) -> ReconciliationResult:
    return ReconciliationResult(
        verdict=VERDICT_UNVERIFIABLE,
        claim=claim,
        computed_value=None,
        difference=None,
        tolerance=tolerance,
        explanation=reason,
        citations=[],
    )


def _reconcile_absolute(claim: Claim, facts: list[FinancialFact], tolerance: float) -> ReconciliationResult:
    fact, error = _find_fact(facts, claim.metric, claim.period_start, claim.period_end, claim.unit)
    if fact is None:
        return _unverifiable(claim, tolerance, error)

    computed_value = fact.value
    denom = abs(computed_value) if computed_value != 0 else 1.0
    difference = abs(computed_value - claim.claimed_value) / denom
    verdict = VERDICT_CONSISTENT if difference <= tolerance else VERDICT_INCONSISTENT

    return ReconciliationResult(
        verdict=verdict,
        claim=claim,
        computed_value=computed_value,
        difference=difference,
        tolerance=tolerance,
        explanation=(
            f"Claimed {claim.metric}={claim.claimed_value:,.0f}; EDGAR reports "
            f"{computed_value:,.0f} for the period ending {claim.period_end} "
            f"(relative difference {difference:.2%})."
        ),
        citations=[fact.accession_number],
    )


def _reconcile_growth_pct(claim: Claim, facts: list[FinancialFact], tolerance: float) -> ReconciliationResult:
    if claim.comparison_period_end is None:
        return _unverifiable(claim, tolerance, "growth_pct claims require comparison_period_end.")

    current, err1 = _find_fact(facts, claim.metric, claim.period_start, claim.period_end, claim.unit)
    prior, err2 = _find_fact(
        facts, claim.metric, claim.comparison_period_start, claim.comparison_period_end, claim.unit
    )
    if current is None or prior is None:
        return _unverifiable(claim, tolerance, err1 or err2)
    if prior.value == 0:
        return _unverifiable(claim, tolerance, "Comparison period value is zero; percentage growth is undefined.")

    computed_value = (current.value - prior.value) / abs(prior.value) * 100
    difference = abs(computed_value - claim.claimed_value)
    verdict = VERDICT_CONSISTENT if difference <= tolerance else VERDICT_INCONSISTENT

    return ReconciliationResult(
        verdict=verdict,
        claim=claim,
        computed_value=computed_value,
        difference=difference,
        tolerance=tolerance,
        explanation=(
            f"Claimed {claim.metric} growth of {claim.claimed_value:.2f}%; EDGAR data implies "
            f"{computed_value:.2f}% ({prior.value:,.0f} -> {current.value:,.0f}), "
            f"a difference of {difference:.2f} percentage points."
        ),
        citations=[current.accession_number, prior.accession_number],
    )


def _reconcile_absolute_change(claim: Claim, facts: list[FinancialFact], tolerance: float) -> ReconciliationResult:
    if claim.comparison_period_end is None:
        return _unverifiable(claim, tolerance, "absolute_change claims require comparison_period_end.")

    current, err1 = _find_fact(facts, claim.metric, claim.period_start, claim.period_end, claim.unit)
    prior, err2 = _find_fact(
        facts, claim.metric, claim.comparison_period_start, claim.comparison_period_end, claim.unit
    )
    if current is None or prior is None:
        return _unverifiable(claim, tolerance, err1 or err2)

    computed_value = current.value - prior.value
    denom = max(abs(current.value), abs(prior.value)) or 1.0
    difference = abs(computed_value - claim.claimed_value) / denom
    verdict = VERDICT_CONSISTENT if difference <= tolerance else VERDICT_INCONSISTENT

    return ReconciliationResult(
        verdict=verdict,
        claim=claim,
        computed_value=computed_value,
        difference=difference,
        tolerance=tolerance,
        explanation=(
            f"Claimed {claim.metric} changed by {claim.claimed_value:,.0f}; EDGAR data implies "
            f"{computed_value:,.0f} ({prior.value:,.0f} -> {current.value:,.0f}), "
            f"a relative difference of {difference:.2%}."
        ),
        citations=[current.accession_number, prior.accession_number],
    )


def _reconcile_bps_change(claim: Claim, facts: list[FinancialFact], tolerance: float) -> ReconciliationResult:
    if claim.comparison_period_end is None or claim.denominator_metric is None:
        return _unverifiable(
            claim, tolerance, "bps_change claims require comparison_period_end and denominator_metric."
        )

    num_current, e1 = _find_fact(facts, claim.metric, claim.period_start, claim.period_end, claim.unit)
    den_current, e2 = _find_fact(facts, claim.denominator_metric, claim.period_start, claim.period_end, claim.unit)
    num_prior, e3 = _find_fact(
        facts, claim.metric, claim.comparison_period_start, claim.comparison_period_end, claim.unit
    )
    den_prior, e4 = _find_fact(
        facts, claim.denominator_metric, claim.comparison_period_start, claim.comparison_period_end, claim.unit
    )

    missing = [f for f in (num_current, den_current, num_prior, den_prior) if f is None]
    if missing:
        return _unverifiable(claim, tolerance, e1 or e2 or e3 or e4)
    if den_current.value == 0 or den_prior.value == 0:
        return _unverifiable(claim, tolerance, "Denominator metric is zero in one of the periods; ratio undefined.")

    margin_current = num_current.value / den_current.value
    margin_prior = num_prior.value / den_prior.value
    computed_value = (margin_current - margin_prior) * 10_000  # fraction -> basis points
    difference = abs(computed_value - claim.claimed_value)
    verdict = VERDICT_CONSISTENT if difference <= tolerance else VERDICT_INCONSISTENT

    return ReconciliationResult(
        verdict=verdict,
        claim=claim,
        computed_value=computed_value,
        difference=difference,
        tolerance=tolerance,
        explanation=(
            f"Claimed {claim.metric}/{claim.denominator_metric} margin changed by "
            f"{claim.claimed_value:.1f} bps; EDGAR data implies {computed_value:.1f} bps "
            f"({margin_prior:.4%} -> {margin_current:.4%}), a difference of {difference:.1f} bps."
        ),
        citations=[
            num_current.accession_number,
            den_current.accession_number,
            num_prior.accession_number,
            den_prior.accession_number,
        ],
    )
