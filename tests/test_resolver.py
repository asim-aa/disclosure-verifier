import pytest

from agents.resolver import resolve_concept, resolve_periods
from tools.schema import FinancialFact


def fact(concept, value, period_start, period_end, accn="a", filed="2026-01-01", unit="USD"):
    return FinancialFact(
        ticker="MSFT",
        cik="0000789019",
        concept=concept,
        label=concept,
        value=value,
        unit=unit,
        period_start=period_start,
        period_end=period_end,
        fiscal_year=2026,
        fiscal_period="FY",
        form="10-K",
        filed=filed,
        accession_number=accn,
    )


# ---------- resolve_concept ----------


def test_resolve_concept_returns_candidate_with_most_recent_data():
    facts = [
        fact("RevenueFromContractWithCustomerExcludingAssessedTax", 100, "2021-01-01", "2022-01-30"),
        fact("Revenues", 200, "2025-01-01", "2026-01-25"),
    ]
    assert resolve_concept("revenue", facts) == "Revenues"


def test_resolve_concept_prefers_recency_over_candidate_order():
    """Regression test for a real bug: NVDA's revenue data confirms companies
    retag concepts over time (RevenueFromContractWithCustomerExcludingAssessedTax
    was used through FY2022, then dropped for plain Revenues). The first-listed
    candidate existing at all is not enough — it can be a stale, discontinued tag
    whose most recent data is years old, while a *later* candidate has current
    data. Must pick by recency, not list order."""
    facts = [
        # first-priority candidate exists, but only with old data
        fact("RevenueFromContractWithCustomerExcludingAssessedTax", 100, "2021-01-01", "2022-01-30"),
        # second-priority candidate has much more recent data
        fact("Revenues", 999, "2025-04-01", "2026-04-26"),
    ]
    assert resolve_concept("revenue", facts) == "Revenues"


def test_resolve_concept_falls_back_to_second_candidate_if_first_unavailable():
    facts = [fact("Revenues", 100, "2025-01-01", "2026-01-25")]
    assert resolve_concept("revenue", facts) == "Revenues"


def test_resolve_concept_case_insensitive():
    facts = [fact("GrossProfit", 40, "2025-01-01", "2026-01-25")]
    assert resolve_concept("Gross Margin", facts) == "GrossProfit"


def test_resolve_concept_known_variant_is_an_explicit_alias():
    facts = [fact("GrossProfit", 40, "2025-01-01", "2026-01-25")]
    assert resolve_concept("gross margin percentage", facts) == "GrossProfit"


def test_resolve_concept_does_not_fuzzy_match_unknown_variants():
    """A wording variant NOT explicitly listed must come back unresolved rather
    than guess via partial matching."""
    facts = [fact("GrossProfit", 40, "2025-01-01", "2026-01-25")]
    assert resolve_concept("gross margin ratio", facts) is None


def test_resolve_concept_none_when_no_candidates_available():
    assert resolve_concept("revenue", []) is None


def test_resolve_concept_none_for_unmapped_segment_specific_metric():
    """Segment-level metrics ('Azure revenue') aren't standard top-level us-gaap
    concepts — must come back unresolved, not guess something wrong."""
    facts = [
        fact("RevenueFromContractWithCustomerExcludingAssessedTax", 100, "2025-01-01", "2026-01-25"),
        fact("GrossProfit", 40, "2025-01-01", "2026-01-25"),
        fact("OperatingExpenses", 20, "2025-01-01", "2026-01-25"),
    ]
    assert resolve_concept("Azure and other cloud services revenue", facts) is None


# ---------- resolve_periods ----------


def test_resolve_periods_picks_most_recent_as_current():
    facts = [
        fact("Revenues", 100, "2024-01-01", "2024-12-31"),
        fact("Revenues", 90, "2023-01-01", "2023-12-31"),
    ]
    current, comparison = resolve_periods(facts, "Revenues")
    assert (current.period_start, current.period_end) == ("2024-01-01", "2024-12-31")
    assert (comparison.period_start, comparison.period_end) == ("2023-01-01", "2023-12-31")


def test_resolve_periods_skips_duplicate_repeated_period_for_comparison():
    facts = [
        fact("Revenues", 100, "2024-01-01", "2024-12-31", accn="new"),
        fact("Revenues", 100, "2024-01-01", "2024-12-31", accn="repeat"),  # same period repeated
        fact("Revenues", 90, "2023-01-01", "2023-12-31"),
    ]
    _current, comparison = resolve_periods(facts, "Revenues")
    assert (comparison.period_start, comparison.period_end) == ("2023-01-01", "2023-12-31")


def test_resolve_periods_as_of_excludes_later_filed_quarterly_data():
    """Regression test for a real bug, confirmed against live NVDA data: a claim
    extracted from a 10-K's MD&A describes that 10-K's own annual figure, but by
    the time verification runs, a later 10-Q may already be filed with a more
    recent (quarterly) period_end. Without an as_of cutoff, "most recent" picks
    the newer quarterly figure and compares the wrong two numbers entirely."""
    facts = [
        fact("Revenues", 215_900_000_000, "2025-01-27", "2026-01-25", filed="2026-02-25"),  # the 10-K's own FY figure
        fact("Revenues", 81_615_000_000, "2026-01-26", "2026-04-26", filed="2026-05-20"),  # a LATER 10-Q, filed after
    ]
    current, _ = resolve_periods(facts, "Revenues", as_of="2026-02-25")
    assert current.value == 215_900_000_000


def test_resolve_periods_as_of_none_uses_globally_most_recent():
    """Without an as_of cutoff (the old, buggy default), the later filing wins -
    documents why callers must pass as_of, not that the fallback is desirable."""
    facts = [
        fact("Revenues", 215_900_000_000, "2025-01-27", "2026-01-25", filed="2026-02-25"),
        fact("Revenues", 81_615_000_000, "2026-01-26", "2026-04-26", filed="2026-05-20"),
    ]
    current, _ = resolve_periods(facts, "Revenues")
    assert current.value == 81_615_000_000


def test_resolve_concept_as_of_ignores_tag_switch_that_happens_later():
    facts = [
        fact("OldTag", 100, "2025-01-01", "2026-01-25", filed="2026-02-25"),
        fact("NewTag", 200, "2026-01-26", "2026-04-26", filed="2026-05-20"),
    ]
    from agents import resolver

    resolver.METRIC_TO_CONCEPTS["_test_metric"] = ["OldTag", "NewTag"]
    try:
        assert resolve_concept("_test_metric", facts, as_of="2026-02-25") == "OldTag"
        assert resolve_concept("_test_metric", facts) == "NewTag"
    finally:
        del resolver.METRIC_TO_CONCEPTS["_test_metric"]


def test_resolve_periods_comparison_none_when_only_one_period():
    facts = [fact("Revenues", 100, "2024-01-01", "2024-12-31")]
    current, comparison = resolve_periods(facts, "Revenues")
    assert (current.period_start, current.period_end) == ("2024-01-01", "2024-12-31")
    assert comparison is None


def test_resolve_periods_returns_actual_fact_with_its_real_unit():
    """The caller needs the real .unit off the fact (USD / USD-per-shares / pure) -
    not a guess based on the claim's own wording."""
    facts = [
        fact("EffectiveIncomeTaxRateContinuingOperations", 0.19, "2024-01-01", "2024-12-31", unit="pure"),
    ]
    current, _ = resolve_periods(facts, "EffectiveIncomeTaxRateContinuingOperations")
    assert current.unit == "pure"
    assert current.value == 0.19


def test_resolve_periods_raises_for_unknown_concept():
    facts = [fact("Revenues", 100, "2024-01-01", "2024-12-31")]
    with pytest.raises(ValueError):
        resolve_periods(facts, "GrossProfit")


def test_resolve_periods_ignores_other_concepts():
    facts = [
        fact("Revenues", 100, "2024-01-01", "2024-12-31"),
        fact("GrossProfit", 40, "2024-01-01", "2024-12-31"),
    ]
    current, _comparison = resolve_periods(facts, "Revenues")
    assert current.concept == "Revenues"
    assert (current.period_start, current.period_end) == ("2024-01-01", "2024-12-31")
