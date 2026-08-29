import pytest

from agents.resolver import resolve_concept, resolve_periods
from tools.schema import FinancialFact


def fact(
    concept, value, period_start, period_end, accn="a", filed="2026-01-01", unit="USD",
    fiscal_year=2026, fiscal_period="FY",
):
    return FinancialFact(
        ticker="MSFT",
        cik="0000789019",
        concept=concept,
        label=concept,
        value=value,
        unit=unit,
        period_start=period_start,
        period_end=period_end,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
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


def test_resolve_periods_default_comparison_skips_a_ytd_fact_between_current_and_prior_year():
    """Reproduces a real bug found via research/specificity_check.py against live
    TXN data: a Q3 10-Q's 9-month year-to-date revenue fact (period_end 3 months
    before the fiscal year end) sorted ahead of the correct prior full year in
    the plain "next distinct period" pick, because that pick never checked
    duration - only period_end recency. A genuinely accurate claim ("increased
    $2.04B, or 13.0%, compared to fiscal 2024") read as wildly inconsistent
    (implied 33% growth) as a result. The comparison must prefer a period whose
    length matches current's, not just whichever is chronologically closest."""
    facts = [
        fact("Revenues", 17_682_000_000, "2025-01-01", "2025-12-31"),  # current: FY2025
        fact("Revenues", 13_259_000_000, "2025-01-01", "2025-09-30"),  # Q3 2025 9-month YTD - a decoy
        fact("Revenues", 15_641_000_000, "2024-01-01", "2024-12-31"),  # correct: FY2024
    ]
    current, comparison = resolve_periods(facts, "Revenues")
    assert current.value == 17_682_000_000
    assert comparison.value == 15_641_000_000


def test_resolve_periods_default_comparison_falls_back_when_no_similar_length_period_exists():
    """If the only prior data available is a different length than current (e.g.
    a company's first annual figure with only quarterly history before it), the
    old "closest distinct period" behavior is still the right fallback - better
    than returning no comparison at all."""
    facts = [
        fact("Revenues", 100, "2024-01-01", "2024-12-31"),  # current: annual
        fact("Revenues", 20, "2023-10-01", "2023-12-31"),  # only a quarter exists before it
    ]
    current, comparison = resolve_periods(facts, "Revenues")
    assert current.value == 100
    assert comparison.value == 20


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


# ---------- period_hint ----------


def test_resolve_periods_period_hint_fixes_the_documented_nvda_bug():
    """Regression test for the exact bug flagged in the module docstring,
    confirmed against live NVDA data: 'Income tax expense was $21.4 billion and
    $11.1 billion for fiscal years 2026 and 2025, respectively' is extracted as
    two claims, one per stated year. Without period_hint both resolved to the
    SAME (current, comparison) pair; with it, each resolves to its own year."""
    facts = [
        fact("IncomeTaxExpenseBenefit", 21_400_000_000, "2025-01-27", "2026-01-25", fiscal_year=2026),
        fact("IncomeTaxExpenseBenefit", 11_100_000_000, "2024-01-29", "2025-01-26", fiscal_year=2025),
        fact("IncomeTaxExpenseBenefit", 5_000_000_000, "2023-01-30", "2024-01-28", fiscal_year=2024),
    ]
    current_2026, comparison_2026 = resolve_periods(
        facts, "IncomeTaxExpenseBenefit", period_hint="fiscal year 2026"
    )
    assert current_2026.value == 21_400_000_000
    assert comparison_2026.value == 11_100_000_000

    current_2025, comparison_2025 = resolve_periods(
        facts, "IncomeTaxExpenseBenefit", period_hint="fiscal year 2025"
    )
    assert current_2025.value == 11_100_000_000
    assert comparison_2025.value == 5_000_000_000  # FY2024, not FY2026 - not "backwards"


def test_resolve_periods_period_hint_recognizes_fy_shorthand():
    facts = [
        fact("Revenues", 200, "2025-01-01", "2026-12-31", fiscal_year=2026),
        fact("Revenues", 180, "2024-01-01", "2024-12-31", fiscal_year=2025),
    ]
    current, _ = resolve_periods(facts, "Revenues", period_hint="FY2025")
    assert current.value == 180


def test_resolve_periods_period_hint_recognizes_ordinal_quarter():
    facts = [
        fact("Revenues", 100, "2026-07-01", "2026-09-30", fiscal_period="Q3"),
        fact("Revenues", 90, "2026-04-01", "2026-06-30", fiscal_period="Q2"),
    ]
    current, _ = resolve_periods(facts, "Revenues", period_hint="the third quarter")
    assert current.value == 100


def test_resolve_periods_period_hint_recognizes_q_shorthand():
    facts = [
        fact("Revenues", 100, "2026-07-01", "2026-09-30", fiscal_period="Q3"),
        fact("Revenues", 90, "2026-04-01", "2026-06-30", fiscal_period="Q2"),
    ]
    current, _ = resolve_periods(facts, "Revenues", period_hint="Q3 results")
    assert current.value == 100


def test_resolve_periods_period_hint_combines_quarter_and_fiscal_year():
    facts = [
        fact("Revenues", 100, "2026-07-01", "2026-09-30", fiscal_year=2026, fiscal_period="Q3"),
        fact("Revenues", 80, "2025-07-01", "2025-09-30", fiscal_year=2025, fiscal_period="Q3"),
    ]
    current, _ = resolve_periods(facts, "Revenues", period_hint="the third quarter of fiscal 2025")
    assert current.value == 80


def test_resolve_periods_calendar_named_quarter_still_declines_to_guess():
    """'the September quarter' names a quarter by calendar month, not fiscal
    ordinal - translating that safely needs the company's fiscal-year-end,
    which isn't available here, so it must fall back to the plain "most
    recent" default rather than incorrectly matching the Q3 fact just because
    the hint contains the word "quarter"."""
    facts = [
        fact("Revenues", 200, "2026-01-01", "2026-12-31", fiscal_period="FY"),  # most recent
        fact("Revenues", 100, "2026-07-01", "2026-09-30", fiscal_period="Q3"),  # older
    ]
    current, _ = resolve_periods(facts, "Revenues", period_hint="the September quarter")
    assert current.value == 200  # unchanged default: most recent, not the Q3 fact


def test_resolve_periods_quarter_hint_prefers_standalone_over_ytd_cumulative():
    """A 10-Q commonly tags both the standalone 3-month figure and the 6-/9-month
    year-to-date cumulative with the identical fiscal_period and period_end -
    'the third quarter' must resolve to the standalone one, not the YTD total.
    Reproduces a real bug found against live NVDA data: an unfiltered match
    picked a $91.166B 9-month cumulative instead of the $35.082B Q3 figure."""
    facts = [
        fact("Revenues", 91_166, "2024-01-29", "2024-10-27", fiscal_year=2025, fiscal_period="Q3"),  # 9-month YTD
        fact("Revenues", 35_082, "2024-07-29", "2024-10-27", fiscal_year=2025, fiscal_period="Q3"),  # standalone
    ]
    current, _ = resolve_periods(facts, "Revenues", period_hint="the third quarter of fiscal 2025")
    assert current.value == 35_082


def test_resolve_periods_quarter_hint_not_found_falls_back_to_default():
    facts = [fact("Revenues", 100, "2026-07-01", "2026-09-30", fiscal_period="Q3")]
    current, _ = resolve_periods(facts, "Revenues", period_hint="the fourth quarter")
    assert current.value == 100  # no Q4 fact - falls back rather than erroring


def test_resolve_periods_unparseable_hint_falls_back_to_default():
    facts = [
        fact("Revenues", 200, "2025-01-01", "2025-12-31"),
        fact("Revenues", 180, "2024-01-01", "2024-12-31"),
    ]
    current, comparison = resolve_periods(facts, "Revenues", period_hint="the September quarter")
    assert current.value == 200  # unchanged default: most recent
    assert comparison.value == 180


def test_resolve_periods_hinted_year_not_found_falls_back_to_default():
    facts = [fact("Revenues", 200, "2025-01-01", "2025-12-31", fiscal_year=2025)]
    current, _ = resolve_periods(facts, "Revenues", period_hint="fiscal year 1999")
    assert current.value == 200  # no FY1999 data - falls back rather than erroring


def test_resolve_periods_sequential_picks_immediately_preceding_same_length_period():
    """'sequentially' means prior quarter, not same quarter last year - a
    quarterly current period should pick the nearest prior quarter, skipping
    over a same-length period from a year further back and any annual figures
    mixed into the same concept's fact history."""
    facts = [
        fact("Revenues", 100, "2026-01-01", "2026-03-31"),  # current: Q1 FY2026
        fact("Revenues", 90, "2025-10-01", "2025-12-31"),  # immediately prior quarter
        fact("Revenues", 80, "2025-01-01", "2025-03-31"),  # same quarter last year
        fact("Revenues", 350, "2025-01-01", "2025-12-31"),  # FY2025 annual - different length
    ]
    current, comparison = resolve_periods(facts, "Revenues", period_hint="sequentially")
    assert current.value == 100
    assert comparison.value == 90


def test_resolve_periods_year_ago_picks_same_quarter_prior_year_not_prior_quarter():
    facts = [
        fact("Revenues", 100, "2026-01-01", "2026-03-31"),  # current: Q1 FY2026
        fact("Revenues", 90, "2025-10-01", "2025-12-31"),  # immediately prior quarter
        fact("Revenues", 80, "2025-01-01", "2025-03-31"),  # same quarter last year
    ]
    current, comparison = resolve_periods(facts, "Revenues", period_hint="up 25% from a year ago")
    assert current.value == 100
    assert comparison.value == 80


def test_resolve_periods_year_ago_ignores_a_match_outside_tolerance():
    """A same-length candidate that isn't actually ~1 year back (e.g. a
    fiscal-calendar quirk landing it 4+ months off) shouldn't be guessed at -
    falls back to the default pick instead of a wrong 'year ago' match."""
    facts = [
        fact("Revenues", 100, "2026-01-01", "2026-03-31"),
        fact("Revenues", 90, "2025-06-01", "2025-08-31"),  # ~7 months back, same length - too far for "year ago"
    ]
    current, comparison = resolve_periods(facts, "Revenues", period_hint="a year ago")
    assert current.value == 100
    assert comparison.value == 90  # default fallback still finds it, just not via the year-ago path


# ---------- concepts added after Phase 6's integration-at-scale run ----------
#
# Each maps to a real, standard, non-dimensional us-gaap concept confirmed present
# in real AAPL/MSFT/NVDA company-facts data (see agents/resolver.py's module
# docstring) — distinct from the genuinely out-of-scope segment/product metrics
# (Azure revenue, LinkedIn revenue, ...) that same run also surfaced.


@pytest.mark.parametrize(
    ("metric_text", "concept"),
    [
        ("net income", "NetIncomeLoss"),
        ("diluted earnings per share", "EarningsPerShareDiluted"),
        ("total assets", "Assets"),
        ("total liabilities", "Liabilities"),
        ("total stockholders' equity", "StockholdersEquity"),
        ("cash and cash equivalents", "CashAndCashEquivalentsAtCarryingValue"),
        ("research and development expense", "ResearchAndDevelopmentExpense"),
        ("selling, general and administrative expenses", "SellingGeneralAndAdministrativeExpense"),
        ("capital expenditures", "PaymentsToAcquirePropertyPlantAndEquipment"),
        ("commercial remaining performance obligation", "RevenueRemainingPerformanceObligation"),
        ("remaining performance obligation", "RevenueRemainingPerformanceObligation"),
        ("interest expense", "InterestExpense"),
    ],
)
def test_resolve_concept_resolves_newly_added_standard_concepts(metric_text, concept):
    facts = [fact(concept, 100, "2025-01-01", "2025-12-31")]
    assert resolve_concept(metric_text, facts) == concept


# ---------- concepts added from real unresolved-claim analysis ----------
#
# Found by checking which metric texts in eval/labeled_claims*.jsonl actually
# failed to resolve, not guessed. New concepts confirmed present in real
# AAPL/MSFT/NVDA/AMZN company-facts data before being added (same discipline
# as above); wording-variant entries reuse an already-confirmed concept.


@pytest.mark.parametrize(
    ("metric_text", "concept"),
    [
        ("net sales", "Revenues"),
        ("sales", "Revenues"),
        ("provision for income taxes", "IncomeTaxExpenseBenefit"),
        ("cash provided by operating activities", "NetCashProvidedByUsedInOperatingActivities"),
        ("cash used in investing activities", "NetCashProvidedByUsedInInvestingActivities"),
        ("cash used in financing activities", "NetCashProvidedByUsedInFinancingActivities"),
        ("long-term debt", "LongTermDebt"),
        ("interest income", "InvestmentIncomeInterest"),
        ("other income (expense), net", "OtherNonoperatingIncomeExpense"),
        ("long-term lease liabilities", "OperatingLeaseLiabilityNoncurrent"),
    ],
)
def test_resolve_concept_resolves_concepts_added_from_unresolved_claim_analysis(metric_text, concept):
    facts = [fact(concept, 100, "2025-01-01", "2025-12-31")]
    assert resolve_concept(metric_text, facts) == concept


def test_resolve_concept_still_declines_a_genuine_segment_level_metric():
    """The dictionary expansion added real standard concepts; it must not have
    accidentally widened matching enough to guess at a segment-specific one that
    was never in scope (see the module docstring's "no dimensional data at all"
    finding) — "Azure revenue" has no safe candidate to fall back to."""
    facts = [fact("Revenues", 100, "2025-01-01", "2025-12-31")]
    assert resolve_concept("azure and other cloud services revenue", facts) is None
