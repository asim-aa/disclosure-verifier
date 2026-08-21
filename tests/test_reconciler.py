"""Unit tests for the Numerical Reconciler — hand-crafted true and false claims,
plus claims built on AAPL's real, known FY2022/FY2023 reported figures (pulled
live from EDGAR while building this module, hardcoded here so these tests run
fast and deterministically without a network call):

  FY2022 (2021-09-26 -> 2022-09-24): Revenue=$394,328,000,000  GrossProfit=$170,782,000,000
  FY2023 (2022-09-25 -> 2023-09-30): Revenue=$383,285,000,000  GrossProfit=$169,148,000,000

  -> real revenue growth FY23 vs FY22:  -2.8005%
  -> real gross margin change FY23 vs FY22: +82.15 bps
"""

import pytest

from tools.reconciler import reconcile
from tools.schema import (
    VERDICT_CONSISTENT,
    VERDICT_INCONSISTENT,
    VERDICT_UNVERIFIABLE,
    Claim,
    FinancialFact,
)

TICKER = "AAPL"
CIK = "0000320193"

FY22_START, FY22_END = "2021-09-26", "2022-09-24"
FY23_START, FY23_END = "2022-09-25", "2023-09-30"


def _fact(concept, value, period_start, period_end, accn, filed="2023-11-03", form="10-K"):
    return FinancialFact(
        ticker=TICKER,
        cik=CIK,
        concept=concept,
        label=concept,
        value=value,
        unit="USD",
        period_start=period_start,
        period_end=period_end,
        fiscal_year=2023,
        fiscal_period="FY",
        form=form,
        filed=filed,
        accession_number=accn,
    )


@pytest.fixture
def aapl_facts():
    return [
        _fact("Revenues", 394_328_000_000, FY22_START, FY22_END, "0000320193-22-000108", filed="2022-10-28"),
        _fact("Revenues", 383_285_000_000, FY23_START, FY23_END, "0000320193-23-000106", filed="2023-11-03"),
        _fact("GrossProfit", 170_782_000_000, FY22_START, FY22_END, "0000320193-22-000108", filed="2022-10-28"),
        _fact("GrossProfit", 169_148_000_000, FY23_START, FY23_END, "0000320193-23-000106", filed="2023-11-03"),
    ]


# ---------- absolute ----------


def test_absolute_true_claim_is_consistent(aapl_facts):
    claim = Claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="absolute",
        claimed_value=383_285_000_000,
        period_end=FY23_END,
        period_start=FY23_START,
    )
    result = reconcile(claim, aapl_facts)
    assert result.verdict == VERDICT_CONSISTENT
    assert result.citations == ["0000320193-23-000106"]


def test_absolute_false_claim_is_inconsistent(aapl_facts):
    claim = Claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="absolute",
        claimed_value=500_000_000_000,  # nowhere close to the real $383.285B
        period_end=FY23_END,
        period_start=FY23_START,
    )
    result = reconcile(claim, aapl_facts)
    assert result.verdict == VERDICT_INCONSISTENT


def test_absolute_within_rounding_tolerance_is_consistent(aapl_facts):
    """Prose often rounds ('~$383 billion'); small relative differences should pass."""
    claim = Claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="absolute",
        claimed_value=383_000_000_000,  # rounded, 0.074% off the real value
        period_end=FY23_END,
        period_start=FY23_START,
    )
    result = reconcile(claim, aapl_facts)
    assert result.verdict == VERDICT_CONSISTENT


def test_absolute_missing_period_is_unverifiable(aapl_facts):
    claim = Claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="absolute",
        claimed_value=100,
        period_end="1999-01-01",
    )
    result = reconcile(claim, aapl_facts)
    assert result.verdict == VERDICT_UNVERIFIABLE
    assert result.citations == []


def test_absolute_ambiguous_period_without_period_start_is_unverifiable(aapl_facts):
    """Two different period_starts share period_end=FY23_END in the fixture below
    (a quarter and the full year can end on the same date) — period_start must be
    given to disambiguate, or the claim can't be safely verified."""
    facts = aapl_facts + [
        _fact("Revenues", 90_000_000_000, "2023-07-01", FY23_END, "0000320193-23-000999"),
    ]
    claim = Claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="absolute",
        claimed_value=383_285_000_000,
        period_end=FY23_END,
        # period_start intentionally omitted
    )
    result = reconcile(claim, facts)
    assert result.verdict == VERDICT_UNVERIFIABLE
    assert "Ambiguous" in result.explanation


def test_absolute_prefers_most_recently_filed_value_for_restated_period(aapl_facts):
    """If a later filing restates a prior period's figure, the most recently filed
    value should win."""
    restated = aapl_facts + [
        _fact(
            "Revenues", 383_300_000_000, FY23_START, FY23_END, "0000320193-24-999999", filed="2024-01-01"
        )
    ]
    claim = Claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="absolute",
        claimed_value=383_300_000_000,
        period_end=FY23_END,
        period_start=FY23_START,
    )
    result = reconcile(claim, restated)
    assert result.verdict == VERDICT_CONSISTENT
    assert result.citations == ["0000320193-24-999999"]


def test_absolute_cites_original_filing_when_later_filings_just_repeat_the_same_value(aapl_facts):
    """A 10-K's income statement typically shows 3 years of history, so the same
    period's figure gets re-reported verbatim in later filings' comparative columns
    (confirmed against real AAPL data: FY2023 revenue appears identically in the
    FY2023, FY2024, and FY2025 10-Ks). That's not a restatement, so the citation
    should point to the original filing, not whichever filing happened to repeat it
    most recently."""
    repeated = aapl_facts + [
        _fact("Revenues", 383_285_000_000, FY23_START, FY23_END, "0000320193-24-000123", filed="2024-11-01"),
        _fact("Revenues", 383_285_000_000, FY23_START, FY23_END, "0000320193-25-000079", filed="2025-10-31"),
    ]
    claim = Claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="absolute",
        claimed_value=383_285_000_000,
        period_end=FY23_END,
        period_start=FY23_START,
    )
    result = reconcile(claim, repeated)
    assert result.verdict == VERDICT_CONSISTENT
    assert result.citations == ["0000320193-23-000106"]  # the original FY2023 10-K


# ---------- growth_pct ----------


def test_growth_pct_true_claim_is_consistent(aapl_facts):
    claim = Claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="growth_pct",
        claimed_value=-2.8,  # real figure is -2.8005%
        period_end=FY23_END,
        period_start=FY23_START,
        comparison_period_end=FY22_END,
        comparison_period_start=FY22_START,
    )
    result = reconcile(claim, aapl_facts)
    assert result.verdict == VERDICT_CONSISTENT
    assert result.computed_value == pytest.approx(-2.8005, abs=0.01)


def test_growth_pct_false_claim_is_inconsistent(aapl_facts):
    claim = Claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="growth_pct",
        claimed_value=12.0,  # AAPL's revenue actually declined this year
        period_end=FY23_END,
        period_start=FY23_START,
        comparison_period_end=FY22_END,
        comparison_period_start=FY22_START,
    )
    result = reconcile(claim, aapl_facts)
    assert result.verdict == VERDICT_INCONSISTENT


def test_growth_pct_missing_comparison_period_is_unverifiable(aapl_facts):
    claim = Claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="growth_pct",
        claimed_value=-2.8,
        period_end=FY23_END,
        period_start=FY23_START,
        # comparison_period_end intentionally omitted
    )
    result = reconcile(claim, aapl_facts)
    assert result.verdict == VERDICT_UNVERIFIABLE


def test_growth_pct_zero_prior_value_is_unverifiable():
    facts = [
        _fact("Revenues", 0, FY22_START, FY22_END, "accn-1"),
        _fact("Revenues", 100, FY23_START, FY23_END, "accn-2"),
    ]
    claim = Claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="growth_pct",
        claimed_value=100.0,
        period_end=FY23_END,
        period_start=FY23_START,
        comparison_period_end=FY22_END,
        comparison_period_start=FY22_START,
    )
    result = reconcile(claim, facts)
    assert result.verdict == VERDICT_UNVERIFIABLE
    assert "undefined" in result.explanation


# ---------- bps_change ----------


def test_bps_change_true_claim_is_consistent(aapl_facts):
    claim = Claim(
        ticker=TICKER,
        metric="GrossProfit",
        denominator_metric="Revenues",
        comparison_type="bps_change",
        claimed_value=82.0,  # real figure is +82.15 bps
        period_end=FY23_END,
        period_start=FY23_START,
        comparison_period_end=FY22_END,
        comparison_period_start=FY22_START,
    )
    result = reconcile(claim, aapl_facts)
    assert result.verdict == VERDICT_CONSISTENT
    assert result.computed_value == pytest.approx(82.15, abs=0.5)
    assert len(result.citations) == 4


def test_bps_change_false_claim_is_inconsistent(aapl_facts):
    claim = Claim(
        ticker=TICKER,
        metric="GrossProfit",
        denominator_metric="Revenues",
        comparison_type="bps_change",
        claimed_value=500.0,  # nowhere close to the real +82 bps
        period_end=FY23_END,
        period_start=FY23_START,
        comparison_period_end=FY22_END,
        comparison_period_start=FY22_START,
    )
    result = reconcile(claim, aapl_facts)
    assert result.verdict == VERDICT_INCONSISTENT


def test_bps_change_missing_denominator_metric_is_unverifiable(aapl_facts):
    claim = Claim(
        ticker=TICKER,
        metric="GrossProfit",
        comparison_type="bps_change",
        claimed_value=82.0,
        period_end=FY23_END,
        period_start=FY23_START,
        comparison_period_end=FY22_END,
        comparison_period_start=FY22_START,
        # denominator_metric intentionally omitted
    )
    result = reconcile(claim, aapl_facts)
    assert result.verdict == VERDICT_UNVERIFIABLE


# ---------- absolute_change ----------


def test_absolute_change_true_claim_is_consistent(aapl_facts):
    # real figure: 383,285,000,000 - 394,328,000,000 = -11,043,000,000
    claim = Claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="absolute_change",
        claimed_value=-11_043_000_000,
        period_end=FY23_END,
        period_start=FY23_START,
        comparison_period_end=FY22_END,
        comparison_period_start=FY22_START,
    )
    result = reconcile(claim, aapl_facts)
    assert result.verdict == VERDICT_CONSISTENT
    assert result.computed_value == pytest.approx(-11_043_000_000)
    assert result.citations == ["0000320193-23-000106", "0000320193-22-000108"]


def test_absolute_change_false_claim_is_inconsistent(aapl_facts):
    claim = Claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="absolute_change",
        claimed_value=50_000_000_000,  # revenue actually declined, not grew by $50B
        period_end=FY23_END,
        period_start=FY23_START,
        comparison_period_end=FY22_END,
        comparison_period_start=FY22_START,
    )
    result = reconcile(claim, aapl_facts)
    assert result.verdict == VERDICT_INCONSISTENT


def test_absolute_change_within_rounding_tolerance_is_consistent(aapl_facts):
    claim = Claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="absolute_change",
        claimed_value=-11_000_000_000,  # rounded, ~0.1% off the real value
        period_end=FY23_END,
        period_start=FY23_START,
        comparison_period_end=FY22_END,
        comparison_period_start=FY22_START,
    )
    result = reconcile(claim, aapl_facts)
    assert result.verdict == VERDICT_CONSISTENT


def test_absolute_change_missing_comparison_period_is_unverifiable(aapl_facts):
    claim = Claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="absolute_change",
        claimed_value=-11_043_000_000,
        period_end=FY23_END,
        period_start=FY23_START,
        # comparison_period_end intentionally omitted
    )
    result = reconcile(claim, aapl_facts)
    assert result.verdict == VERDICT_UNVERIFIABLE


# ---------- misc ----------


def test_unknown_comparison_type_is_unverifiable(aapl_facts):
    claim = Claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="not_a_real_type",
        claimed_value=1,
        period_end=FY23_END,
    )
    result = reconcile(claim, aapl_facts)
    assert result.verdict == VERDICT_UNVERIFIABLE


def test_custom_tolerance_overrides_default(aapl_facts):
    claim = Claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="absolute",
        claimed_value=380_000_000_000,  # 0.86% off — within default 1% tolerance
        period_end=FY23_END,
        period_start=FY23_START,
        tolerance=0.005,  # tighten to 0.5%
    )
    result = reconcile(claim, aapl_facts)
    assert result.verdict == VERDICT_INCONSISTENT


# ---------- as_of (bitemporal correctness) ----------


def test_as_of_protects_a_claim_from_a_later_restatement():
    """A claim made in the FY2023 10-K (filed 2023-11-03) states the FY2023 revenue
    it was accurate about at the time. A LATER filing (e.g. a 10-K/A amendment filed
    in 2024) restates that same period with a different figure. Without an as_of
    cutoff, the reconciler's own "prefer most recent if values differ" rule (a
    legitimate restatement-handling rule in general) would compare the original
    claim against the *restated* number and wrongly call it inconsistent — even
    though the claim was correct as of when it was written."""
    facts = [
        _fact("Revenues", 383_285_000_000, FY23_START, FY23_END, "0000320193-23-000106", filed="2023-11-03"),
        _fact("Revenues", 390_000_000_000, FY23_START, FY23_END, "0000320193-24-999999", filed="2024-03-01"),
    ]
    claim = Claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="absolute",
        claimed_value=383_285_000_000,  # accurate as of the FY2023 10-K itself
        period_end=FY23_END,
        period_start=FY23_START,
    )

    # without as_of: the later restatement wins, and a genuinely accurate claim
    # reads as inconsistent
    result_no_cutoff = reconcile(claim, facts)
    assert result_no_cutoff.verdict == VERDICT_INCONSISTENT
    assert result_no_cutoff.citations == ["0000320193-24-999999"]

    # with as_of pinned to the claim's own source filing date: correctly consistent
    result_with_cutoff = reconcile(claim, facts, as_of="2023-11-03")
    assert result_with_cutoff.verdict == VERDICT_CONSISTENT
    assert result_with_cutoff.citations == ["0000320193-23-000106"]


def test_as_of_still_prefers_most_recent_restatement_within_the_cutoff():
    """as_of doesn't mean "always use the earliest value" — if there's a genuine
    restatement filed BEFORE the claim's own source filing, that restatement is
    correctly the authoritative contemporaneous figure."""
    facts = [
        _fact("Revenues", 383_000_000_000, FY23_START, FY23_END, "0000320193-23-000001", filed="2023-10-01"),
        _fact("Revenues", 383_285_000_000, FY23_START, FY23_END, "0000320193-23-000106", filed="2023-11-03"),
    ]
    claim = Claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="absolute",
        claimed_value=383_285_000_000,
        period_end=FY23_END,
        period_start=FY23_START,
    )
    result = reconcile(claim, facts, as_of="2023-11-03")
    assert result.verdict == VERDICT_CONSISTENT
    assert result.citations == ["0000320193-23-000106"]


def test_as_of_none_preserves_prior_behavior():
    """Backward compatibility: omitting as_of behaves exactly as before this fix —
    verified against the existing restatement-preference test's own facts shape."""
    facts = [
        _fact("Revenues", 383_285_000_000, FY23_START, FY23_END, "0000320193-23-000106", filed="2023-11-03"),
        _fact("Revenues", 383_500_000_000, FY23_START, FY23_END, "0000320193-24-999999", filed="2024-03-01"),
    ]
    claim = Claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="absolute",
        claimed_value=383_500_000_000,
        period_end=FY23_END,
        period_start=FY23_START,
    )
    result = reconcile(claim, facts)  # no as_of at all
    assert result.verdict == VERDICT_CONSISTENT
    assert result.citations == ["0000320193-24-999999"]


def test_as_of_growth_pct_uses_contemporaneous_values_on_both_sides(aapl_facts):
    """The as_of cutoff must apply to the comparison period's fact lookup too, not
    just the current period's — a growth_pct claim has two fact lookups."""
    restated_prior = _fact(
        "Revenues", 999_999_999_999, FY22_START, FY22_END, "0000320193-24-888888", filed="2024-06-01"
    )
    facts = aapl_facts + [restated_prior]
    claim = Claim(
        ticker=TICKER,
        metric="Revenues",
        comparison_type="growth_pct",
        claimed_value=-2.8,  # real figure, computed from the ORIGINAL FY22 value
        period_end=FY23_END,
        period_start=FY23_START,
        comparison_period_end=FY22_END,
        comparison_period_start=FY22_START,
    )
    result = reconcile(claim, facts, as_of="2023-11-03")  # before the restatement was filed
    assert result.verdict == VERDICT_CONSISTENT
