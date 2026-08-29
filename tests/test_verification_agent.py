from agents.verification_agent import RealVerificationAgent
from eval.schema import ExtractedClaim
from tools.schema import (
    VERDICT_CONSISTENT,
    VERDICT_INCONSISTENT,
    VERDICT_UNVERIFIABLE,
    FinancialFact,
)


def fact(concept, value, period_start, period_end, unit="USD", accn="a", filed="2026-01-01", fiscal_year=2026):
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
        fiscal_period="FY",
        form="10-K",
        filed=filed,
        accession_number=accn,
    )


def extracted(metric, value, value_unit, comparison_type, period="", quote=""):
    return ExtractedClaim(metric=metric, value=value, value_unit=value_unit, period=period, comparison_type=comparison_type, quote=quote)


AGENT = RealVerificationAgent()


def test_absolute_claim_resolves_and_verifies_consistent():
    facts = [fact("RevenueFromContractWithCustomerExcludingAssessedTax", 215_900_000_000, "2025-01-01", "2026-01-25")]
    outcome = AGENT.verify(
        extracted("revenue", 215_900_000_000, "USD", "absolute"), "NVDA", facts
    )
    assert outcome.verdict == VERDICT_CONSISTENT
    assert outcome.citations


def test_absolute_claim_resolves_and_verifies_inconsistent():
    facts = [fact("RevenueFromContractWithCustomerExcludingAssessedTax", 215_900_000_000, "2025-01-01", "2026-01-25")]
    outcome = AGENT.verify(
        extracted("revenue", 999_999_999_999, "USD", "absolute"), "NVDA", facts
    )
    assert outcome.verdict == VERDICT_INCONSISTENT


def test_unresolvable_metric_is_unverifiable_with_clear_reason():
    facts = [fact("RevenueFromContractWithCustomerExcludingAssessedTax", 100, "2025-01-01", "2026-01-25")]
    outcome = AGENT.verify(
        extracted("Azure and other cloud services revenue", 41, "percent", "growth_pct"), "MSFT", facts
    )
    assert outcome.verdict == VERDICT_UNVERIFIABLE
    assert "resolve" in outcome.explanation.lower()
    assert outcome.citations == []


def test_growth_pct_claim_needs_comparison_period():
    """Only one period of data available -> can't check a growth_pct claim, must
    come back unverifiable rather than silently treating it as absolute."""
    facts = [fact("RevenueFromContractWithCustomerExcludingAssessedTax", 100, "2025-01-01", "2026-01-25")]
    outcome = AGENT.verify(
        extracted("revenue", 10, "percent", "growth_pct"), "NVDA", facts
    )
    assert outcome.verdict == VERDICT_UNVERIFIABLE
    assert "comparison period" in outcome.explanation.lower()


def test_growth_pct_claim_resolves_with_two_periods():
    facts = [
        fact("RevenueFromContractWithCustomerExcludingAssessedTax", 215_900_000_000, "2025-01-27", "2026-01-25"),
        fact("RevenueFromContractWithCustomerExcludingAssessedTax", 130_500_000_000, "2024-01-29", "2025-01-26"),
    ]
    # real growth: (215.9-130.5)/130.5*100 = 65.44%
    outcome = AGENT.verify(
        extracted("revenue", 65.4, "percent", "growth_pct"), "NVDA", facts
    )
    assert outcome.verdict == VERDICT_CONSISTENT


def test_pure_unit_percentage_claim_converts_correctly():
    """The single most important regression test in this file: without the
    percent->fraction conversion, this would compare 19.0 against 0.19 and
    always report inconsistent even for a genuinely correct claim."""
    facts = [fact("EffectiveIncomeTaxRateContinuingOperations", 0.19, "2025-01-01", "2026-01-25", unit="pure")]
    outcome = AGENT.verify(
        extracted("effective tax rate", 19.0, "percent", "absolute"), "MSFT", facts
    )
    assert outcome.verdict == VERDICT_CONSISTENT


def test_pure_unit_percentage_claim_correctly_flags_wrong_value():
    facts = [fact("EffectiveIncomeTaxRateContinuingOperations", 0.19, "2025-01-01", "2026-01-25", unit="pure")]
    outcome = AGENT.verify(
        extracted("effective tax rate", 45.0, "percent", "absolute"), "MSFT", facts
    )
    assert outcome.verdict == VERDICT_INCONSISTENT


def test_as_of_anchors_to_the_source_filing_not_a_later_one():
    """Regression test replicating the real NVDA bug this fix was built for: a
    claim extracted from a 10-K (filed 2026-02-25) states the 10-K's own FY2026
    revenue. A 10-Q reporting Q1 FY2027 has since been filed (2026-05-20) with a
    more recent period_end. Without as_of, verification would compare the FY2026
    claim against the newer quarterly figure and wrongly call it inconsistent."""
    facts = [
        fact(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            215_900_000_000, "2025-01-27", "2026-01-25", filed="2026-02-25",
        ),
        fact("Revenues", 81_615_000_000, "2026-01-26", "2026-04-26", filed="2026-05-20"),
    ]
    outcome = AGENT.verify(
        extracted("revenue", 215_900_000_000, "USD", "absolute"),
        "NVDA", facts, as_of="2026-02-25",
    )
    assert outcome.verdict == VERDICT_CONSISTENT


# ---------- percent claim against a non-ratio (e.g. dollar) concept ----------
#
# Real bug, confirmed against TXN's actual MD&A: "gross profit decreased to
# 57.0% from 58.1%." resolves "gross profit margin" to GrossProfit, a real XBRL
# concept - but one reported as a raw dollar figure, never a ratio. Comparing a
# claimed 57.0 against a $10.083B fact isn't a real mismatch, it's an
# unresolvable claim: no concept mapping can make that comparison meaningful.


def test_percent_claim_against_a_dollar_concept_is_unverifiable_not_wrong():
    facts = [fact("GrossProfit", 10_083_000_000, "2025-01-01", "2025-12-31", unit="USD")]
    outcome = AGENT.verify(
        extracted("gross profit margin", 57.0, "percent", "absolute"), "TXN", facts
    )
    assert outcome.verdict == VERDICT_UNVERIFIABLE
    assert "percentage" in outcome.explanation.lower() or "ratio" in outcome.explanation.lower()


def test_percent_claim_against_a_pure_concept_still_verifies_normally():
    """The guard must only block genuinely incompatible units - a percent claim
    against a real "pure" ratio concept (the actual supported case) must keep
    working exactly as before."""
    facts = [fact("EffectiveIncomeTaxRateContinuingOperations", 0.19, "2025-01-01", "2026-01-25", unit="pure")]
    outcome = AGENT.verify(
        extracted("effective tax rate", 19.0, "percent", "absolute"), "MSFT", facts
    )
    assert outcome.verdict == VERDICT_CONSISTENT


def test_dollar_claim_against_a_dollar_concept_is_unaffected_by_the_guard():
    facts = [fact("GrossProfit", 10_083_000_000, "2025-01-01", "2025-12-31", unit="USD")]
    outcome = AGENT.verify(
        extracted("gross profit", 10_083_000_000, "USD", "absolute"), "TXN", facts
    )
    assert outcome.verdict == VERDICT_CONSISTENT


# ---------- occurrence: the "compared with" pairing bug ----------
#
# Real bug, found via research/specificity_check.py: a sentence like "Net income
# was $5.00 billion compared with $4.80 billion." extracts correctly as two
# separate absolute claims (confirmed directly against the live extractor - the
# extraction step was never the problem). But neither claim states an explicit
# period, so independently verifying each with occurrence=0 would resolve both
# against the SAME current-period fact, wrongly flagging the true $4.80B prior-
# year claim as inconsistent. Coordinator.run numbers same-metric/same-quote
# repeats and passes `occurrence` through for exactly this reason.


def test_occurrence_zero_checks_against_the_current_period_as_before():
    facts = [
        fact("NetIncomeLoss", 5_000_000_000, "2025-01-01", "2025-12-31"),
        fact("NetIncomeLoss", 4_800_000_000, "2024-01-01", "2024-12-31"),
    ]
    outcome = AGENT.verify(
        extracted("net income", 5_000_000_000, "USD", "absolute"), "TXN", facts, occurrence=0
    )
    assert outcome.verdict == VERDICT_CONSISTENT


def test_occurrence_one_checks_the_repeated_absolute_claim_against_the_prior_period():
    """The actual fix: the second same-metric absolute claim with no stated
    period (real TXN case: "$4.80 billion" from "compared with $4.80 billion")
    is checked against the comparison fact, not re-resolved against current."""
    facts = [
        fact("NetIncomeLoss", 5_000_000_000, "2025-01-01", "2025-12-31"),
        fact("NetIncomeLoss", 4_800_000_000, "2024-01-01", "2024-12-31"),
    ]
    outcome = AGENT.verify(
        extracted("net income", 4_800_000_000, "USD", "absolute"), "TXN", facts, occurrence=1
    )
    assert outcome.verdict == VERDICT_CONSISTENT


def test_occurrence_one_still_catches_a_genuinely_wrong_repeated_claim():
    """The fix must not become a rubber stamp - a repeated absolute claim that
    doesn't match the prior-period fact either should still be flagged."""
    facts = [
        fact("NetIncomeLoss", 5_000_000_000, "2025-01-01", "2025-12-31"),
        fact("NetIncomeLoss", 4_800_000_000, "2024-01-01", "2024-12-31"),
    ]
    outcome = AGENT.verify(
        extracted("net income", 999_000_000, "USD", "absolute"), "TXN", facts, occurrence=1
    )
    assert outcome.verdict == VERDICT_INCONSISTENT


def test_occurrence_one_with_no_comparison_fact_is_unverifiable_not_wrong():
    facts = [fact("NetIncomeLoss", 5_000_000_000, "2025-01-01", "2025-12-31")]
    outcome = AGENT.verify(
        extracted("net income", 4_800_000_000, "USD", "absolute"), "TXN", facts, occurrence=1
    )
    assert outcome.verdict == VERDICT_UNVERIFIABLE


def test_occurrence_one_does_not_apply_when_the_claim_states_its_own_period():
    """occurrence only kicks in for claims with no stated period - a claim that
    DOES state its period (e.g. "fiscal 2024") already resolves correctly via
    the existing period_hint machinery and must not be swapped a second time."""
    facts = [
        fact("NetIncomeLoss", 5_000_000_000, "2025-01-01", "2025-12-31", fiscal_year=2025),
        fact("NetIncomeLoss", 4_800_000_000, "2024-01-01", "2024-12-31", fiscal_year=2024),
    ]
    outcome = AGENT.verify(
        extracted("net income", 4_800_000_000, "USD", "absolute", period="fiscal 2024"),
        "TXN", facts, occurrence=1,
    )
    assert outcome.verdict == VERDICT_CONSISTENT


def test_occurrence_one_does_not_apply_to_non_absolute_comparison_types():
    """A growth_pct/absolute_change/bps_change claim already uses comparison_fact
    for its own computation - the occurrence swap is absolute-only."""
    facts = [
        fact("NetIncomeLoss", 5_000_000_000, "2025-01-01", "2025-12-31"),
        fact("NetIncomeLoss", 4_800_000_000, "2024-01-01", "2024-12-31"),
    ]
    # real growth: (5.0-4.8)/4.8*100 = 4.17%
    outcome = AGENT.verify(
        extracted("net income", 4.17, "percent", "growth_pct"), "TXN", facts, occurrence=1
    )
    assert outcome.verdict == VERDICT_CONSISTENT


def test_without_as_of_the_later_filing_wins_documenting_why_it_matters():
    facts = [
        fact(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            215_900_000_000, "2025-01-27", "2026-01-25", filed="2026-02-25",
        ),
        fact("Revenues", 81_615_000_000, "2026-01-26", "2026-04-26", filed="2026-05-20"),
    ]
    outcome = AGENT.verify(
        extracted("revenue", 215_900_000_000, "USD", "absolute"), "NVDA", facts,
    )
    assert outcome.verdict == VERDICT_INCONSISTENT
