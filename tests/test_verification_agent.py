from agents.verification_agent import RealVerificationAgent
from eval.schema import ExtractedClaim
from tools.schema import (
    VERDICT_CONSISTENT,
    VERDICT_INCONSISTENT,
    VERDICT_UNVERIFIABLE,
    FinancialFact,
)


def fact(concept, value, period_start, period_end, unit="USD", accn="a", filed="2026-01-01"):
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
