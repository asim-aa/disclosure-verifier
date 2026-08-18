"""Live checkpoint test for the Numerical Reconciler: calls the actual MCP tool
against real, current EDGAR data for AAPL. Marked `network` — run explicitly with
`pytest -m network`."""

import pytest

from tools.numerical_reconciler import reconcile_claim

pytestmark = pytest.mark.network


def test_true_absolute_claim_reconciles_against_live_edgar():
    result = reconcile_claim(
        ticker="AAPL",
        metric="RevenueFromContractWithCustomerExcludingAssessedTax",
        comparison_type="absolute",
        claimed_value=383_285_000_000,
        period_end="2023-09-30",
        period_start="2022-09-25",
    )
    assert result["verdict"] == "consistent"
    assert result["citations"] == ["0000320193-23-000106"]


def test_false_absolute_claim_is_flagged_against_live_edgar():
    result = reconcile_claim(
        ticker="AAPL",
        metric="RevenueFromContractWithCustomerExcludingAssessedTax",
        comparison_type="absolute",
        claimed_value=999_999_999_999,
        period_end="2023-09-30",
        period_start="2022-09-25",
    )
    assert result["verdict"] == "inconsistent"


def test_true_growth_claim_reconciles_against_live_edgar():
    result = reconcile_claim(
        ticker="AAPL",
        metric="RevenueFromContractWithCustomerExcludingAssessedTax",
        comparison_type="growth_pct",
        claimed_value=-2.8,
        period_end="2023-09-30",
        period_start="2022-09-25",
        comparison_period_end="2022-09-24",
        comparison_period_start="2021-09-26",
    )
    assert result["verdict"] == "consistent"


def test_unverifiable_claim_against_live_edgar():
    result = reconcile_claim(
        ticker="AAPL",
        metric="ThisConceptDoesNotExist",
        comparison_type="absolute",
        claimed_value=1,
        period_end="2023-09-30",
    )
    assert result["verdict"] == "unverifiable"
