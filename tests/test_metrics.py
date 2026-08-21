import pytest

from eval.metrics import (
    claims_match,
    merge_category_counts,
    precision_recall_f1,
    score_example,
    score_example_by_category,
)
from eval.schema import ExtractedClaim


def claim(**kwargs):
    defaults = {"metric": "revenue", "value": 27.0, "value_unit": "percent", "period": "", "comparison_type": "growth_pct", "quote": ""}
    return ExtractedClaim(**{**defaults, **kwargs})


def test_identical_claims_match():
    assert claims_match(claim(), claim())


def test_minor_metric_wording_difference_still_matches():
    assert claims_match(claim(metric="gross margin"), claim(metric="Gross margin percentage"))


def test_generic_metric_does_not_match_a_more_specific_one():
    """A bare 'Revenue' prediction must NOT match a segment-qualified gold label
    like 'Microsoft Cloud revenue' just because they share the word 'revenue' —
    those are different metrics, and conflating them would silently inflate
    measured precision/recall."""
    assert not claims_match(claim(metric="Revenue"), claim(metric="Microsoft Cloud revenue"))


def test_completely_different_metric_does_not_match():
    assert not claims_match(claim(metric="LinkedIn revenue"), claim(metric="Xbox hardware revenue"))


def test_different_comparison_type_does_not_match():
    assert not claims_match(claim(comparison_type="growth_pct"), claim(comparison_type="absolute"))


def test_different_unit_does_not_match():
    assert not claims_match(claim(value_unit="percent"), claim(value_unit="USD"))


def test_value_within_tolerance_matches():
    assert claims_match(claim(value=100.0), claim(value=100.5))


def test_value_outside_tolerance_does_not_match():
    assert not claims_match(claim(value=27.0), claim(value=50.0))


def test_score_example_all_correct():
    gold = [claim(metric="revenue", value=27.0), claim(metric="margin", value=66.0, comparison_type="absolute", value_unit="percent")]
    predicted = [claim(metric="revenue", value=27.0), claim(metric="margin", value=66.0, comparison_type="absolute", value_unit="percent")]
    tp, fp, fn = score_example(predicted, gold)
    assert (tp, fp, fn) == (2, 0, 0)


def test_score_example_missed_claim_is_false_negative():
    gold = [claim(metric="revenue", value=27.0), claim(metric="margin", value=66.0)]
    predicted = [claim(metric="revenue", value=27.0)]
    tp, fp, fn = score_example(predicted, gold)
    assert (tp, fp, fn) == (1, 0, 1)


def test_score_example_extra_claim_is_false_positive():
    gold = [claim(metric="revenue", value=27.0)]
    predicted = [claim(metric="revenue", value=27.0), claim(metric="margin", value=66.0)]
    tp, fp, fn = score_example(predicted, gold)
    assert (tp, fp, fn) == (1, 1, 0)


def test_score_example_true_negative_no_claims():
    tp, fp, fn = score_example([], [])
    assert (tp, fp, fn) == (0, 0, 0)


def test_score_example_does_not_double_count_duplicate_predictions():
    gold = [claim(metric="revenue", value=27.0)]
    predicted = [claim(metric="revenue", value=27.0), claim(metric="revenue", value=27.0)]
    tp, fp, fn = score_example(predicted, gold)
    assert (tp, fp, fn) == (1, 1, 0)


def test_precision_recall_f1_basic():
    result = precision_recall_f1(total_tp=8, total_fp=2, total_fn=2)
    assert result["precision"] == pytest.approx(0.8)
    assert result["recall"] == pytest.approx(0.8)
    assert result["f1"] == pytest.approx(0.8)


def test_precision_recall_f1_no_predictions_and_no_gold_is_perfect():
    result = precision_recall_f1(total_tp=0, total_fp=0, total_fn=0)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


# ---------- stratified (per comparison_type) scoring ----------


def test_score_by_category_separates_categories():
    gold = [
        claim(metric="revenue", value=27.0, comparison_type="growth_pct"),
        claim(metric="revenue", value=100.0, comparison_type="absolute"),
    ]
    predicted = [
        claim(metric="revenue", value=27.0, comparison_type="growth_pct"),  # matches
        # absolute claim missing entirely -> fn in that category
    ]
    result = score_example_by_category(predicted, gold)
    assert result["growth_pct"] == (1, 0, 0)
    assert result["absolute"] == (0, 0, 1)


def test_score_by_category_attributes_false_positive_to_predicted_type():
    gold = [claim(metric="revenue", value=27.0, comparison_type="growth_pct")]
    predicted = [
        claim(metric="revenue", value=27.0, comparison_type="growth_pct"),  # matches
        claim(metric="margin", value=200.0, comparison_type="bps_change"),  # extra, no gold match
    ]
    result = score_example_by_category(predicted, gold)
    assert result["growth_pct"] == (1, 0, 0)
    assert result["bps_change"] == (0, 1, 0)


def test_score_by_category_a_weak_category_does_not_hide_in_a_strong_pooled_average():
    """The whole point of stratification: a category that's failing completely can
    still produce a fine-looking pooled score if it's rare relative to a category
    that's doing well."""
    gold = [
        claim(metric="revenue", value=27.0, comparison_type="growth_pct"),
        claim(metric="revenue", value=28.0, comparison_type="growth_pct"),
        claim(metric="revenue", value=29.0, comparison_type="growth_pct"),
        claim(metric="margin", value=200.0, comparison_type="bps_change"),
    ]
    predicted = [
        claim(metric="revenue", value=27.0, comparison_type="growth_pct"),
        claim(metric="revenue", value=28.0, comparison_type="growth_pct"),
        claim(metric="revenue", value=29.0, comparison_type="growth_pct"),
        # bps_change claim entirely missed
    ]
    pooled = score_example(predicted, gold)
    assert pooled == (3, 0, 1)  # looks fine pooled: 75% recall

    by_category = score_example_by_category(predicted, gold)
    assert by_category["bps_change"] == (0, 0, 1)  # 0% recall, invisible in the pooled number


def test_merge_category_counts_accumulates_across_examples():
    total: dict = {}
    merge_category_counts(total, {"growth_pct": (2, 1, 0)})
    merge_category_counts(total, {"growth_pct": (1, 0, 1), "absolute": (3, 0, 0)})
    assert total["growth_pct"] == [3, 1, 1]
    assert total["absolute"] == [3, 0, 0]
