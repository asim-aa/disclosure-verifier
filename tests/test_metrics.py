import pytest

from eval.metrics import claims_match, precision_recall_f1, score_example
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
