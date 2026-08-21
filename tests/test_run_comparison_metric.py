"""Offline tests for eval/run_comparison.py's DSPy metric functions — no LLM calls,
since both dspy_metric and dspy_metric_with_feedback are pure scoring functions
over already-built dspy.Example/dspy.Prediction objects, not extractors."""

import dspy

from eval.run_comparison import dspy_metric, dspy_metric_with_feedback
from eval.schema import ExtractedClaim


def claim(**kwargs):
    defaults = {
        "metric": "revenue", "value": 27.0, "value_unit": "percent",
        "period": "", "comparison_type": "growth_pct", "quote": "",
    }
    return ExtractedClaim(**{**defaults, **kwargs})


def _example_and_pred(gold_claims, predicted_claims):
    example = dspy.Example(claims=gold_claims).with_inputs()
    pred = dspy.Prediction(claims=predicted_claims)
    return example, pred


def test_dspy_metric_with_feedback_matches_the_plain_scalar_metric():
    gold = [claim(metric="revenue", value=27.0)]
    predicted = [claim(metric="revenue", value=27.0)]
    example, pred = _example_and_pred(gold, predicted)

    plain_score = dspy_metric(example, pred)
    result = dspy_metric_with_feedback(example, pred)

    assert result.score == plain_score


def test_dspy_metric_with_feedback_returns_diagnostic_text():
    gold = [claim(metric="Data Center revenue", value=68.0, comparison_type="growth_pct")]
    example, pred = _example_and_pred(gold, [])

    result = dspy_metric_with_feedback(example, pred)

    assert result.score == 0.0
    assert "Data Center revenue" in result.feedback


def test_dspy_metric_with_feedback_accepts_gepa_style_extra_arguments():
    # dspy.GEPA calls metric(example, pred, trace, pred_name, pred_trace) —
    # confirms this can be passed directly as GEPA's `metric` without an adapter.
    gold = [claim(metric="revenue", value=27.0)]
    example, pred = _example_and_pred(gold, gold)

    result = dspy_metric_with_feedback(example, pred, None, "extractor", None)

    assert result.score == 1.0
