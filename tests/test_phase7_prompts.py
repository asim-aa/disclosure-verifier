from phase7.prompts import build_prompt


def base_example(**overrides):
    defaults = {
        "id": "x", "ticker": "AAPL", "metric": "Revenues", "comparison_type": "absolute",
        "claimed_value": 100.0, "unit": "USD", "tolerance": 0.01,
        "period_end": "2023-09-30", "comparison_period_end": None, "denominator_metric": None,
        "current_value": 100.0, "current_value_unit": "USD", "comparison_value": None,
        "denominator_current_value": None, "denominator_comparison_value": None,
        "gold_verdict": "consistent", "gold_reason_code": "match",
        "gold_computed_value": 100.0, "gold_difference": 0.0, "source": "real", "note": "",
    }
    return {**defaults, **overrides}


def test_absolute_claim_shows_the_underlying_unit():
    prompt = build_prompt(base_example())
    assert "100 USD" in prompt


def test_growth_pct_claimed_value_is_labeled_percent_not_the_underlying_dollar_unit():
    # Regression: build_prompt used to reuse the fact-lookup unit ("USD") to
    # label the claimed_value line even for growth_pct, which is a percentage.
    ex = base_example(
        comparison_type="growth_pct", claimed_value=27.0,
        comparison_period_end="2022-09-30", comparison_value=80.0,
    )
    prompt = build_prompt(ex)
    assert "27 %" in prompt
    assert "27 USD" not in prompt


def test_bps_change_claimed_value_is_labeled_basis_points():
    ex = base_example(
        comparison_type="bps_change", claimed_value=82.0, denominator_metric="Revenues",
        comparison_period_end="2022-09-30", comparison_value=80.0,
        denominator_current_value=200.0, denominator_comparison_value=190.0,
    )
    prompt = build_prompt(ex)
    assert "82 basis points" in prompt
    assert "82 USD" not in prompt


def test_absolute_change_claimed_value_stays_in_the_underlying_unit():
    ex = base_example(
        comparison_type="absolute_change", claimed_value=20.0,
        comparison_period_end="2022-09-30", comparison_value=80.0,
    )
    prompt = build_prompt(ex)
    assert "20 USD" in prompt


def test_missing_value_renders_as_not_found():
    ex = base_example(
        comparison_type="growth_pct", claimed_value=27.0,
        comparison_period_end="2022-09-30", comparison_value=None,
    )
    prompt = build_prompt(ex)
    assert "not found in retrieved data" in prompt


def test_prompt_ends_with_the_verdict_instruction():
    prompt = build_prompt(base_example())
    assert prompt.strip().endswith("VERDICT: <consistent|inconsistent|unverifiable>")
