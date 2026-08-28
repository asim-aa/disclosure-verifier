from phase7.reward import (
    FALSE_CONSISTENT_PENALTY,
    FORMAT_FAILURE_REWARD,
    compute_reward,
    parse_verdict,
)


def test_parse_verdict_reads_the_trailing_tag():
    assert parse_verdict("Some reasoning here.\nVERDICT: consistent") == "consistent"


def test_parse_verdict_is_case_insensitive():
    assert parse_verdict("VERDICT: Inconsistent") == "inconsistent"


def test_parse_verdict_takes_the_last_occurrence():
    # A model that reasons its way to a self-correction should be scored on its
    # final answer, not an earlier draft.
    text = "First I thought VERDICT: consistent, but recomputing... VERDICT: inconsistent"
    assert parse_verdict(text) == "inconsistent"


def test_parse_verdict_returns_none_with_no_tag():
    assert parse_verdict("I think this claim looks fine.") is None


def test_correct_verdict_scores_one():
    assert compute_reward("... VERDICT: consistent", gold_verdict="consistent", gold_reason_code="match") == 1.0


def test_wrong_verdict_near_miss_gets_partial_credit():
    reward = compute_reward("... VERDICT: consistent", gold_verdict="inconsistent", gold_reason_code="near_miss")
    # base=0, shaping=+0.3, but predicted=consistent and gold!=consistent -> penalty applies too
    assert reward == 0.0 + 0.3 - FALSE_CONSISTENT_PENALTY


def test_wrong_verdict_large_miss_gets_no_partial_credit():
    reward = compute_reward("... VERDICT: unverifiable", gold_verdict="inconsistent", gold_reason_code="large_miss")
    assert reward == 0.0


def test_false_consistent_is_penalized_beyond_a_plain_wrong_answer():
    # The dangerous asymmetric case: predicting consistent when gold is not.
    false_consistent = compute_reward("VERDICT: consistent", gold_verdict="inconsistent", gold_reason_code="large_miss")
    plain_wrong = compute_reward("VERDICT: unverifiable", gold_verdict="inconsistent", gold_reason_code="large_miss")
    assert false_consistent < plain_wrong
    assert false_consistent == 0.0 + 0.0 - FALSE_CONSISTENT_PENALTY


def test_predicting_consistent_when_gold_is_consistent_is_never_penalized():
    assert compute_reward("VERDICT: consistent", gold_verdict="consistent", gold_reason_code="match") == 1.0


def test_unparseable_completion_gets_the_flat_format_failure_reward():
    assert compute_reward("I am not sure.", gold_verdict="consistent", gold_reason_code="match") == FORMAT_FAILURE_REWARD


def test_format_failure_is_worse_than_any_parseable_wrong_answer():
    worst_parseable = compute_reward("VERDICT: consistent", gold_verdict="inconsistent", gold_reason_code="large_miss")
    assert FORMAT_FAILURE_REWARD < worst_parseable
