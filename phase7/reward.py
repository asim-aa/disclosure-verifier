"""The GRPO reward function — implements the shape designed (before any GPU time
was spent) in docs/phase7-reward-design.md:

    reward = base_verdict_match          # 1.0 if predicted verdict == gold, else 0.0
           + magnitude_shaping           # partial credit for a wrong verdict, scaled
                                          # by how close the gold case was to the
                                          # tolerance boundary (reason_code)
           - false_consistent_penalty    # extra penalty for the dangerous asymmetric
                                          # case: predicting "consistent" when gold
                                          # is not — see eval/reconciler_audit.py

A completion that never produces a parseable "VERDICT: ..." line gets a flat
worst-case reward — a model that can't follow the output format shouldn't score
better than one that follows it and gets the arithmetic wrong.
"""

import re

VERDICTS = ("consistent", "inconsistent", "unverifiable")

_VERDICT_RE = re.compile(r"VERDICT:\s*(consistent|inconsistent|unverifiable)", re.IGNORECASE)

FALSE_CONSISTENT_PENALTY = 1.0
# Strictly below the worst possible *parseable* outcome (0 base - FALSE_CONSISTENT_PENALTY
# = -1.0), so skipping the output format is never preferable to attempting the
# task and getting the dangerous case wrong.
FORMAT_FAILURE_REWARD = -1.0 - FALSE_CONSISTENT_PENALTY

# Shaping for a wrong verdict, keyed by the *gold* case's reason_code — a wrong
# answer on a near-tolerance-boundary case is a different (more forgivable)
# training signal than a wrong answer on an obviously-large miss.
_SHAPING_BY_REASON_CODE = {
    "near_miss": 0.3,
    "large_miss": 0.0,
    "missing_fact": 0.1,
    "ambiguous_period": 0.1,
    "zero_denominator": 0.1,
    "missing_comparison_context": 0.1,
    "unsupported_comparison_type": 0.1,
    "match": 0.0,  # gold was "consistent"; no near/far distinction applies
}


def parse_verdict(completion_text: str) -> str | None:
    """Takes the LAST "VERDICT: ..." occurrence, so a model that reasons through
    a wrong answer, then corrects itself, is scored on its final answer."""
    matches = _VERDICT_RE.findall(completion_text)
    if not matches:
        return None
    return matches[-1].lower()


def compute_reward(completion_text: str, gold_verdict: str, gold_reason_code: str) -> float:
    predicted = parse_verdict(completion_text)
    if predicted is None:
        return FORMAT_FAILURE_REWARD

    base = 1.0 if predicted == gold_verdict else 0.0
    shaping = 0.0 if predicted == gold_verdict else _SHAPING_BY_REASON_CODE.get(gold_reason_code, 0.0)
    penalty = FALSE_CONSISTENT_PENALTY if (predicted == "consistent" and gold_verdict != "consistent") else 0.0

    return base + shaping - penalty
