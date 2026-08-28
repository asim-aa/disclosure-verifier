"""Pillar 2 follow-up: re-measures the DSPy extractors against a genuinely
never-touched held-out set, closing the repeated-touch caveat flagged since
eval/run_comparison.py was first built.

eval/labeled_claims.jsonl's 16-example test slice has been scored on every run
of run_comparison.py and run_gepa.py during this project's development -
baseline, zero-shot, BootstrapFewShot, GEPA, the stratified breakdown, the
noise-floor check. By this project's own discipline (see README's Pillar 2
section), that means every delta reported against it is better read as "the
best available estimate from a set we've looked at many times" than a clean
holdout result.

eval/labeled_claims_fresh_holdout.jsonl fixes that directly: 25 examples / 59
claims, hand-labeled by reading real AMZN and AAPL 10-K MD&A text (see
build_fresh_holdout.py in the session that built this) - two companies that
appear nowhere in eval/labeled_claims.jsonl, so this set has never been scored
by any extractor, optimizer, or metric in this project before this script
runs it for the first time. Training still uses the full original 78-example
set to bootstrap BootstrapFewShot's demos - reusing training data isn't the
concern here, only repeated *evaluation* against the same test set is.

Run: python -m eval.run_fresh_holdout
"""

import json
from pathlib import Path

import dspy

from eval.baseline_extractor import extract_claims_baseline
from eval.dataset import load_records, to_dspy_example
from eval.dspy_extractor import ClaimExtractor
from eval.llm_config import configure_dspy
from eval.run_comparison import (
    dspy_metric,
    evaluate,
    noise_floor_half_width,
    print_category_breakdown,
    print_result,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_PATH = REPO_ROOT / "eval" / "labeled_claims_fresh_holdout.jsonl"


def load_holdout_records() -> list[dict]:
    with open(HOLDOUT_PATH) as f:
        return [json.loads(line) for line in f]


def main() -> None:
    configure_dspy()

    train_records = load_records()  # the full original 78 - fine to reuse for training
    holdout_records = load_holdout_records()
    print(f"Training set: {len(train_records)} examples (original eval/labeled_claims.jsonl, unchanged)")
    print(f"Fresh holdout: {len(holdout_records)} examples, "
          f"{sum(len(r['claims']) for r in holdout_records)} claims "
          f"(AMZN/AAPL - never scored by anything before this run)\n")

    print("Evaluating baseline (hand-written prompt) on the fresh holdout...")
    baseline_result = evaluate("baseline", extract_claims_baseline, holdout_records)
    print_result(baseline_result)
    print_category_breakdown(baseline_result)

    print("\nEvaluating DSPy zero-shot (ChainOfThought, no demos) on the fresh holdout...")
    zero_shot_extractor = ClaimExtractor()
    zero_shot_result = evaluate(
        "dspy_zero_shot", lambda p: zero_shot_extractor(paragraph=p).claims, holdout_records
    )
    print_result(zero_shot_result)
    print_category_breakdown(zero_shot_result)

    print("\nOptimizing with BootstrapFewShot on the ORIGINAL training set...")
    train_examples = [to_dspy_example(r) for r in train_records]
    optimizer = dspy.BootstrapFewShot(metric=dspy_metric, max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=1)
    optimized_extractor = optimizer.compile(ClaimExtractor(), trainset=train_examples)

    print("\nEvaluating DSPy optimized on the fresh holdout...")
    optimized_result = evaluate(
        "dspy_optimized", lambda p: optimized_extractor(paragraph=p).claims, holdout_records
    )
    print_result(optimized_result)
    print_category_breakdown(optimized_result)

    print("\n=== Summary (fresh, never-touched holdout) ===")
    print_result(baseline_result)
    print_result(zero_shot_result)
    print_result(optimized_result)
    delta_f1 = optimized_result["f1"] - baseline_result["f1"]
    print(f"\nDSPy-optimized vs. hand-written baseline: delta F1 = {delta_f1:+.3f}")

    n_decisions = optimized_result["tp"] + optimized_result["fp"] + optimized_result["fn"]
    half_width = noise_floor_half_width(optimized_result["f1"], n_decisions)
    print(
        f"\nNoise floor at n={n_decisions} claim-level decisions: F1 +/- {half_width:.3f} (~95%). "
        f"{'Delta is within the noise floor - not distinguishable from chance at this sample size.' if abs(delta_f1) < half_width else 'Delta EXCEEDS the noise floor - a real, resolved result.'}"
    )

    print("\n=== Compare to the original (repeatedly-touched) test set ===")
    print("Original (eval/labeled_claims.jsonl, n=16, MSFT/NVDA):")
    print("  baseline=0.730  zero_shot=0.784  optimized=0.784  (delta +0.054, inside +/-0.120 noise floor)")
    print(f"Fresh holdout (this run, n={len(holdout_records)}, AMZN/AAPL):")
    print(f"  baseline={baseline_result['f1']:.3f}  zero_shot={zero_shot_result['f1']:.3f}  "
          f"optimized={optimized_result['f1']:.3f}  (delta {delta_f1:+.3f})")


if __name__ == "__main__":
    main()
