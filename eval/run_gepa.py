"""Pillar 2 follow-up: a real dspy.GEPA (reflective prompt optimization) run,
closing the "prerequisite built, GEPA never run" gap noted in the README.

`eval/run_comparison.py` showed `BootstrapFewShot` on top of `ChainOfThought`
added nothing over `ChainOfThought` alone (both landed at 0.784 F1) — reasoning
already captured the available signal on this task. GEPA is a different kind of
optimizer: instead of bootstrapping few-shot demos, it reflects on *why* specific
attempts failed (using `dspy_metric_with_feedback`'s diagnostic text) and rewrites
the instruction itself. Worth trying precisely because the bootstrapping approach
already proved to be a dead end here.

Honest caveat, stated up front rather than discovered in the results: GEPA's
reflective step is designed to be guided by a model stronger than the one being
optimized — that's what lets it propose fixes the student model couldn't reach on
its own. This project has exactly one LLM endpoint (`LLM_BASE_URL`, gpt-oss-20b),
so the reflection model here is the *same* model reflecting on its own failures,
not a stronger one. That's a real limitation on how much this run can show, not
a simulated one — reported as such below, whatever the result turns out to be.

Run: python -m eval.run_gepa   (many LLM calls - budget ~15-40 min against a
local single-GPU vLLM server, auto="light" is GEPA's smallest preset)
"""

import time

import dspy

from eval.dataset import load_records, to_dspy_example, train_test_split
from eval.dspy_extractor import ClaimExtractor
from eval.llm_config import configure_dspy
from eval.run_comparison import (
    dspy_metric_with_feedback,
    evaluate,
    noise_floor_half_width,
    print_category_breakdown,
    print_result,
)


def main() -> None:
    lm = configure_dspy()

    records = load_records()
    train_records, test_records = train_test_split(records)
    train_examples = [to_dspy_example(r) for r in train_records]
    test_examples = [to_dspy_example(r) for r in test_records]
    print(f"Dataset: {len(records)} examples -> {len(train_records)} train / {len(test_records)} test\n")

    print("Baseline: zero-shot ChainOfThought (no optimization) on the test set...")
    zero_shot_extractor = ClaimExtractor()
    zero_shot_result = evaluate("dspy_zero_shot", lambda p: zero_shot_extractor(paragraph=p).claims, test_records)
    print_result(zero_shot_result)

    print("\nRunning dspy.GEPA (auto='light', reflection_lm=same gpt-oss-20b endpoint - see module docstring)...")
    print("This makes many LLM calls sequentially against one local GPU. Expect this to take a while.\n")
    start = time.monotonic()

    gepa = dspy.GEPA(
        metric=dspy_metric_with_feedback,
        auto="light",
        reflection_lm=lm,
        num_threads=4,
        track_stats=True,
    )
    optimized = gepa.compile(ClaimExtractor(), trainset=train_examples, valset=test_examples)
    optimize_elapsed = time.monotonic() - start
    print(f"\nGEPA optimization finished in {optimize_elapsed:.0f}s")

    print("\nEvaluating GEPA-optimized program on the held-out test set...")
    gepa_result = evaluate("dspy_gepa", lambda p: optimized(paragraph=p).claims, test_records)
    print_result(gepa_result)
    print_category_breakdown(gepa_result)

    print("\n=== Summary (Pillar 2 GEPA follow-up) ===")
    print_result(zero_shot_result)
    print_result(gepa_result)
    delta_f1 = gepa_result["f1"] - zero_shot_result["f1"]
    print(f"\nGEPA-optimized vs. zero-shot ChainOfThought: delta F1 = {delta_f1:+.3f}")

    n_decisions = gepa_result["tp"] + gepa_result["fp"] + gepa_result["fn"]
    half_width = noise_floor_half_width(gepa_result["f1"], n_decisions)
    print(
        f"\nNoise floor at n={n_decisions} claim-level decisions: F1 +/- {half_width:.3f} (~95%). "
        f"{'Delta is within the noise floor - not distinguishable from chance at this sample size.' if abs(delta_f1) < half_width else 'Delta exceeds the noise floor.'}"
    )

    print("\n=== Optimized instruction GEPA produced ===")
    try:
        predictor = next(iter(optimized.predictors()))
        print(predictor.signature.instructions)
    except Exception as exc:  # noqa: BLE001 - reporting this is best-effort, not the point of the run
        print(f"(could not print optimized instruction: {exc})")


if __name__ == "__main__":
    main()
