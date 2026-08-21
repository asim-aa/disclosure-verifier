"""Pillar 2 deliverable: baseline (hand-written prompt) vs. DSPy-optimized claim
extraction, measured by precision/recall/F1 on a held-out test set.

Also reports two things a raw pooled F1 number hides:
  - a per-comparison_type breakdown, so a weak category (e.g. bps_change) can't
    quietly fail inside a pooled average that "smiles" because the common
    categories (absolute, growth_pct) are doing fine.
  - a noise-floor estimate for the test set size actually used, since a small
    holdout (this project's is n=16 examples) can make a modest F1 delta
    indistinguishable from noise unless that's checked and reported honestly.

Run: python -m eval.run_comparison
"""

import math
import time

import dspy

from eval.baseline_extractor import extract_claims_baseline
from eval.dataset import load_records, to_dspy_example, train_test_split
from eval.dspy_extractor import ClaimExtractor
from eval.llm_config import configure_dspy
from eval.metrics import (
    merge_category_counts,
    precision_recall_f1,
    score_example,
    score_example_by_category,
)
from eval.schema import ExtractedClaim


def evaluate(name: str, predict_fn, test_records: list[dict]) -> dict:
    total_tp = total_fp = total_fn = 0
    by_category: dict[str, list[int]] = {}
    errors = 0
    start = time.monotonic()

    for record in test_records:
        gold = [ExtractedClaim(**c) for c in record["claims"]]
        try:
            predicted = predict_fn(record["text"])
        except Exception as exc:  # noqa: BLE001 - one bad LLM response (bad JSON, timeout) shouldn't kill the whole eval run
            print(f"  [{name}] extraction error on id={record['id']}: {exc}")
            errors += 1
            predicted = []

        tp, fp, fn = score_example(predicted, gold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        merge_category_counts(by_category, score_example_by_category(predicted, gold))

    elapsed = time.monotonic() - start
    result = precision_recall_f1(total_tp, total_fp, total_fn)
    result.update(
        name=name, errors=errors, n_examples=len(test_records), elapsed_seconds=elapsed, by_category=by_category
    )
    return result


def noise_floor_half_width(p: float, n: int) -> float:
    """SE(p) = sqrt(p(1-p)/n); returns the ~95% half-width (1.96 * SE) in the same
    units as p (e.g. 0.05 = +/-5 points). A measured delta smaller than this is not
    distinguishable from noise at this sample size."""
    if n == 0:
        return float("nan")
    return 1.96 * math.sqrt(p * (1 - p) / n)


def print_category_breakdown(result: dict) -> None:
    print(f"  by comparison_type ({result['name']}):")
    for category, (tp, fp, fn) in sorted(result["by_category"].items()):
        cat_result = precision_recall_f1(tp, fp, fn)
        print(
            f"    {category:<18} precision={cat_result['precision']:.3f}  recall={cat_result['recall']:.3f}  "
            f"f1={cat_result['f1']:.3f}  (tp={tp} fp={fp} fn={fn})"
        )


def dspy_metric(example, pred, trace=None):
    tp, fp, fn = score_example(pred.claims, example.claims)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def print_result(result: dict) -> None:
    print(
        f"{result['name']:>20}: precision={result['precision']:.3f}  recall={result['recall']:.3f}  "
        f"f1={result['f1']:.3f}  (tp={result['tp']} fp={result['fp']} fn={result['fn']}, "
        f"{result['errors']} errors, {result['elapsed_seconds']:.1f}s for {result['n_examples']} examples)"
    )


def main():
    configure_dspy()

    records = load_records()
    train_records, test_records = train_test_split(records)
    print(f"Dataset: {len(records)} examples -> {len(train_records)} train / {len(test_records)} test\n")

    print("Evaluating baseline (hand-written prompt)...")
    baseline_result = evaluate("baseline", extract_claims_baseline, test_records)
    print_result(baseline_result)
    print_category_breakdown(baseline_result)

    print("\nEvaluating DSPy zero-shot (unoptimized signature, no few-shot demos)...")
    zero_shot_extractor = ClaimExtractor()
    zero_shot_result = evaluate(
        "dspy_zero_shot", lambda p: zero_shot_extractor(paragraph=p).claims, test_records
    )
    print_result(zero_shot_result)
    print_category_breakdown(zero_shot_result)

    print("\nOptimizing with BootstrapFewShot on the training set (this calls the LLM many times)...")
    train_examples = [to_dspy_example(r) for r in train_records]
    optimizer = dspy.BootstrapFewShot(metric=dspy_metric, max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=1)
    optimized_extractor = optimizer.compile(ClaimExtractor(), trainset=train_examples)

    print("\nEvaluating DSPy optimized...")
    optimized_result = evaluate(
        "dspy_optimized", lambda p: optimized_extractor(paragraph=p).claims, test_records
    )
    print_result(optimized_result)
    print_category_breakdown(optimized_result)

    print("\n=== Summary (Pillar 2 deliverable) ===")
    print_result(baseline_result)
    print_result(zero_shot_result)
    print_result(optimized_result)
    delta_f1 = optimized_result["f1"] - baseline_result["f1"]
    print(f"\nDSPy-optimized vs. hand-written baseline: delta F1 = {delta_f1:+.3f}")

    # Noise-floor honesty check: is that delta even distinguishable from sample
    # noise at this test set size, or could an optimizer (or a lucky run) have
    # "manufactured" it? n here is the total number of claim-level tp+fp+fn
    # decisions, not just the 16 paragraphs — that's the actual sample size the
    # precision/recall proportions are computed over.
    n_decisions = optimized_result["tp"] + optimized_result["fp"] + optimized_result["fn"]
    half_width = noise_floor_half_width(optimized_result["f1"], n_decisions)
    print(
        f"\nNoise floor at n={n_decisions} claim-level decisions: F1 +/- {half_width:.3f} "
        f"(~95%). {'Delta is within the noise floor — not distinguishable from chance at this sample size.' if abs(delta_f1) < half_width else 'Delta exceeds the noise floor.'}"
    )


if __name__ == "__main__":
    main()
