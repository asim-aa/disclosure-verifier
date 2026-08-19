"""Compiles the DSPy-optimized ClaimExtractor (same as in run_comparison.py) and
saves it to disk, so Phase 5's extraction agent can load a ready-made optimized
extractor instead of recompiling (and re-hitting the LLM for every bootstrap
attempt) on every run.

Run: python -m eval.compile_and_save
"""

from pathlib import Path

import dspy

from eval.dataset import load_records, to_dspy_example, train_test_split
from eval.dspy_extractor import ClaimExtractor
from eval.llm_config import configure_dspy
from eval.metrics import score_example

REPO_ROOT = Path(__file__).resolve().parent.parent
SAVE_PATH = REPO_ROOT / "eval" / "optimized_extractor.json"


def dspy_metric(example, pred, trace=None):
    tp, fp, fn = score_example(pred.claims, example.claims)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def main():
    configure_dspy()
    records = load_records()
    train_records, _ = train_test_split(records)
    train_examples = [to_dspy_example(r) for r in train_records]

    optimizer = dspy.BootstrapFewShot(metric=dspy_metric, max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=1)
    optimized = optimizer.compile(ClaimExtractor(), trainset=train_examples)

    optimized.save(str(SAVE_PATH), save_program=False)
    print(f"Saved optimized extractor state to {SAVE_PATH}")


if __name__ == "__main__":
    main()
