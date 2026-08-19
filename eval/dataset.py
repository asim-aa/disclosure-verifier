"""Loads eval/labeled_claims.jsonl and produces a deterministic train/test split.

The split is by index modulo 5 (every 5th example -> test), not a contiguous
slice: examples in the file are grouped in blocks by ticker/form (MSFT 10-K, then
MSFT 10-Q, then NVDA 10-K, then NVDA 10-Q), so a contiguous slice would put an
entire company/form-type out of the training set. The modulo split keeps every
group represented in both train and test.
"""

import json
from pathlib import Path

import dspy

from eval.schema import ExtractedClaim

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "eval" / "labeled_claims.jsonl"


def load_records() -> list[dict]:
    with open(DATASET_PATH) as f:
        return [json.loads(line) for line in f]


def train_test_split(records: list[dict]) -> tuple[list[dict], list[dict]]:
    test = [r for i, r in enumerate(records) if i % 5 == 0]
    train = [r for i, r in enumerate(records) if i % 5 != 0]
    return train, test


def to_dspy_example(record: dict) -> dspy.Example:
    gold_claims = [ExtractedClaim(**c) for c in record["claims"]]
    return dspy.Example(paragraph=record["text"], claims=gold_claims).with_inputs("paragraph")
