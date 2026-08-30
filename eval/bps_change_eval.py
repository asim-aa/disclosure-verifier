"""Regression coverage for claim extraction on "increased/decreased by X
percentage points"-shaped sentences, using different metrics and phrasing
than the real CSCO training example — the first (and, before this fix, only)
real bps_change example in the training set at all; see
eval/labeled_claims.jsonl id=81.

research/specificity_check.py found a real CSCO false positive: "Total gross
margin increased by 0.2 percentage points" extracted as a growth_pct claim of
0.2% for GrossProfit (a dollar concept) — comparing a margin-ratio point
change against a raw dollar figure's relative growth, two completely
different numbers. See docs/specificity-check-results.md's "Root cause 8" for
the full story. bps_change is correct-by-construction safe even before full
denominator_metric plumbing exists (see agents/verification_agent.py and
tools/reconciler.py's `_reconcile_bps_change`): with no denominator_metric
set, it returns unverifiable rather than computing anything wrong — the fix's
whole job is just getting comparison_type right at extraction time, not
building new verification machinery.

Run: python -m eval.bps_change_eval [path/to/optimized_extractor.json]
"""

import sys
from pathlib import Path

from eval.dspy_extractor import ClaimExtractor
from eval.llm_config import configure_dspy
from eval.metrics import feedback_for_example
from eval.schema import ExtractedClaim

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXTRACTOR_PATH = REPO_ROOT / "eval" / "optimized_extractor.json"

# Synthetic — deliberately not real filing text, so a pass is evidence the
# model generalized "percentage points/bps means bps_change, not growth_pct"
# rather than memorizing the one real CSCO training example. Different
# metrics and phrasing (including bare "bps").
CASES = [
    {
        "paragraph": "Operating margin expanded by 150 basis points compared to the prior year.",
        "claims": [
            ExtractedClaim(metric="operating margin", value=150.0, value_unit="bps", period="", comparison_type="bps_change", quote="Operating margin expanded by 150 basis points compared to the prior year."),
        ],
    },
    {
        "paragraph": "Net margin contracted by 0.5 percentage points due to higher input costs.",
        "claims": [
            ExtractedClaim(metric="net margin", value=-50.0, value_unit="bps", period="", comparison_type="bps_change", quote="Net margin contracted by 0.5 percentage points due to higher input costs."),
        ],
    },
]


def main() -> None:
    extractor_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EXTRACTOR_PATH
    configure_dspy()
    extractor = ClaimExtractor()
    if extractor_path.exists():
        extractor.load(str(extractor_path))
        print(f"Loaded {extractor_path}\n")
    else:
        print(f"{extractor_path} not found — testing the zero-shot signature\n")

    passed = 0
    for i, case in enumerate(CASES):
        predicted = extractor(paragraph=case["paragraph"]).claims
        # Only comparison_type matters here (the exact bps magnitude/sign a
        # model infers from "expanded"/"contracted" wording is secondary to
        # not miscategorizing the claim as growth_pct against a dollar concept).
        ok = len(predicted) >= 1 and all(c.comparison_type == "bps_change" for c in predicted)
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] case {i}: {case['paragraph']!r}")
        print(f"  {feedback_for_example(predicted, case['claims'])}")
        print(f"  predicted: {[(c.metric, c.value, c.value_unit, c.comparison_type) for c in predicted]}")
        print()

    print(f"{passed}/{len(CASES)} generalization cases passed")


if __name__ == "__main__":
    main()
