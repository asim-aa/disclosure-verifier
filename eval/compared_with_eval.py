"""Regression coverage for claim extraction on "[current] ... compared with
[prior]"-shaped sentences, using different connective words ("compared to",
"versus", "up from") and different metrics/companies than anything in
eval/labeled_claims.jsonl.

research/specificity_check.py found real TXN false positives on sentences
shaped like this, and the first diagnosis blamed extraction (see
docs/specificity-check-results.md's "Root cause 2, corrected" for the full
story) — but testing the extractor directly against the real TXN paragraphs
showed it already produced the correct two absolute claims every time. The
actual bug was downstream, in how agents/verification_agent.py resolved a
period for claims that state no period at all (fixed via agents/coordinator.py's
`occurrence` numbering, not here). This script exists to keep confirming that
premise going forward — a generalization regression here would mean extraction
*has* started mishandling this pattern, which would need a different fix than
the one that actually shipped.

Run: python -m eval.compared_with_eval [path/to/optimized_extractor.json]
"""

import sys
from pathlib import Path

from eval.dspy_extractor import ClaimExtractor
from eval.llm_config import configure_dspy
from eval.metrics import feedback_for_example, score_example
from eval.schema import ExtractedClaim

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXTRACTOR_PATH = REPO_ROOT / "eval" / "optimized_extractor.json"

# Synthetic — deliberately not real filing text, so a pass is evidence the model
# generalized the "two absolute claims" rule rather than memorizing training demos.
# Different connective words and metrics than the 2 real TXN training examples.
CASES = [
    {
        "paragraph": "Research and development expense was $1.2 billion compared to $980 million.",
        "claims": [
            ExtractedClaim(metric="research and development expense", value=1_200_000_000.0, value_unit="USD", period="", comparison_type="absolute", quote="Research and development expense was $1.2 billion compared to $980 million."),
            ExtractedClaim(metric="research and development expense", value=980_000_000.0, value_unit="USD", period="", comparison_type="absolute", quote="Research and development expense was $1.2 billion compared to $980 million."),
        ],
    },
    {
        "paragraph": "Total revenue was $8.4 billion versus $7.1 billion.",
        "claims": [
            ExtractedClaim(metric="total revenue", value=8_400_000_000.0, value_unit="USD", period="", comparison_type="absolute", quote="Total revenue was $8.4 billion versus $7.1 billion."),
            ExtractedClaim(metric="total revenue", value=7_100_000_000.0, value_unit="USD", period="", comparison_type="absolute", quote="Total revenue was $8.4 billion versus $7.1 billion."),
        ],
    },
    {
        "paragraph": "Operating margin was 22.3%, up from 19.8%.",
        "claims": [
            ExtractedClaim(metric="operating margin", value=22.3, value_unit="percent", period="", comparison_type="absolute", quote="Operating margin was 22.3%, up from 19.8%."),
            ExtractedClaim(metric="operating margin", value=19.8, value_unit="percent", period="", comparison_type="absolute", quote="Operating margin was 22.3%, up from 19.8%."),
        ],
    },
    {
        "paragraph": "Diluted earnings per share was $2.15 compared with $1.94.",
        "claims": [
            ExtractedClaim(metric="diluted earnings per share", value=2.15, value_unit="USD", period="", comparison_type="absolute", quote="Diluted earnings per share was $2.15 compared with $1.94."),
            ExtractedClaim(metric="diluted earnings per share", value=1.94, value_unit="USD", period="", comparison_type="absolute", quote="Diluted earnings per share was $2.15 compared with $1.94."),
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
        tp, fp, _fn = score_example(predicted, case["claims"])
        ok = tp == len(case["claims"]) and fp == 0
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] case {i}: {case['paragraph']!r}")
        print(f"  {feedback_for_example(predicted, case['claims'])}")
        if not ok:
            print(f"  predicted: {[(c.metric, c.value, c.comparison_type) for c in predicted]}")
        print()

    print(f"{passed}/{len(CASES)} generalization cases passed")


if __name__ == "__main__":
    main()
