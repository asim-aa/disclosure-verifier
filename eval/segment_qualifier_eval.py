"""Regression coverage for claim extraction on sentences that state an overall
company-wide figure AND a segment/geography/subsidiary-qualified sub-figure in
the same breath, using different segment names and phrasing than the real
SNPS training example.

research/specificity_check_fresh.py (a second, disjoint control set of tech
companies, none touched by this project's earlier fixes) found a real SNPS
false positive: "Revenues were $7.1 billion, an increase of $926.8 million or
15%, which includes revenues from Ansys of $756.6 million" got the Ansys
sub-figure ($756.6M) extracted under the bare "Revenue" label instead of the
real $7.1B company-wide figure — comparing a segment/acquisition contribution
against the whole company's real revenue, a ~89% "difference" that isn't a
mismatch at all, just a mislabeled number. See docs/specificity-check-
results.md's fresh-holdout writeup for the full story.

Run: python -m eval.segment_qualifier_eval [path/to/optimized_extractor.json]
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
# model generalized "keep the segment/geography qualifier in the metric text"
# rather than memorizing the one real SNPS training example. Different
# segment/geography names and phrasing. `overall_metric`/`overall_values` are
# what the BARE metric name must (and only) carry; `qualified_value` is what
# must show up under a DIFFERENT, qualified metric name instead.
CASES = [
    {
        "paragraph": "Total revenue was $4.2 billion, up $500 million, which includes $310 million contributed by our recent Acme acquisition.",
        "overall_metric": "total revenue",
        "overall_values": {4_200_000_000.0, 500_000_000.0},
        "qualified_value": 310_000_000.0,
        "gold": [
            ExtractedClaim(metric="Total revenue", value=4_200_000_000.0, value_unit="USD", period="", comparison_type="absolute", quote=""),
            ExtractedClaim(metric="Total revenue", value=500_000_000.0, value_unit="USD", period="", comparison_type="absolute_change", quote=""),
            ExtractedClaim(metric="revenue from Acme", value=310_000_000.0, value_unit="USD", period="", comparison_type="absolute", quote=""),
        ],
    },
    {
        "paragraph": "Net sales grew 9% overall, while sales in EMEA declined 6% compared to the prior year.",
        "overall_metric": "net sales",
        "overall_values": {9.0},
        "qualified_value": -6.0,
        "gold": [
            ExtractedClaim(metric="Net sales", value=9.0, value_unit="percent", period="", comparison_type="growth_pct", quote=""),
            ExtractedClaim(metric="EMEA sales", value=-6.0, value_unit="percent", period="", comparison_type="growth_pct", quote=""),
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
        # The real check that matters: no claim uses the BARE overall metric
        # name for the qualified sub-figure's value, and the qualified value
        # shows up under some OTHER (non-bare) metric name - not exact
        # metric-text match, since "revenue from Acme" vs "Acme revenue" are
        # equally correct.
        bare_claims_have_right_values = all(
            c.value not in {case["qualified_value"]}
            for c in predicted
            if c.metric.strip().lower() == case["overall_metric"]
        )
        found_qualified = any(
            c.metric.strip().lower() != case["overall_metric"] and c.value == case["qualified_value"]
            for c in predicted
        )
        ok = bare_claims_have_right_values and found_qualified
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] case {i}: {case['paragraph']!r}")
        print(f"  {feedback_for_example(predicted, case['gold'])}")
        print(f"  predicted: {[(c.metric, c.value, c.comparison_type) for c in predicted]}")
        print()

    print(f"{passed}/{len(CASES)} generalization cases passed")


if __name__ == "__main__":
    main()
