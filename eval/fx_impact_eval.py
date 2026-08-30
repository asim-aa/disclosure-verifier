"""Regression coverage for claim extraction on "was positively/negatively
impacted by X%"-shaped sentences, using different factor words ("benefited
from", "hurt by") and different metrics than the real CRM training example.

research/specificity_check.py found real CRM false positives on sentences
like "Total revenues ... was positively impacted by approximately one percent
in foreign currency fluctuations" — extraction read this as a growth_pct claim
of 1% for revenue, when the sentence actually states only currency's *partial
contribution* to revenue's real (separately-stated) growth, not revenue's own
total movement. See docs/specificity-check-results.md's "Root cause 6" for the
period-resolution bug this was originally confused with, and the write-up of
this extraction-semantics fix.

This also caught a real, separate infrastructure gap: eval/optimized_extractor.json
freezes its signature instructions at compile time (eval/compile_and_save.py),
so a live edit to eval/dspy_extractor.py's docstring has NO effect on
agents.extraction_agent.RealExtractionAgent (which loads that frozen file)
until compile_and_save is re-run. Confirmed directly: the same paragraph
extracted correctly zero-shot immediately after the docstring edit, but still
wrongly with the stale compiled artifact loaded, until recompiling. Anyone
changing eval/dspy_extractor.py's docstring must re-run
`python -m eval.compile_and_save` for it to reach production - this script
verifies the *persisted* artifact, not the live signature, for exactly that
reason.

Run: python -m eval.fx_impact_eval [path/to/optimized_extractor.json]
"""

import sys
from pathlib import Path

from eval.dspy_extractor import ClaimExtractor
from eval.llm_config import configure_dspy
from eval.metrics import feedback_for_example, score_example

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXTRACTOR_PATH = REPO_ROOT / "eval" / "optimized_extractor.json"

# Synthetic — deliberately not real filing text, so a pass is evidence the model
# generalized "a factor-contribution percentage isn't the metric's own growth",
# not memorization of the one real CRM training example. Different factor words
# and metrics than that example.
CASES = [
    {
        "paragraph": "Operating margin was negatively impacted by approximately 2 percentage points due to higher freight costs.",
        "claims": [],
    },
    {
        "paragraph": "Gross margin benefited from favorable component pricing by roughly 3 percent compared to the prior year.",
        "claims": [],
    },
    {
        "paragraph": "Net income was hurt by approximately 5 percent as a result of restructuring charges recognized during the period.",
        "claims": [],
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
