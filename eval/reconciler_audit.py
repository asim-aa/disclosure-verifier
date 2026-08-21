"""Audits the Numerical Reconciler's own correctness in isolation from claim
extraction — before it's trusted as a Phase 7 RLVR reward function.

Why this exists, separate from Pillar 2's extraction F1: end-to-end accuracy is
bounded by *both* extraction quality and verification quality, and a claim-
extraction score alone can't tell you which one is the ceiling. If the
Reconciler itself has any exploitable seam — a tolerance boundary that's too
loose, a sign or magnitude confusion it doesn't catch, a comparison_type it
mishandles — then a policy trained against it as a reward signal (GRPO) doesn't
just get a noisy reward, it gets a *targeted* one: optimization pressure will
find and amplify whatever gap exists between what the Reconciler measures and
what it's meant to measure. This audit exists to find that gap now, cheaply,
before any GPU time is spent trusting the Reconciler's verdict as ground truth.

Every case here is either:
  - a real, hand-verified number (reused from tests/test_reconciler.py's known
    AAPL FY2022/FY2023 figures), or
  - an adversarial construction targeting a *specific* way the Reconciler could
    plausibly be fooled — sign flips, order-of-magnitude confusion, mislabeled
    comparison_type, and exact tolerance-boundary behavior.

The single number that matters most is the false-CONSISTENT rate: how often the
Reconciler says "consistent" for a claim that is actually wrong. That's the
dangerous failure mode for a reward function — a false "inconsistent" or
"unverifiable" just costs training signal; a false "consistent" actively
teaches a policy that a wrong answer was right.

Run: python -m eval.reconciler_audit
"""

from dataclasses import dataclass

from tools.reconciler import reconcile
from tools.schema import Claim, FinancialFact

TICKER = "AAPL"
CIK = "0000320193"
FY22_START, FY22_END = "2021-09-26", "2022-09-24"
FY23_START, FY23_END = "2022-09-25", "2023-09-30"

# Real, hand-verified figures (pulled live from EDGAR while building Phase 3):
#   FY2022 Revenue=$394,328,000,000  GrossProfit=$170,782,000,000
#   FY2023 Revenue=$383,285,000,000  GrossProfit=$169,148,000,000
#   -> real growth FY23 vs FY22: -2.8005%   real bps change: +82.15 bps


def _fact(concept, value, period_start, period_end, accn, filed="2023-11-03", unit="USD"):
    return FinancialFact(
        ticker=TICKER, cik=CIK, concept=concept, label=concept, value=value, unit=unit,
        period_start=period_start, period_end=period_end, fiscal_year=2023, fiscal_period="FY",
        form="10-K", filed=filed, accession_number=accn,
    )


FACTS = [
    _fact("Revenues", 394_328_000_000, FY22_START, FY22_END, "0000320193-22-000108", filed="2022-10-28"),
    _fact("Revenues", 383_285_000_000, FY23_START, FY23_END, "0000320193-23-000106", filed="2023-11-03"),
    _fact("GrossProfit", 170_782_000_000, FY22_START, FY22_END, "0000320193-22-000108", filed="2022-10-28"),
    _fact("GrossProfit", 169_148_000_000, FY23_START, FY23_END, "0000320193-23-000106", filed="2023-11-03"),
    _fact("EffectiveIncomeTaxRateContinuingOperations", 0.147, FY23_START, FY23_END, "0000320193-23-000106", unit="pure"),
]


@dataclass
class Case:
    label: str
    category: str  # "known_good" | "known_bad" | "adversarial"
    claim: Claim
    expected_verdict: str
    note: str = ""


CASES: list[Case] = [
    # ---------- known-good: real, correct claims ----------
    Case(
        "real FY23 revenue, correct",
        "known_good",
        Claim(ticker=TICKER, metric="Revenues", comparison_type="absolute",
              claimed_value=383_285_000_000, period_end=FY23_END, period_start=FY23_START),
        "consistent",
    ),
    Case(
        "real FY23 growth, correct (-2.8%)",
        "known_good",
        Claim(ticker=TICKER, metric="Revenues", comparison_type="growth_pct",
              claimed_value=-2.8, period_end=FY23_END, period_start=FY23_START,
              comparison_period_end=FY22_END, comparison_period_start=FY22_START),
        "consistent",
    ),
    Case(
        "real FY23 margin bps change, correct (+82 bps)",
        "known_good",
        Claim(ticker=TICKER, metric="GrossProfit", denominator_metric="Revenues", comparison_type="bps_change",
              claimed_value=82.0, period_end=FY23_END, period_start=FY23_START,
              comparison_period_end=FY22_END, comparison_period_start=FY22_START),
        "consistent",
    ),
    Case(
        "real FY23 revenue absolute_change, correct (-$11.043B)",
        "known_good",
        Claim(ticker=TICKER, metric="Revenues", comparison_type="absolute_change",
              claimed_value=-11_043_000_000, period_end=FY23_END, period_start=FY23_START,
              comparison_period_end=FY22_END, comparison_period_start=FY22_START),
        "consistent",
    ),

    # ---------- known-bad: real, plainly wrong claims ----------
    Case(
        "fabricated FY23 revenue (way off)",
        "known_bad",
        Claim(ticker=TICKER, metric="Revenues", comparison_type="absolute",
              claimed_value=500_000_000_000, period_end=FY23_END, period_start=FY23_START),
        "inconsistent",
    ),
    Case(
        "fabricated growth claim (real is a decline, claim says +12%)",
        "known_bad",
        Claim(ticker=TICKER, metric="Revenues", comparison_type="growth_pct",
              claimed_value=12.0, period_end=FY23_END, period_start=FY23_START,
              comparison_period_end=FY22_END, comparison_period_start=FY22_START),
        "inconsistent",
    ),

    # ---------- adversarial: constructed to probe specific exploitable seams ----------
    Case(
        "sign flip on growth_pct (right magnitude, wrong direction)",
        "adversarial",
        Claim(ticker=TICKER, metric="Revenues", comparison_type="growth_pct",
              claimed_value=2.8, period_end=FY23_END, period_start=FY23_START,  # real is -2.8
              comparison_period_end=FY22_END, comparison_period_start=FY22_START),
        "inconsistent",
        "Sign-flipped claims must not slip through as 'close enough' just because |value| matches.",
    ),
    Case(
        "sign flip on bps_change (right magnitude, wrong direction)",
        "adversarial",
        Claim(ticker=TICKER, metric="GrossProfit", denominator_metric="Revenues", comparison_type="bps_change",
              claimed_value=-82.0, period_end=FY23_END, period_start=FY23_START,  # real is +82
              comparison_period_end=FY22_END, comparison_period_start=FY22_START),
        "inconsistent",
        "Same probe as above, for the ratio-based comparison type.",
    ),
    Case(
        "sign flip on absolute_change",
        "adversarial",
        Claim(ticker=TICKER, metric="Revenues", comparison_type="absolute_change",
              claimed_value=11_043_000_000, period_end=FY23_END, period_start=FY23_START,  # real is negative
              comparison_period_end=FY22_END, comparison_period_start=FY22_START),
        "inconsistent",
        "A claimed *increase* when the real figure *decreased* by the same magnitude.",
    ),
    Case(
        "magnitude confusion: claim stated in millions, facts in raw dollars",
        "adversarial",
        Claim(ticker=TICKER, metric="Revenues", comparison_type="absolute",
              claimed_value=383_285, period_end=FY23_END, period_start=FY23_START),  # off by 1,000,000x
        "inconsistent",
        "A unit-confused claim must not be lucky enough to fall inside a wide tolerance by coincidence.",
    ),
    Case(
        "comparison_type mislabeled: a growth percentage fed in as an absolute dollar claim",
        "adversarial",
        Claim(ticker=TICKER, metric="Revenues", comparison_type="absolute",
              claimed_value=-2.8, period_end=FY23_END, period_start=FY23_START),
        "inconsistent",
        "Type confusion (percent value used where a dollar level is expected) shouldn't accidentally "
        "read as 'consistent' just because the arithmetic still runs.",
    ),
    Case(
        "tolerance boundary: exactly at the 1% relative-tolerance edge for absolute claims",
        "adversarial",
        Claim(ticker=TICKER, metric="Revenues", comparison_type="absolute",
              claimed_value=383_285_000_000 * 1.01, period_end=FY23_END, period_start=FY23_START),
        "consistent",
        "Documents the boundary is inclusive (<=) by construction, not an accident — exactly 1.00% off.",
    ),
    Case(
        "tolerance boundary: just past the 1% edge",
        "adversarial",
        Claim(ticker=TICKER, metric="Revenues", comparison_type="absolute",
              claimed_value=383_285_000_000 * 1.0101, period_end=FY23_END, period_start=FY23_START),
        "inconsistent",
        "One basis point past the boundary must flip the verdict — confirms the boundary isn't fuzzy.",
    ),
    Case(
        "pure-unit percentage claim (effective tax rate), correct",
        "adversarial",
        Claim(ticker=TICKER, metric="EffectiveIncomeTaxRateContinuingOperations", comparison_type="absolute",
              claimed_value=0.147, period_end=FY23_END, period_start=FY23_START, unit="pure"),
        "consistent",
        "Regression probe for the pure-fraction-vs-percent unit bug found in Phase 5 "
        "(agents/verification_agent.py does the 19.0-vs-0.19 conversion; the raw reconciler "
        "must still correctly compare like-for-like when given the fraction directly).",
    ),

    # ---------- bitemporal: without as_of, a later restatement produces a false-inconsistent ----------
    Case(
        "restatement without as_of: correct-at-the-time claim reads as false inconsistent",
        "adversarial",
        Claim(ticker=TICKER, metric="Revenues", comparison_type="absolute",
              claimed_value=383_285_000_000, period_end=FY23_END, period_start=FY23_START),
        "inconsistent",  # expected to FAIL here on purpose — see note
        "This is the one case in this audit that is EXPECTED to disagree with 'known-good' intent: "
        "without an as_of cutoff, the most-recent-restatement rule intentionally overrides the "
        "original figure. It's included to make that tradeoff visible in the report, not hidden — "
        "the fix (agents/verification_agent.py passing as_of through) is what makes this safe in "
        "the real pipeline, and is covered separately by tests/test_reconciler.py's as_of tests.",
    ),
]

RESTATEMENT_PROBE_FACTS = FACTS + [
    _fact("Revenues", 390_000_000_000, FY23_START, FY23_END, "0000320193-24-999999", filed="2024-03-01"),
]

# The restatement-demo case above is deliberately excluded from the recall/false-
# positive estimate below: its expected_verdict documents the pre-as_of *bug*
# behavior, not the real pipeline's behavior (which always passes as_of), so
# treating it as a genuine "should be X" case would corrupt the estimate.
_RESTATEMENT_DEMO_LABEL = "restatement without as_of: correct-at-the-time claim reads as false inconsistent"

# DSPy-optimized claim-extraction precision, from eval/run_comparison.py's live run
# (see README's Results section) — the "p" (generator correctness) term in the
# precision-ceiling formula below. Not re-derived here since that requires a live
# LLM call; hardcoded with its source cited so it's checkable, not silently assumed.
EXTRACTION_PRECISION_DSPY_OPTIMIZED = 0.763


def reconciler_recall_and_false_positive_rate(facts: list[FinancialFact] = FACTS) -> dict:
    """The Reconciler's own recall (r) and false-positive rate (f), estimated from
    this audit's cases: r = fraction of genuinely-consistent claims correctly
    accepted; f = fraction of genuinely-not-consistent claims wrongly accepted.
    Small n (6 and 8 cases respectively) — illustrative, not a statistically tight
    estimate. When f is observed as exactly 0, the "rule of three" gives a rough
    95% upper bound of ~3/n on the true rate a zero count doesn't rule out."""
    cases = [c for c in CASES if c.label != _RESTATEMENT_DEMO_LABEL]
    should_be_consistent = [c for c in cases if c.expected_verdict == "consistent"]
    should_not_be_consistent = [c for c in cases if c.expected_verdict != "consistent"]

    true_positives = sum(1 for c in should_be_consistent if reconcile(c.claim, facts).verdict == "consistent")
    false_positives = sum(1 for c in should_not_be_consistent if reconcile(c.claim, facts).verdict == "consistent")

    n_good, n_bad = len(should_be_consistent), len(should_not_be_consistent)
    recall = true_positives / n_good if n_good else float("nan")
    false_positive_rate = false_positives / n_bad if n_bad else float("nan")

    return {
        "recall": recall,
        "n_should_be_consistent": n_good,
        "true_positives": true_positives,
        "false_positive_rate": false_positive_rate,
        "n_should_not_be_consistent": n_bad,
        "false_positives": false_positives,
        "false_positive_rate_upper_bound_95": (3.0 / n_bad) if false_positives == 0 and n_bad else None,
    }


def precision_ceiling(p: float, r: float, f: float) -> float:
    """Pr(correct | accepted) = p*r / (p*r + (1-p)*f) — the fraction of claims the
    verifier accepts as "consistent" that are actually correct, given generator
    (extraction) correctness p, verifier recall r, and verifier false-positive
    rate f. The point of this formula (from the course material this audit was
    built against): verification precision, not generation quality, sets the
    ceiling on what a maker/checker loop can deliver end-to-end — a high-recall
    verifier still ships garbage if f is non-trivial, no matter how good p is."""
    numerator = p * r
    denominator = numerator + (1 - p) * f
    return numerator / denominator if denominator else float("nan")


def run_audit() -> dict:
    results = []
    for case in CASES:
        facts = RESTATEMENT_PROBE_FACTS if "restatement" in case.label else FACTS
        result = reconcile(case.claim, facts)
        results.append((case, result))

    total = len(results)
    correct = sum(1 for c, r in results if r.verdict == c.expected_verdict)

    false_consistent = [
        (c, r) for c, r in results if r.verdict == "consistent" and c.expected_verdict != "consistent"
    ]

    print(f"{'CASE':<70} {'CATEGORY':<12} {'EXPECTED':<14} {'GOT':<14} {'OK'}")
    print("-" * 130)
    for case, result in results:
        ok = "✓" if result.verdict == case.expected_verdict else "✗"
        print(f"{case.label:<70} {case.category:<12} {case.expected_verdict:<14} {result.verdict:<14} {ok}")
        if case.note:
            print(f"    note: {case.note}")

    print()
    print(f"{correct}/{total} cases matched expected verdict")
    print(f"False-CONSISTENT count (the dangerous failure mode): {len(false_consistent)}")
    for c, r in false_consistent:
        print(f"  - {c.label}: {r.explanation}")

    stats = reconciler_recall_and_false_positive_rate()
    p = EXTRACTION_PRECISION_DSPY_OPTIMIZED
    ceiling = precision_ceiling(p, stats["recall"], stats["false_positive_rate"])
    print()
    print(f"Reconciler recall (r): {stats['recall']:.3f} ({stats['true_positives']}/{stats['n_should_be_consistent']})")
    print(
        f"Reconciler false-positive rate (f): {stats['false_positive_rate']:.3f} "
        f"({stats['false_positives']}/{stats['n_should_not_be_consistent']})"
        + (
            f" — 0 observed, ~{stats['false_positive_rate_upper_bound_95']:.3f} 95% upper bound (rule of three) at this n"
            if stats["false_positive_rate_upper_bound_95"] is not None
            else ""
        )
    )
    print(
        f"System precision ceiling Pr(correct|accepted), given extraction precision p={p}: {ceiling:.3f} "
        f"(verification precision, not extraction quality, is what actually bounds this)"
    )

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "false_consistent_count": len(false_consistent),
        "false_consistent_cases": [c.label for c, r in false_consistent],
        "recall": stats["recall"],
        "false_positive_rate": stats["false_positive_rate"],
        "false_positive_rate_upper_bound_95": stats["false_positive_rate_upper_bound_95"],
        "precision_ceiling": ceiling,
    }


if __name__ == "__main__":
    run_audit()
