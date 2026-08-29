"""Naive-LLM-judge baseline: does the deterministic resolver + reconciler
actually earn its keep, or would a general LLM just prompted with the raw data
get the same answer anyway?

Every other result in this project measures whether *this* pipeline works.
Nothing so far measures whether the *architecture choice* -- structured
extraction, exact concept resolution, then deterministic bitemporal arithmetic,
instead of "hand an LLM the claim and the raw numbers and let it decide" -- was
actually necessary. This is that comparison.

Two test sets, both reused from already-verified ground truth (nothing
hand-labeled fresh for this):

1. eval/reconciler_audit.py's 15 adversarial cases. Deliberately isolates
   arithmetic-reasoning capability from concept-resolution capability: each
   case's Claim.metric is already the exact XBRL concept string, so there's no
   metric-to-concept matching for the naive baseline to get right or wrong --
   only the sign-flip/magnitude-confusion/tolerance-boundary probes the audit
   was built to catch.

2. The 122 real backtest matches (research/data/phase_b_matches.json). Harder
   and more realistic: for each match, the naive baseline is handed the SAME
   already-correctly-resolved concept name reconcile() would use, but the RAW,
   UNFILTERED fact history for that concept -- every filed value across every
   amendment, not the one bitemporally-correct value resolve_periods()/
   _find_fact() would have picked. It's told the claim's own filing date and
   asked to judge consistency "as of" that date, then asked again "as of
   today" -- exactly mirroring the two reconcile() calls run_backtest.py makes,
   but via prompting instead of an as_of cutoff in code. This directly tests
   whether a naive read reproduces the bitemporal design's actual job, or
   defaults to "most recent value" regardless of what it's told -- the exact
   bug this project already found and fixed once in its own deterministic code
   (see docs/robustness-and-scope.md's bitemporal-correctness section).

Deliberately run against this project's own LLM endpoint (the same model used
everywhere else here), not a stronger commercial model -- isolates "does
structure help, holding model capability constant" as the cleanest version of
the comparison. A stronger-model pass is a natural next step, not run here.

Run: python -m research.naive_llm_baseline
"""

import json
from pathlib import Path

import dspy

from eval.llm_config import configure_dspy
from eval.reconciler_audit import CASES, FACTS, RESTATEMENT_PROBE_FACTS, _RESTATEMENT_DEMO_LABEL
from tools.edgar_client import EdgarClient
from tools.schema import FinancialFact
from tools.xbrl_parser import parse_company_facts

REPO_ROOT = Path(__file__).resolve().parent.parent
MATCHES_PATH = REPO_ROOT / "research" / "data" / "phase_b_matches.json"
OUT_PATH = REPO_ROOT / "research" / "data" / "naive_baseline_results.json"

VERDICTS = {"consistent", "inconsistent", "unverifiable"}


class NaiveVerify(dspy.Signature):
    """You are given a quantitative claim from a company's SEC filing and the raw,
    unfiltered reported data for that exact concept -- every value the company has
    ever filed for it, each tagged with the date it was filed. Some of these may be
    duplicates, revisions, or restatements of each other. Decide whether the claim
    is CONSISTENT with what the company reported, using only data filed on or
    before the given as-of date -- a later value filed after that date does not
    count, even if it's the most recent one shown. Reason step by step, then give
    your verdict."""

    claim_text: str = dspy.InputField(desc="the concept, claimed value, and period the claim is about")
    as_of_date: str = dspy.InputField(desc="only use facts filed on or before this date")
    raw_facts: str = dspy.InputField(desc="every raw reported value for this concept, each with its own filed date")
    reasoning: str = dspy.OutputField()
    verdict: str = dspy.OutputField(desc="exactly one of: consistent, inconsistent, unverifiable")


def _fact_line(f: FinancialFact) -> str:
    return (
        f"- value={f.value:,} {f.unit}, period {f.period_start}..{f.period_end}, "
        f"filed={f.filed}, form={f.form}, accession={f.accession_number}"
    )


def judge(module: dspy.Predict, claim_text: str, as_of_date: str, facts: list[FinancialFact]) -> dict:
    raw_facts = "\n".join(_fact_line(f) for f in facts) if facts else "(no facts found for this concept)"
    result = module(claim_text=claim_text, as_of_date=as_of_date, raw_facts=raw_facts)
    verdict = result.verdict.strip().lower()
    if verdict not in VERDICTS:
        # the model didn't return a clean label -- honest miss, not a crash
        verdict = f"unparseable:{result.verdict.strip()[:60]}"
    return {"verdict": verdict, "reasoning": result.reasoning}


def run_audit_set(module: dspy.Predict) -> list[dict]:
    print(f"\n=== Test set 1: reconciler_audit.py's {len(CASES)} adversarial cases ===", flush=True)
    results = []
    for case in CASES:
        facts = RESTATEMENT_PROBE_FACTS if case.label == _RESTATEMENT_DEMO_LABEL else FACTS
        # the real pipeline always passes as_of; judge every case as of the
        # latest fact's own filed date, i.e. "no later restatement exists yet",
        # matching every case's intent except the deliberately-excluded demo one
        as_of = max(f.filed for f in facts)
        claim_text = (
            f"metric={case.claim.metric}, comparison_type={case.claim.comparison_type}, "
            f"claimed_value={case.claim.claimed_value}, period_end={case.claim.period_end}, "
            f"period_start={case.claim.period_start}, "
            f"comparison_period_end={case.claim.comparison_period_end}, "
            f"comparison_period_start={case.claim.comparison_period_start}, "
            f"denominator_metric={case.claim.denominator_metric}"
        )
        relevant_concepts = {case.claim.metric, case.claim.denominator_metric} - {None}
        judged = judge(module, claim_text, as_of, [f for f in facts if f.concept in relevant_concepts])
        correct = judged["verdict"] == case.expected_verdict
        print(
            f"  [{'OK' if correct else 'MISS'}] {case.label!r}: expected={case.expected_verdict} "
            f"got={judged['verdict']}",
            flush=True,
        )
        results.append({
            "test_set": "reconciler_audit",
            "label": case.label,
            "category": case.category,
            "expected_verdict": case.expected_verdict,
            "naive_verdict": judged["verdict"],
            "correct": correct,
            "reasoning": judged["reasoning"],
        })
    return results


def run_backtest_set(module: dspy.Predict) -> list[dict]:
    matches = json.loads(MATCHES_PATH.read_text())
    print(f"\n=== Test set 2: {len(matches)} real backtest matches, two as_of checks each ===", flush=True)

    client = EdgarClient()
    facts_by_cik: dict[str, list[FinancialFact]] = {}
    results = []

    for i, match in enumerate(matches):
        fp = match["fingerprint"]
        cik = fp["cik"]
        if cik not in facts_by_cik:
            raw = client.get_company_facts(cik)
            facts_by_cik[cik] = parse_company_facts(raw, ticker=match["ticker"], concepts=None)
        facts = [
            f for f in facts_by_cik[cik]
            if f.concept == fp["concept"] and f.period_start == fp["period_start"] and f.period_end == fp["period_end"]
        ]
        claim_text = (
            f"metric (already resolved to XBRL concept)={fp['concept']}, "
            f"claimed_value={match['claim']['value']}, "
            f"period {fp['period_start']}..{fp['period_end']}"
        )

        before = judge(module, claim_text, fp["original_filed"], facts)
        after = judge(module, claim_text, fp["restated_filed"], facts)
        # expected: consistent as of the original filing, inconsistent as of the
        # restated filing -- the exact pattern run_backtest.py's real reconcile()
        # calls established for the 118/122 that flipped or were already caught
        correctly_flips = before["verdict"] == "consistent" and after["verdict"] == "inconsistent"

        print(
            f"  [{i + 1}/{len(matches)}] {fp['entity_name'].strip()} {fp['concept']} {fp['period_end']} "
            f"-> as_of_original={before['verdict']} as_of_restated={after['verdict']} "
            f"[{'flips correctly' if correctly_flips else 'does not flip'}]",
            flush=True,
        )
        results.append({
            "test_set": "backtest",
            "entity_name": fp["entity_name"].strip(),
            "concept": fp["concept"],
            "period_end": fp["period_end"],
            "as_of_original_verdict": before["verdict"],
            "as_of_restated_verdict": after["verdict"],
            "correctly_flips": correctly_flips,
            "reasoning_original": before["reasoning"],
            "reasoning_restated": after["reasoning"],
        })
    return results


def main() -> None:
    configure_dspy()
    module = dspy.Predict(NaiveVerify)

    audit_results = run_audit_set(module)
    backtest_results = run_backtest_set(module)
    all_results = audit_results + backtest_results

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(all_results, indent=2))

    n_audit_correct = sum(1 for r in audit_results if r["correct"])
    n_backtest_flip = sum(1 for r in backtest_results if r["correctly_flips"])
    print(f"\nAudit set: {n_audit_correct}/{len(audit_results)} correct verdicts", flush=True)
    print(f"Backtest set: {n_backtest_flip}/{len(backtest_results)} correctly flipped consistent->inconsistent "
          f"across the as_of boundary", flush=True)
    print(f"Saved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
