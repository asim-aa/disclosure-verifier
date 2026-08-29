"""Phase C of the disclosure-inconsistency backtest: for each real (prose claim,
restatement fingerprint) pair Phase B found, checks whether this project's actual
bitemporal reconciler would have flagged the claim as consistent when it was made
and inconsistent once the restatement existed — using only real EDGAR data, no
LLM calls (reconcile() is pure arithmetic against already-known claim values).

Deliberately bypasses agents.resolver.resolve_periods() and
agents.verification_agent.RealVerificationAgent.verify() and calls
tools.reconciler.reconcile() directly. resolve_periods() picks the "most recent
reported period" by sorting facts by period_end descending — correct for a live
claim about "the latest quarter," but wrong here: by the time a restatement
happens, the company has always filed later fiscal years, so "most recent
period" would silently jump to a fiscal year the claim was never about, defeating
the point of checking a *specific* historical period. Each fingerprint already
carries its own period_start/period_end (from Phase A's XBRL scan) and its own
claimed_value (from Phase B's prose match), so a Claim is built directly from
data already known to be correct rather than re-derived.

The `as_of` bitemporal cutoff (see tools/reconciler.py's _find_fact) does the
real work: as_of=original_filed restricts fact lookups to data filed by that
date (the restated value doesn't exist yet, so only the original is a
candidate); as_of=restated_filed (or later) admits both, and _find_fact prefers
the most-recently-filed value when they differ - the restated one.

Two real per-match outcomes both count as informative, not just a clean
consistent->inconsistent flip:
  - A claim already inconsistent even as of the ORIGINAL filing (the prose
    rounded the figure enough, e.g. "$1.2 billion" for $1.242 billion, to fall
    outside the reconciler's own tolerance) - the reconciler would have flagged
    it immediately, restatement or not.
  - A claim that stays consistent even as of the RESTATED filing (the rounded
    prose value happens to sit within tolerance of both the original AND the
    restated figure) - the restatement was real but too small, relative to how
    the prose rounded, for this reconciler's tolerance to catch.
Only the fingerprint/claim data + reconcile()'s own tolerances determine which
bucket a match falls into - nothing here is tuned to produce a particular story.

Run: python -m research.run_backtest
"""

import json
from pathlib import Path

from tools.edgar_client import EdgarClient
from tools.reconciler import reconcile
from tools.schema import Claim
from tools.xbrl_parser import parse_company_facts

REPO_ROOT = Path(__file__).resolve().parent.parent
MATCHES_PATH = REPO_ROOT / "research" / "data" / "phase_b_matches.json"
OUT_PATH = REPO_ROOT / "research" / "data" / "backtest_results.json"


def build_claim(match: dict) -> Claim:
    fp = match["fingerprint"]
    return Claim(
        ticker=match["ticker"],
        metric=fp["concept"],  # reconcile() matches Claim.metric directly against
        # FinancialFact.concept (the raw XBRL tag, e.g. "NetIncomeLoss") - not the
        # human-language metric text ("net income") extraction produced, which is
        # only used upstream for Phase B's matching, not by the reconciler itself.
        comparison_type=match["claim"]["comparison_type"],
        claimed_value=match["claim"]["value"],
        period_end=fp["period_end"],
        period_start=fp["period_start"],
        unit=fp["unit"],
    )


def classify(before: str, after: str) -> str:
    if before == "consistent" and after == "inconsistent":
        return "expected_flip"
    if before == "inconsistent":
        return "already_inconsistent_at_original"
    if before == "consistent" and after == "consistent":
        return "restatement_too_small_for_tolerance"
    return f"other({before}->{after})"


def main() -> None:
    matches = json.loads(MATCHES_PATH.read_text())
    print(f"Loaded {len(matches)} Phase B matches", flush=True)

    client = EdgarClient()
    facts_by_cik: dict[str, list] = {}

    results = []
    for i, match in enumerate(matches):
        cik = match["fingerprint"]["cik"]
        if cik not in facts_by_cik:
            raw = client.get_company_facts(cik)
            facts_by_cik[cik] = parse_company_facts(raw, ticker=match["ticker"], concepts=None)
            print(f"  fetched {len(facts_by_cik[cik])} facts for CIK {cik}", flush=True)
        facts = facts_by_cik[cik]

        claim = build_claim(match)
        fp = match["fingerprint"]

        before = reconcile(claim, facts, as_of=fp["original_filed"])
        after = reconcile(claim, facts, as_of=fp["restated_filed"])

        outcome = classify(before.verdict, after.verdict)
        print(
            f"[{i + 1}/{len(matches)}] {fp['entity_name'].strip()} {fp['concept']} "
            f"{fp['period_end']} claimed={claim.claimed_value:,.0f} -> "
            f"as_of_original={before.verdict} ({before.difference and f'{before.difference:.2%}'}) "
            f"as_of_restated={after.verdict} ({after.difference and f'{after.difference:.2%}'}) "
            f"[{outcome}]",
            flush=True,
        )

        results.append({
            "entity_name": fp["entity_name"].strip(),
            "concept": fp["concept"],
            "period_end": fp["period_end"],
            "claimed_value": claim.claimed_value,
            "original_value": fp["original_value"],
            "restated_value": fp["restated_value"],
            "original_filed": fp["original_filed"],
            "restated_filed": fp["restated_filed"],
            "as_of_original": {
                "verdict": before.verdict,
                "computed_value": before.computed_value,
                "difference": before.difference,
                "reason_code": before.reason_code,
            },
            "as_of_restated": {
                "verdict": after.verdict,
                "computed_value": after.computed_value,
                "difference": after.difference,
                "reason_code": after.reason_code,
            },
            "outcome": outcome,
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    from collections import Counter
    tally = Counter(r["outcome"] for r in results)
    print(f"\n{len(results)} matches backtested. Outcome tally:", flush=True)
    for outcome, count in tally.most_common():
        print(f"  {outcome}: {count}", flush=True)
    print(f"\nSaved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
