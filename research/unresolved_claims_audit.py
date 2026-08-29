"""What's actually in the "unverifiable" bucket for the tech vertical?

research/specificity_check.py only kept full detail for inconsistent claims -
its own point was the false-positive rate, not coverage. But 113 of the 137
claims it found across 8 real tech companies came back unverifiable, and
that's exactly the kind of bucket that hid real, fixable dictionary gaps
before (see agents/resolver.py's module docstring and docs/robustness-and-
scope.md's coverage section: the fresh-holdout resolution rate went from
13.6% to 44.1% once someone actually looked at which real metric texts were
failing, instead of assuming they were all genuinely segment-specific).

Runs the real Coordinator (no mocks) against the same 8-company tech control
set specificity_check.py uses, capturing metric text + verdict + reason_code
for every claim, not just the inconsistent ones. Tallies unverifiable claims
by metric text so repeated real gaps (worth a dictionary entry) are visible
against one-off genuinely segment-specific claims (correctly out of scope).

Run: python -m research.unresolved_claims_audit
"""

import json
from collections import Counter
from pathlib import Path

from agents.coordinator import Coordinator
from agents.extraction_agent import RealExtractionAgent
from agents.retrieval_agent import RealRetrievalAgent
from agents.schema import Budget
from agents.verification_agent import RealVerificationAgent
from research.specificity_check import CANDIDATE_TICKERS, confirm_clean
from tools.edgar_client import EdgarClient

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "research" / "data" / "unresolved_claims_audit.json"

PER_TICKER_BUDGET = Budget(max_chunks=40, max_extraction_calls=40, max_seconds=300)


def main() -> None:
    client = EdgarClient()
    tickers = confirm_clean(CANDIDATE_TICKERS, client)
    print(f"Control set: {tickers}\n", flush=True)

    retrieval = RealRetrievalAgent()
    extraction = RealExtractionAgent()
    verification = RealVerificationAgent()
    coordinator = Coordinator(retrieval, extraction, verification)

    all_claims = []
    for ticker in tickers:
        print(f"--- {ticker} ---", flush=True)
        report = coordinator.run(ticker, budget=PER_TICKER_BUDGET, resume=False)
        for c in report.verified_claims:
            all_claims.append({
                "ticker": ticker,
                "metric": c.extracted.metric,
                "verdict": c.verdict,
                "reason_code": c.reconciliation.reason_code if c.reconciliation else None,
                "quote": c.extracted.quote,
            })
        print(f"  {len(report.verified_claims)} claims", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(all_claims, indent=2))

    unverifiable = [c for c in all_claims if c["verdict"] == "unverifiable"]
    by_metric = Counter(c["metric"].strip().lower() for c in unverifiable)

    print(f"\n{len(all_claims)} total claims, {len(unverifiable)} unverifiable", flush=True)
    print("\nUnverifiable claims by metric text, most common first:", flush=True)
    for metric, count in by_metric.most_common(40):
        print(f"  {count:3d}  {metric}", flush=True)

    print(f"\nSaved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
