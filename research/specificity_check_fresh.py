"""Does the specificity result (0/33 false positives, docs/specificity-check-
results.md) generalize, or is it an artifact of fixing exactly what turned up
in the original 8-ticker control set (ADBE, CRM, ORCL, CSCO, INTU, IBM, QCOM,
TXN)? Every one of this project's period-resolution, chunking, and extraction
fixes was diagnosed against real bugs found IN those specific companies' real
filings — CRM's own fiscal-year tagging quirk, CSCO's specific phrasing
patterns, TXN's specific sentence structures. A 0% rate measured only against
the companies the fixes were built to fix proves the fixes work, not that the
underlying approach generalizes.

Runs the exact same methodology (research/specificity_check.py's
Coordinator/Budget/confirm_clean, reused directly, not reimplemented) against
a second, disjoint set of large tech/software companies — none touched by any
fix, test, or diagnostic call this project has made. Also disjoint from
AAPL/MSFT/NVDA/GOOGL/AMZN (the pipeline's own original dev/tuning companies).

Run: python -m research.specificity_check_fresh
"""

import json
from dataclasses import asdict
from pathlib import Path

from agents.coordinator import Coordinator
from agents.extraction_agent import RealExtractionAgent
from agents.retrieval_agent import RealRetrievalAgent
from agents.verification_agent import RealVerificationAgent
from research.specificity_check import aggregate, confirm_clean, run_one
from tools.edgar_client import EdgarClient

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "research" / "data" / "specificity_check_fresh.json"

# Large tech/software names, structurally the same class as the original
# control set - genuinely untouched by this project's development at any
# point. Confirmed clean (no restatement fingerprint) the same way as before,
# not assumed.
CANDIDATE_TICKERS = ["NOW", "PANW", "CRWD", "WDAY", "SNPS", "FTNT", "INTC", "MU"]


def main() -> None:
    client = EdgarClient()
    print("Confirming the fresh control set carries no restatement fingerprint...", flush=True)
    tickers = confirm_clean(CANDIDATE_TICKERS, client)
    print(f"Fresh control set: {tickers}\n", flush=True)

    retrieval = RealRetrievalAgent()
    extraction = RealExtractionAgent()
    verification = RealVerificationAgent()
    coordinator = Coordinator(retrieval, extraction, verification)

    runs = []
    for ticker in tickers:
        print(f"--- {ticker} ---", flush=True)
        run = run_one(coordinator, ticker)
        runs.append(run)
        if run.ok:
            print(f"  ok: {run.summary['n_claims']} claims, {run.summary['by_verdict']}, "
                  f"{run.elapsed_seconds:.1f}s"
                  + (" [PARTIAL - budget hit]" if run.summary.get("partial") else ""), flush=True)
            for c in run.inconsistent_claims:
                print(f"  INCONSISTENT: {c['metric']}={c['claimed_value']:,} @ {c['period']!r} "
                      f"-> {c['explanation']}", flush=True)
        else:
            print(f"  FAILED: {run.error.splitlines()[0]}", flush=True)

    result = aggregate(runs)
    print("\n=== Fresh specificity check aggregate ===", flush=True)
    print(json.dumps(result, indent=2), flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "control_tickers": tickers,
        "aggregate": result,
        "runs": [asdict(r) for r in runs],
    }, indent=2))
    print(f"\nSaved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
