"""Specificity check: the backtest (docs/backtest-results.md) only tests
sensitivity -- it shows the reconciler flags claims that turned out to be
wrong, on companies that actually restated. It's never been checked the other
way: does it stay quiet on companies that *didn't* restate, or does it cry
wolf on clean, accurate filings?

Runs the real Coordinator (no mocks -- real EDGAR retrieval, real LLM
extraction, real Reconciler verification, same machinery as phase6/run_integration.py)
against a control set of large, well-known companies confirmed -- not assumed
-- to carry no restatement fingerprint anywhere in Phase A's full scan
(research/data/restatement_fingerprints.json, 13,827 fingerprints across 541
companies). Deliberately a different set from the tickers used everywhere else
in this project's development (AAPL/MSFT/NVDA/GOOGL/AMZN) to avoid measuring
specificity on the same companies the pipeline was built and tuned against.

The honest framing: "no restatement fingerprint in this project's own Phase A
sweep" is not the same claim as "this company has literally never restated
anything, ever" -- Phase A's candidate list itself comes from a 3-year window
of 8-K Item 4.02 search results, not an exhaustive registry. It's the same
standard of evidence this project already applies everywhere else: checked
directly against real data, stated as what it actually is.

What counts as a false positive here isn't "the claim is factually true" (it
almost certainly is -- these are large companies' own audited disclosures,
never flagged for a restatement) -- it's "the reconciler's own concept
resolution, period resolution, and tolerance arithmetic produced an
'inconsistent' verdict it shouldn't have." Every inconsistent verdict from
this run is printed for manual review, not silently folded into a rate.

Run: python -m research.specificity_check
"""

import json
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

from agents.coordinator import Coordinator
from agents.extraction_agent import RealExtractionAgent
from agents.retrieval_agent import RealRetrievalAgent
from agents.schema import Budget
from agents.verification_agent import RealVerificationAgent
from tools.edgar_client import EdgarClient

REPO_ROOT = Path(__file__).resolve().parent.parent
FINGERPRINTS_PATH = REPO_ROOT / "research" / "data" / "restatement_fingerprints.json"
OUT_PATH = REPO_ROOT / "research" / "data" / "specificity_check.json"

# Large, well-known companies never used elsewhere in this project's development
# (AAPL/MSFT/NVDA/GOOGL/AMZN are the ones this pipeline was built and tuned
# against) -- confirmed below, not assumed, to carry no restatement fingerprint.
#
# A first pass at this used financials/consumer-staples names (JPM, JNJ, V,
# COST, HD, PG, KO, MA) and got zero resolved verdicts: 5 of 8 tickers had no
# MD&A the parser could find at all (a real coverage gap for those filer
# types, distinct from what's being measured here), and the other 3 spent
# their whole extraction budget on segment/product detail before reaching a
# claim the resolver covers -- these companies' MD&As lead with pharma
# segments or payment volume, not top-line GAAP figures. Switched to large
# tech/software names, structurally closer to AAPL/MSFT/NVDA (the companies
# this pipeline's MD&A parsing and concept dictionary were actually built and
# tested against), and raised the budget to get past any segment-heavy
# opening. Reported as what it is: a genuine methodological correction, not
# silently swapped without explanation.
CANDIDATE_TICKERS = ["ADBE", "CRM", "ORCL", "CSCO", "INTU", "IBM", "QCOM", "TXN"]

PER_TICKER_BUDGET = Budget(max_chunks=40, max_extraction_calls=40, max_seconds=300)


@dataclass
class TickerRun:
    ticker: str
    ok: bool
    error: str | None
    summary: dict | None
    inconsistent_claims: list[dict]
    elapsed_seconds: float


def confirm_clean(tickers: list[str], client: EdgarClient) -> list[str]:
    """Drop any ticker that DOES carry a restatement fingerprint - checked
    directly against Phase A's real scan output, not assumed clean because it's
    a well-known company."""
    fingerprints = json.loads(FINGERPRINTS_PATH.read_text())
    flagged_ciks = {f["cik"] for f in fingerprints}
    clean = []
    for t in tickers:
        cik = client.resolve_cik(t)
        if cik in flagged_ciks:
            print(f"  {t} ({cik}) carries a restatement fingerprint - excluding from the control set", flush=True)
            continue
        clean.append(t)
    return clean


def run_one(coordinator: Coordinator, ticker: str) -> TickerRun:
    start = time.monotonic()
    try:
        report = coordinator.run(ticker, budget=PER_TICKER_BUDGET, resume=False)
        inconsistent = [
            {
                "metric": c.extracted.metric, "claimed_value": c.extracted.value,
                "period": c.extracted.period, "quote": c.extracted.quote,
                "explanation": c.explanation,
                "reason_code": c.reconciliation.reason_code if c.reconciliation else None,
            }
            for c in report.verified_claims if c.verdict == "inconsistent"
        ]
        return TickerRun(
            ticker=ticker, ok=True, error=None,
            summary=report.summary(), inconsistent_claims=inconsistent,
            elapsed_seconds=time.monotonic() - start,
        )
    except Exception as exc:  # noqa: BLE001 - one ticker's failure shouldn't kill the batch
        return TickerRun(
            ticker=ticker, ok=False, error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            summary=None, inconsistent_claims=[], elapsed_seconds=time.monotonic() - start,
        )


def aggregate(runs: list[TickerRun]) -> dict:
    ok_runs = [r for r in runs if r.ok]
    by_verdict: dict[str, int] = {}
    total_claims = 0
    for r in ok_runs:
        s = r.summary
        total_claims += s["n_claims"]
        for verdict, count in s["by_verdict"].items():
            by_verdict[verdict] = by_verdict.get(verdict, 0) + count

    n_resolved = by_verdict.get("consistent", 0) + by_verdict.get("inconsistent", 0)
    n_inconsistent = by_verdict.get("inconsistent", 0)

    return {
        "n_tickers": len(runs),
        "n_succeeded": len(ok_runs),
        "failed_tickers": [r.ticker for r in runs if not r.ok],
        "total_claims_verified": total_claims,
        "by_verdict": by_verdict,
        "n_resolved_verdicts": n_resolved,
        "n_inconsistent": n_inconsistent,
        "apparent_false_positive_rate": (n_inconsistent / n_resolved) if n_resolved else None,
    }


def main() -> None:
    client = EdgarClient()
    print("Confirming the control set carries no restatement fingerprint...", flush=True)
    tickers = confirm_clean(CANDIDATE_TICKERS, client)
    print(f"Control set: {tickers}\n", flush=True)

    retrieval = RealRetrievalAgent()
    extraction = RealExtractionAgent()
    verification = RealVerificationAgent()
    coordinator = Coordinator(retrieval, extraction, verification)

    runs: list[TickerRun] = []
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
    print("\n=== Specificity check aggregate ===", flush=True)
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
