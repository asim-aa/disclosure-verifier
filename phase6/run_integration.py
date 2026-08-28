"""Phase 6: end-to-end integration at scale.

Runs the real Coordinator (no mocks — real EDGAR retrieval, real LLM extraction,
real Reconciler verification) against multiple companies' 10-Ks, not just the one
hand-picked NVDA example in the README. Aggregates real numbers across the batch:
verdict distribution, tool-call counts, wall-clock, and per-ticker failures.

Requires network access to SEC EDGAR and to the LLM backend (LLM_BASE_URL in
.env) — not part of the regular offline test suite.

Run: python -m phase6.run_integration
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

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "phase6" / "results"

# AAPL/MSFT/NVDA are the companies this project was built and tested against
# throughout; GOOGL/AMZN are new — included specifically to check whether the
# pipeline (concept resolution in particular, which is a fixed lookup table)
# generalizes, or was quietly tuned to the three it had already seen.
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

# Bounds cost/time per ticker — a real 10-K can have 100+ MD&A chunks, each an
# LLM call against a reasoning model. Without this, one large filing could
# dominate the whole batch's wall-clock.
PER_TICKER_BUDGET = Budget(max_chunks=25, max_extraction_calls=20, max_seconds=240)


@dataclass
class TickerRun:
    ticker: str
    ok: bool
    error: str | None
    summary: dict | None
    elapsed_seconds: float


def run_one(coordinator: Coordinator, ticker: str) -> TickerRun:
    start = time.monotonic()
    try:
        report = coordinator.run(ticker, budget=PER_TICKER_BUDGET, resume=False)
        return TickerRun(
            ticker=ticker, ok=True, error=None,
            summary=report.summary(), elapsed_seconds=time.monotonic() - start,
        )
    except Exception as exc:  # noqa: BLE001 - one ticker's failure shouldn't kill the batch
        return TickerRun(
            ticker=ticker, ok=False, error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            summary=None, elapsed_seconds=time.monotonic() - start,
        )


def aggregate(runs: list[TickerRun]) -> dict:
    ok_runs = [r for r in runs if r.ok]
    by_verdict: dict[str, int] = {}
    total_claims = total_tool_calls = total_reasoning_steps = 0
    for r in ok_runs:
        s = r.summary
        total_claims += s["n_claims"]
        total_tool_calls += s["n_tool_calls"]
        total_reasoning_steps += s["n_reasoning_steps"]
        for verdict, count in s["by_verdict"].items():
            by_verdict[verdict] = by_verdict.get(verdict, 0) + count

    return {
        "n_tickers": len(runs),
        "n_succeeded": len(ok_runs),
        "n_failed": len(runs) - len(ok_runs),
        "failed_tickers": [r.ticker for r in runs if not r.ok],
        "total_claims_verified": total_claims,
        "by_verdict": by_verdict,
        "total_tool_calls": total_tool_calls,
        "total_reasoning_steps": total_reasoning_steps,
        "total_elapsed_seconds": sum(r.elapsed_seconds for r in runs),
        "partial_runs": [r.ticker for r in ok_runs if r.summary.get("partial")],
    }


def main() -> None:
    retrieval = RealRetrievalAgent()
    extraction = RealExtractionAgent()  # configures the DSPy LM once, reused across tickers
    verification = RealVerificationAgent()
    coordinator = Coordinator(retrieval, extraction, verification)

    runs: list[TickerRun] = []
    for ticker in TICKERS:
        print(f"--- {ticker} ---")
        run = run_one(coordinator, ticker)
        runs.append(run)
        if run.ok:
            print(f"  ok: {run.summary['n_claims']} claims, "
                  f"{run.summary['by_verdict']}, {run.elapsed_seconds:.1f}s"
                  + (" [PARTIAL - budget hit]" if run.summary.get("partial") else ""))
        else:
            print(f"  FAILED: {run.error.splitlines()[0]}")

    result = aggregate(runs)
    print("\n=== Phase 6 aggregate ===")
    print(json.dumps(result, indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "run.json"
    out_path.write_text(json.dumps({
        "aggregate": result,
        "runs": [asdict(r) for r in runs],
    }, indent=2))
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
