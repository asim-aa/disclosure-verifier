"""MCP server: Numerical Reconciler.

Given a claim (metric, value, period, comparison type), fetches the company's live
XBRL facts and determines whether the claim is arithmetically consistent,
inconsistent, or unverifiable. This is Pillar 1, tool 3 of 3 — and its core logic
(tools/reconciler.py) is reused directly as the RLVR reward function in Phase 7.

Run directly for a stdio smoke test:
    python -m tools.numerical_reconciler
"""

from mcp.server.fastmcp import FastMCP

from tools.edgar_client import EdgarClient
from tools.reconciler import reconcile
from tools.schema import Claim
from tools.xbrl_parser import parse_company_facts

mcp = FastMCP("numerical-reconciler")

_client: EdgarClient | None = None


def get_client() -> EdgarClient:
    global _client
    if _client is None:
        _client = EdgarClient()
    return _client


@mcp.tool()
def reconcile_claim(
    ticker: str,
    metric: str,
    comparison_type: str,
    claimed_value: float,
    period_end: str,
    period_start: str | None = None,
    comparison_period_end: str | None = None,
    comparison_period_start: str | None = None,
    denominator_metric: str | None = None,
    unit: str = "USD",
    tolerance: float | None = None,
) -> dict:
    """Check a quantitative claim against a company's real XBRL data.

    comparison_type is one of:
      - "absolute": `metric` at `period_end` should equal `claimed_value`.
      - "growth_pct": percent change in `metric` from comparison_period_end to
        period_end should equal `claimed_value` (e.g. "revenue grew 12% YoY").
      - "bps_change": change in `metric / denominator_metric`, in basis points,
        from comparison_period_end to period_end (e.g. "margin expanded 200 bps").

    Returns a verdict ("consistent" | "inconsistent" | "unverifiable"), the value
    computed from EDGAR, and citations (accession numbers) of the facts used.
    """
    claim = Claim(
        ticker=ticker,
        metric=metric,
        comparison_type=comparison_type,
        claimed_value=claimed_value,
        period_end=period_end,
        period_start=period_start,
        comparison_period_end=comparison_period_end,
        comparison_period_start=comparison_period_start,
        denominator_metric=denominator_metric,
        unit=unit,
        tolerance=tolerance,
    )

    client = get_client()
    cik = client.resolve_cik(ticker)
    raw_facts = client.get_company_facts(cik)

    concepts = [metric] + ([denominator_metric] if denominator_metric else [])
    facts = parse_company_facts(raw_facts, ticker, concepts=concepts)

    result = reconcile(claim, facts)
    return result.to_dict()


if __name__ == "__main__":
    mcp.run()
