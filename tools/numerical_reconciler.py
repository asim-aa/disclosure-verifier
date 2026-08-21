"""MCP server: Numerical Reconciler.

Given a claim (metric, value, period, comparison type), fetches the company's live
XBRL facts and determines whether the claim is arithmetically consistent,
inconsistent, or unverifiable. This is Pillar 1, tool 3 of 3 — and its core logic
(tools/reconciler.py) is reused directly as the RLVR reward function in Phase 7.

Cost/latency: one XBRL facts fetch per call (disk-cached per ticker — see
tools/edgar_client.py), plus pure in-memory computation; cheap and fast on a
cache hit. Never raises on missing/ambiguous data — see verdict="unverifiable"
below, since a reward function must always produce a signal rather than crash.
Audited against a known-good/known-bad/adversarial case set before being
trusted as ground truth for anything — see eval/reconciler_audit.py.

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
    as_of: str | None = None,
) -> dict:
    """Check a quantitative claim against a company's real XBRL data.

    comparison_type is one of:
      - "absolute": `metric` at `period_end` should equal `claimed_value`.
      - "growth_pct": percent change in `metric` from comparison_period_end to
        period_end should equal `claimed_value` (e.g. "revenue grew 12% YoY").
      - "absolute_change": dollar change in `metric` from comparison_period_end
        to period_end should equal `claimed_value` (e.g. "revenue increased $50.1 billion").
      - "bps_change": change in `metric / denominator_metric`, in basis points,
        from comparison_period_end to period_end (e.g. "margin expanded 200 bps").

    `as_of` (optional): the filing_date of the claim's own source filing. SEC data
    restates — a later filing can revise an earlier period's XBRL value. Without
    `as_of`, this checks the claim against the *latest* known value for that period,
    which can wrongly read a claim as "inconsistent" when it was accurate as of when
    it was made but the world's record has since changed. Pass it whenever the
    claim's source filing date is known; omit only for "is this true as of today"
    queries with no particular filing in mind.

    Returns a verdict ("consistent" | "inconsistent" | "unverifiable"), the value
    computed from EDGAR, a machine-readable `reason_code` for why (e.g.
    "missing_fact", "near_miss", "zero_denominator" — see tools/schema.py), and
    citations (accession numbers) of the facts used.
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

    result = reconcile(claim, facts, as_of=as_of)
    return result.to_dict()


if __name__ == "__main__":
    mcp.run()
