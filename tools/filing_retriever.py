"""MCP server: Filing Retriever.

Pulls 10-K/10-Q/8-K filing metadata and XBRL structured facts from SEC EDGAR
for a given ticker. This is Pillar 1, tool 1 of 3.

Run directly for a stdio smoke test:
    python -m tools.filing_retriever
"""

from typing import Optional

from mcp.server.fastmcp import FastMCP

from tools.edgar_client import EdgarClient
from tools.xbrl_parser import parse_filings, parse_company_facts, list_available_concepts

mcp = FastMCP("filing-retriever")

_client: Optional[EdgarClient] = None


def get_client() -> EdgarClient:
    global _client
    if _client is None:
        _client = EdgarClient()
    return _client


@mcp.tool()
def list_filings(ticker: str, form_types: Optional[list[str]] = None, limit: int = 10) -> list[dict]:
    """List recent filings (10-K/10-Q/8-K by default) for a ticker, most recent first.
    Each result includes the accession number, form type, filing/report dates, and a
    direct URL to the primary document (the citation source)."""
    client = get_client()
    cik = client.resolve_cik(ticker)
    submissions = client.get_submissions(cik)
    form_types_tuple = tuple(form_types) if form_types else ("10-K", "10-Q", "8-K")
    filings = parse_filings(submissions, ticker, form_types=form_types_tuple, limit=limit)
    return [
        {**f.to_dict(), "filing_url": f.filing_url()}
        for f in filings
    ]


@mcp.tool()
def get_xbrl_facts(ticker: str, concepts: Optional[list[str]] = None) -> list[dict]:
    """Get reported XBRL facts for a ticker (e.g. concepts=["Revenues"]). Omit `concepts`
    to get every reported us-gaap concept (large). Each fact carries the value, unit,
    period, and the accession number of the filing that reported it — the citation."""
    client = get_client()
    cik = client.resolve_cik(ticker)
    facts = client.get_company_facts(cik)
    return [f.to_dict() for f in parse_company_facts(facts, ticker, concepts=concepts)]


@mcp.tool()
def list_concepts(ticker: str) -> list[str]:
    """List every us-gaap XBRL concept a company has reported (e.g. 'Revenues',
    'GrossProfit', 'Assets') — use this to discover what's available before calling
    get_xbrl_facts with a specific concept list."""
    client = get_client()
    cik = client.resolve_cik(ticker)
    facts = client.get_company_facts(cik)
    return list_available_concepts(facts)


if __name__ == "__main__":
    mcp.run()
