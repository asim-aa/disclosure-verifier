"""MCP server: MD&A Extractor.

Pulls the Management's Discussion and Analysis section out of a company's 10-K/10-Q
filings, chunked into paragraph-level text — the claim source used in place of
earnings-call transcripts (see the proposal's Week 13 scope safeguard). This is
Pillar 1, tool 2 of 3.

Cost/latency: dominated by fetching and parsing each filing's full HTML document
(disk-cached by accession number after the first fetch — see
tools/edgar_client.py); `limit` controls how many filings are parsed per call,
so cost scales linearly with it, not with document size alone.

Error taxonomy: an invalid `form_type` raises ValueError immediately (fix the
call, don't retry). A filing whose MD&A section can't be located is not an
exception — it's included in the returned list as an `{"error": ...}` entry
carrying that filing's own accession_number/form/filing_date, so a caller
processing multiple filings gets partial results plus an explicit account of
what was skipped, not a silent gap or a hard failure of the whole batch.

Run directly for a stdio smoke test:
    python -m tools.mdna_extractor
"""


from mcp.server.fastmcp import FastMCP

from tools.edgar_client import EdgarClient
from tools.mdna_parser import MdnaNotFoundError, chunk_mdna
from tools.xbrl_parser import parse_filings

mcp = FastMCP("mdna-extractor")

_client: EdgarClient | None = None


def get_client() -> EdgarClient:
    global _client
    if _client is None:
        _client = EdgarClient()
    return _client


@mcp.tool()
def get_mdna(ticker: str, form_type: str = "10-K", limit: int = 1) -> list[dict]:
    """Get MD&A prose chunks from a company's most recent 10-K or 10-Q filings.
    Each chunk carries the accession number and filing date it came from — the
    citation. Filings where MD&A can't be located are skipped and reported
    separately rather than silently omitted."""
    if form_type not in ("10-K", "10-Q"):
        raise ValueError("form_type must be '10-K' or '10-Q'")

    client = get_client()
    cik = client.resolve_cik(ticker)
    submissions = client.get_submissions(cik)
    filings = parse_filings(submissions, ticker, form_types=(form_type,), limit=limit)

    chunks: list[dict] = []
    for filing in filings:
        html = client.get_document(
            filing.filing_url(), cache_key=f"doc_{filing.accession_number}"
        )
        try:
            filing_chunks = chunk_mdna(
                html,
                ticker=ticker,
                cik=cik,
                accession_number=filing.accession_number,
                form=filing.form,
                filing_date=filing.filing_date,
            )
        except MdnaNotFoundError as exc:
            chunks.append(
                {
                    "error": str(exc),
                    "accession_number": filing.accession_number,
                    "form": filing.form,
                    "filing_date": filing.filing_date,
                }
            )
            continue

        chunks.extend(c.to_dict() for c in filing_chunks)

    return chunks


if __name__ == "__main__":
    mcp.run()
