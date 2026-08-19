"""Retrieval agent: given a ticker, get MD&A text chunks (what extraction reads)
and XBRL facts (what verification checks against). Deterministic — no LLM call,
just calls into Phase 1's Filing Retriever and Phase 2's MD&A Extractor.
"""

from typing import Protocol

from tools.edgar_client import EdgarClient
from tools.mdna_parser import MdnaNotFoundError, chunk_mdna
from tools.schema import FinancialFact, TextChunk
from tools.xbrl_parser import parse_company_facts, parse_filings


class RetrievalAgent(Protocol):
    def get_mdna_chunks(self, ticker: str, form_type: str, limit: int) -> list[TextChunk]: ...

    def get_facts(self, ticker: str, concepts: list[str] | None = None) -> list[FinancialFact]: ...


class RealRetrievalAgent:
    def __init__(self, client: EdgarClient | None = None):
        self.client = client or EdgarClient()

    def get_mdna_chunks(self, ticker: str, form_type: str = "10-K", limit: int = 1) -> list[TextChunk]:
        cik = self.client.resolve_cik(ticker)
        submissions = self.client.get_submissions(cik)
        filings = parse_filings(submissions, ticker, form_types=(form_type,), limit=limit)

        chunks: list[TextChunk] = []
        for filing in filings:
            html = self.client.get_document(filing.filing_url(), cache_key=f"doc_{filing.accession_number}")
            try:
                chunks.extend(
                    chunk_mdna(
                        html,
                        ticker=ticker,
                        cik=cik,
                        accession_number=filing.accession_number,
                        form=filing.form,
                        filing_date=filing.filing_date,
                    )
                )
            except MdnaNotFoundError:
                continue  # this filing's MD&A couldn't be located; skip it, don't fail the whole retrieval
        return chunks

    def get_facts(self, ticker: str, concepts: list[str] | None = None) -> list[FinancialFact]:
        cik = self.client.resolve_cik(ticker)
        raw = self.client.get_company_facts(cik)
        return parse_company_facts(raw, ticker, concepts=concepts)


class MockRetrievalAgent:
    """Returns canned data fixed at construction time — no network, no filesystem."""

    def __init__(self, chunks: list[TextChunk], facts: list[FinancialFact]):
        self._chunks = chunks
        self._facts = facts

    def get_mdna_chunks(self, ticker: str, form_type: str = "10-K", limit: int = 1) -> list[TextChunk]:
        return list(self._chunks)

    def get_facts(self, ticker: str, concepts: list[str] | None = None) -> list[FinancialFact]:
        if concepts is None:
            return list(self._facts)
        return [f for f in self._facts if f.concept in concepts]
