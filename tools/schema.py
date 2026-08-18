"""Internal data schema for filings and XBRL facts.

Every retriever tool normalizes EDGAR's raw JSON into these shapes so downstream
phases (reconciler, extraction, orchestration) don't need to know EDGAR's wire format.
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class FilingMeta:
    """One filing (10-K, 10-Q, 8-K, ...) listed in a company's EDGAR submission history."""

    ticker: str
    cik: str
    accession_number: str
    form: str
    filing_date: str
    report_date: str
    primary_document: str
    primary_doc_description: str

    def filing_url(self) -> str:
        """Direct URL to the primary document on EDGAR."""
        accn_nodash = self.accession_number.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/{int(self.cik)}/"
            f"{accn_nodash}/{self.primary_document}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TextChunk:
    """One paragraph-level block of prose from a filing's narrative sections (e.g.
    MD&A) — the source material claim extraction (Phase 4) runs over."""

    ticker: str
    cik: str
    accession_number: str
    form: str
    filing_date: str
    section: str
    chunk_index: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FinancialFact:
    """One reported XBRL data point (a metric, value, unit, and the period it covers),
    tied back to the exact filing that reported it — this is the citation."""

    ticker: str
    cik: str
    concept: str
    label: Optional[str]
    value: float
    unit: str
    period_start: Optional[str]
    period_end: str
    fiscal_year: Optional[int]
    fiscal_period: Optional[str]
    form: str
    filed: str
    accession_number: str

    def to_dict(self) -> dict:
        return asdict(self)
