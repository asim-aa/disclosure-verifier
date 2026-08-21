"""Internal data schema for filings and XBRL facts.

Every retriever tool normalizes EDGAR's raw JSON into these shapes so downstream
phases (reconciler, extraction, orchestration) don't need to know EDGAR's wire format.
"""

from dataclasses import asdict, dataclass


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
    label: str | None
    value: float
    unit: str
    period_start: str | None
    period_end: str
    fiscal_year: int | None
    fiscal_period: str | None
    form: str
    filed: str
    accession_number: str

    def to_dict(self) -> dict:
        return asdict(self)


# Recognized ways a claim can compare a metric to source data.
COMPARISON_ABSOLUTE = "absolute"  # claimed_value should equal the reported metric
COMPARISON_GROWTH_PCT = "growth_pct"  # claimed_value is % change vs a comparison period
COMPARISON_ABSOLUTE_CHANGE = "absolute_change"  # claimed_value is the $ change vs a comparison period
COMPARISON_BPS_CHANGE = "bps_change"  # claimed_value is the bps change in a ratio
COMPARISON_TYPES = (
    COMPARISON_ABSOLUTE,
    COMPARISON_GROWTH_PCT,
    COMPARISON_ABSOLUTE_CHANGE,
    COMPARISON_BPS_CHANGE,
)

VERDICT_CONSISTENT = "consistent"
VERDICT_INCONSISTENT = "inconsistent"
VERDICT_UNVERIFIABLE = "unverifiable"  # source data needed to check the claim is missing

# Machine-readable reason codes for *why* a verdict came out the way it did — a
# typed complement to ReconciliationResult.explanation's free text. Two uses:
# (1) reward shaping for Phase 7 (a wrong verdict from a genuinely ambiguous
# period is a different training signal than one from a clean large numeric
# miss), and (2) error-analysis reporting that doesn't require re-parsing prose.
REASON_MATCH = "match"  # consistent
REASON_NEAR_MISS = "near_miss"  # inconsistent, difference within 2x tolerance
REASON_LARGE_MISS = "large_miss"  # inconsistent, difference beyond 2x tolerance
REASON_MISSING_FACT = "missing_fact"  # unverifiable: no matching fact in the retrieved data
REASON_AMBIGUOUS_PERIOD = "ambiguous_period"  # unverifiable: multiple period_start candidates
REASON_ZERO_DENOMINATOR = "zero_denominator"  # unverifiable: division by a zero prior/denominator value
REASON_MISSING_COMPARISON_CONTEXT = "missing_comparison_context"  # unverifiable: claim omitted required fields
REASON_UNSUPPORTED_COMPARISON_TYPE = "unsupported_comparison_type"  # unverifiable: unrecognized comparison_type


@dataclass(frozen=True)
class Claim:
    """A discrete, checkable quantitative claim (Pillar 2's extraction target): a
    metric, a value, a period, and how the value relates to the source data.

    - absolute: `metric` at `period` should equal `claimed_value`.
    - growth_pct: percent change in `metric` from `comparison_period` to `period`
      should equal `claimed_value` (e.g. "revenue grew 12% YoY" -> claimed_value=12.0).
    - absolute_change: dollar change in `metric` from `comparison_period` to `period`
      should equal `claimed_value` (e.g. "revenue increased $50.1 billion" ->
      claimed_value=50_100_000_000.0).
    - bps_change: change in the ratio `metric / denominator_metric`, in basis points,
      from `comparison_period` to `period` (e.g. "gross margin expanded 200 bps" with
      metric="GrossProfit", denominator_metric="Revenues" -> claimed_value=200.0).
    """

    ticker: str
    metric: str
    comparison_type: str
    claimed_value: float
    period_end: str
    period_start: str | None = None
    comparison_period_end: str | None = None
    comparison_period_start: str | None = None
    denominator_metric: str | None = None
    unit: str = "USD"
    tolerance: float | None = None  # None -> use the type's default tolerance

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ReconciliationResult:
    """The outcome of checking a Claim against retrieved FinancialFacts — the
    programmatically verifiable signal Pillar 4's RLVR reward function reuses."""

    verdict: str  # consistent | inconsistent | unverifiable
    claim: Claim
    computed_value: float | None
    difference: float | None
    tolerance: float
    explanation: str
    citations: list[str]  # accession numbers of the facts used to compute this
    reason_code: str = "unknown"  # one of the REASON_* constants above

    def to_dict(self) -> dict:
        return {**asdict(self), "claim": self.claim.to_dict()}
