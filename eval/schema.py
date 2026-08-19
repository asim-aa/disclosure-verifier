"""Schema for Pillar 2's claim-extraction task: given a paragraph of prose, extract
discrete, checkable claims.

Deliberately distinct from tools.schema.Claim: that type requires an exact XBRL
concept name and ISO period dates, which an LLM reading prose can't produce
reliably ("revenue" isn't "RevenueFromContractWithCustomerExcludingAssessedTax",
and "the September quarter" isn't "2023-09-30"). ExtractedClaim captures what's
actually extractable from text — free-text metric/period descriptions plus the
numeric value and comparison type — using mostly the same comparison_type
vocabulary as the reconciler (absolute/growth_pct/bps_change), plus one type the
reconciler doesn't have yet (absolute_change — see below), so a later resolution
step (out of scope for Phase 4) can map ExtractedClaim -> Claim and hand most of
these straight to Phase 3.

comparison_type is one of:
  - "absolute": a single reported value at one period ("revenue was $215.9 billion",
    "gross margin was 71.1% in fiscal 2026"). If a sentence states two absolute
    values at two periods side by side with no stated delta ("...71.1% in fiscal
    2026 from 75.0% in fiscal 2025"), that's TWO absolute claims — never compute
    the delta yourself; only assign growth_pct/absolute_change/bps_change when the
    text itself states the change.
  - "growth_pct": an explicit percent change vs. a prior period ("increased 27%",
    "up 65% from a year ago").
  - "absolute_change": an explicit dollar-amount change vs. a prior period
    ("increased $50.1 billion"). Real filings very commonly state a claim as BOTH
    a dollar delta and a percent in the same sentence ("Revenue increased $50.1
    billion or 18%") — that's two separate claims, one of each type, not one.
    This type has no counterpart in tools.schema.Claim yet; extending the
    reconciler to check it is follow-up work, not done as part of Phase 4.
  - "bps_change": an explicit basis-point (or percentage-point) change in a ratio
    vs. a prior period ("margin expanded 200 bps"). Rare in practice — MSFT/NVDA
    both report margin changes as two absolute percentages rather than a stated
    delta, so most margin claims end up "absolute" x2, not "bps_change".
"""

from pydantic import BaseModel, Field

COMPARISON_ABSOLUTE = "absolute"
COMPARISON_GROWTH_PCT = "growth_pct"
COMPARISON_ABSOLUTE_CHANGE = "absolute_change"
COMPARISON_BPS_CHANGE = "bps_change"
COMPARISON_TYPES = (
    COMPARISON_ABSOLUTE,
    COMPARISON_GROWTH_PCT,
    COMPARISON_ABSOLUTE_CHANGE,
    COMPARISON_BPS_CHANGE,
)


class ExtractedClaim(BaseModel):
    metric: str = Field(description="The financial metric being claimed, in the words used by the text (e.g. 'revenue', 'gross margin', 'iPhone segment revenue').")
    value: float = Field(description="The numeric value claimed (e.g. 12.0 for '12%', 391000000000 for '$391 billion', 200.0 for '200 basis points').")
    value_unit: str = Field(description="Unit of `value`: 'USD', 'percent', or 'bps'.")
    period: str = Field(description="The period the claim is about, in the words used by the text (e.g. 'fiscal 2024', 'the September quarter'). Empty string if the paragraph doesn't state one.")
    comparison_type: str = Field(description="One of: 'absolute' (a claimed value at one period), 'growth_pct' (a claimed percent change vs. a prior period), 'absolute_change' (a claimed dollar-amount change vs. a prior period), 'bps_change' (a claimed basis-point change in a ratio vs. a prior period).")
    quote: str = Field(description="The exact verbatim span of the source text this claim was extracted from.")


class ExtractedClaims(BaseModel):
    claims: list[ExtractedClaim]
