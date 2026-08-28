"""Resolves an ExtractedClaim (free-text metric + free-text period) into a
tools.schema.Claim (exact XBRL concept + ISO period dates) that the reconciler can
check. This is the "later resolution step" explicitly deferred out of Phase 4's
scope — Phase 5's coordinator needs it for real end-to-end verification.

Honest scope limit, now confirmed precisely rather than assumed: this only
resolves metrics with a clean, confident mapping to a single standard,
*non-dimensional* us-gaap concept. Phase 6's integration-at-scale run found
segment/product-level claims ("Azure and other cloud services revenue", "Xbox
hardware revenue", "LinkedIn revenue", "Data Center revenue") dominate real MD&A
prose, and inspecting the actual data confirmed why they can't resolve here: SEC's
company-facts API (what the Filing Retriever uses) returns each data point as a
flat `{start, end, val, accn, fy, fp, form, filed, frame}` record with *no*
segment/member/axis field at all — this data source has no per-segment breakdown
to find, at any level of effort in this file. Reaching it would mean parsing a
filing's raw XBRL instance document or its rendered financial-statement "R" pages
instead — a genuinely different data source, out of scope here. Those claims
correctly come back unresolved rather than guessing a wrong concept — a false
"verified" claim would be worse than an honest "can't check this yet".

What Phase 6 also showed: some of what looked like segment-specific coverage gaps
were actually just missing dictionary entries for real, standard, non-dimensional
concepts ("Commercial remaining performance obligation" is `RevenueRemainingPerformanceObligation`,
a genuine top-level us-gaap concept, not a segment figure) — those are added below,
each verified present in real AAPL/MSFT/NVDA company-facts data before being added,
not guessed from the XBRL taxonomy's existence alone.

`resolve_periods` below also closes a real bug that was flagged as a known
limitation rather than fixed for several phases: it now reads the claim's own
free-text period (`ExtractedClaim.period`) instead of ignoring it, so two
claims from the same sentence naming two different fiscal years ("fiscal
years 2026 and 2025, respectively") resolve to two different, correct period
pairs instead of silently both resolving to the same one. See its docstring
for what "sequentially" vs. "a year ago" now does, and what's still not
attempted (named quarters in prose).
"""

import re
from datetime import date

from tools.schema import FinancialFact

# metric text (lowercased) -> candidate XBRL concepts in priority order. Verified
# against real AAPL/MSFT/NVDA company-facts concept listings, not guessed. Known
# close variants are listed as their own explicit keys (not handled by fuzzy
# matching — a naive substring/overlap check here is actively dangerous: "revenue"
# is a substring of "Azure and other cloud services revenue", so it would resolve
# a segment-specific claim against *total company* revenue data and produce a
# confidently wrong verdict. Better to require an exact, deliberate mapping and
# leave anything else unresolved than to guess.
METRIC_TO_CONCEPTS: dict[str, list[str]] = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"],
    "cost of revenue": ["CostOfRevenue"],
    "gross margin": ["GrossProfit"],
    "gross margin percentage": ["GrossProfit"],
    "gross profit": ["GrossProfit"],
    "operating expenses": ["OperatingExpenses"],
    "operating income": ["OperatingIncomeLoss"],
    "income tax expense": ["IncomeTaxExpenseBenefit"],
    "effective tax rate": ["EffectiveIncomeTaxRateContinuingOperations"],
    "cash dividends paid": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "common stock repurchased": ["PaymentsForRepurchaseOfCommonStock"],
    "quarterly cash dividend per share": ["CommonStockDividendsPerShareDeclared"],
    "u.s. income before income taxes": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic"],
    "foreign income before income taxes": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesForeign"],
    # Added after Phase 6's integration-at-scale run — each confirmed present with
    # real, plausible values in cached AAPL/MSFT/NVDA company-facts data, not added
    # on the strength of existing in the us-gaap taxonomy alone.
    "net income": ["NetIncomeLoss"],
    "diluted earnings per share": ["EarningsPerShareDiluted"],
    "basic earnings per share": ["EarningsPerShareBasic"],
    "total assets": ["Assets"],
    "current assets": ["AssetsCurrent"],
    "total liabilities": ["Liabilities"],
    "current liabilities": ["LiabilitiesCurrent"],
    "total stockholders' equity": ["StockholdersEquity"],
    "stockholders' equity": ["StockholdersEquity"],
    "shareholders' equity": ["StockholdersEquity"],
    "cash and cash equivalents": ["CashAndCashEquivalentsAtCarryingValue"],
    "research and development expense": ["ResearchAndDevelopmentExpense"],
    "research and development expenses": ["ResearchAndDevelopmentExpense"],
    "selling, general and administrative expenses": ["SellingGeneralAndAdministrativeExpense"],
    "selling, general and administrative expense": ["SellingGeneralAndAdministrativeExpense"],
    "general and administrative expenses": ["GeneralAndAdministrativeExpense"],
    "general and administrative expense": ["GeneralAndAdministrativeExpense"],
    "depreciation and amortization": ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet"],
    "net cash provided by operating activities": ["NetCashProvidedByUsedInOperatingActivities"],
    "cash flow from operations": ["NetCashProvidedByUsedInOperatingActivities"],
    "operating cash flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capital expenditures": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "purchases of property and equipment": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "remaining performance obligation": ["RevenueRemainingPerformanceObligation"],
    "commercial remaining performance obligation": ["RevenueRemainingPerformanceObligation"],
    "total remaining performance obligation": ["RevenueRemainingPerformanceObligation"],
    "total costs and expenses": ["CostsAndExpenses"],
    "interest expense": ["InterestExpense"],
}


def resolve_concept(metric_text: str, facts: list[FinancialFact], as_of: str | None = None) -> str | None:
    """Return whichever candidate concept for `metric_text` has the most recently
    reported data in `facts` as of `as_of` (a filing date — see resolve_periods
    for why this cutoff matters), or None if the metric has no known mapping or
    none of its candidates appear at all before that cutoff.

    Picks by data recency, not just "first candidate that exists at all" —
    companies retag concepts over time (confirmed against real data: NVDA
    reported revenue under RevenueFromContractWithCustomerExcludingAssessedTax
    through FY2022, then switched to plain Revenues; that old tag never
    disappears from the company's concept list, it just stops getting new data).
    Picking the first *existing* candidate would silently lock onto a
    discontinued tag and reconcile claims against multi-year-stale data. Exact
    metric-name match only (after normalization) — see the module/table
    docstring for why fuzzy matching isn't safe here."""
    key = metric_text.strip().lower()
    candidates = METRIC_TO_CONCEPTS.get(key)
    if not candidates:
        return None

    eligible = facts if as_of is None else [f for f in facts if f.filed <= as_of]

    best_concept, best_period_end = None, None
    for concept in candidates:
        period_ends = [f.period_end for f in eligible if f.concept == concept]
        if not period_ends:
            continue
        latest = max(period_ends)
        if best_period_end is None or latest > best_period_end:
            best_concept, best_period_end = concept, latest

    return best_concept


# Matches "fiscal year 2025", "fiscal 2025", or "FY2025"/"FY 2025" in a claim's
# free-text period. Deliberately does NOT try to parse a named quarter ("the
# September quarter", "Q3") into a specific fiscal_period - real filing prose
# names quarters inconsistently enough (calendar month vs. fiscal quarter
# number vs. "the December quarter") that guessing wrong would silently pick
# the wrong period instead of the honest "couldn't tell, use the default."
_FISCAL_YEAR_RE = re.compile(r"fiscal\s+(?:year\s+)?(\d{4})|\bFY\s?(\d{4})\b", re.IGNORECASE)

# "sequentially" / "sequential" signals the comparison period is the immediately
# preceding period of the SAME length (e.g. prior quarter), not a year prior.
_SEQUENTIAL_RE = re.compile(r"\bsequential(?:ly)?\b", re.IGNORECASE)

# "a year ago", "year-over-year", "prior year", "same quarter last year" all
# signal the comparison period is ~12 months before the current one, same
# length - as opposed to "sequentially" (immediately preceding period).
_YEAR_AGO_RE = re.compile(
    r"\ba year (?:ago|earlier)\b|\byear[- ]over[- ]year\b|\bprior year\b|\bsame (?:period|quarter) last year\b",
    re.IGNORECASE,
)

# Tolerance for matching a "year ago" candidate by calendar proximity to
# (current period start - 1 year) - real fiscal calendars drift by a few days
# year to year (52/53-week fiscal years, weekend-adjusted quarter ends), but a
# match 2+ months off is more likely a different, wrong period than a shifted
# same one.
_YEAR_AGO_TOLERANCE_DAYS = 60

# Tolerance for treating two periods as "the same length" (e.g. both quarters).
# Real calendar quarters aren't exactly equal length - Jan-Mar is ~89 days,
# Oct-Dec is ~92 - so exact day-count equality would wrongly treat two
# adjacent real quarters as different lengths and miss a genuine sequential
# match. 20 days safely absorbs that variation while staying far short of the
# ~90-day gap between a quarter and a half-year, or the ~275-day gap to a
# full year - so it can't accidentally match a quarter against either.
_LENGTH_TOLERANCE_DAYS = 20


def _period_length_days(fact: FinancialFact) -> int | None:
    if fact.period_start is None:
        return None
    return (date.fromisoformat(fact.period_end) - date.fromisoformat(fact.period_start)).days


def _similar_length(a: int, b: int) -> bool:
    return abs(a - b) <= _LENGTH_TOLERANCE_DAYS


def _extract_fiscal_year(period_hint: str) -> int | None:
    match = _FISCAL_YEAR_RE.search(period_hint)
    if not match:
        return None
    return int(match.group(1) or match.group(2))


def resolve_periods(
    facts: list[FinancialFact], concept: str, as_of: str | None = None, period_hint: str = ""
) -> tuple[FinancialFact, FinancialFact | None]:
    """Pick the FinancialFact for `concept` that matches `period_hint` (or, if
    `period_hint` doesn't name a specific period, the most recent reporting) as
    of `as_of`, and a comparison FinancialFact for the prior period `period_hint`
    implies (or, by default, the next most recent *distinct* period). Returns
    (current, comparison); comparison is None if no matching/distinct period is
    available before the cutoff.

    `as_of` should be the filing_date of the filing the claim's source paragraph
    came from — without it, "most recent" means most recent across the
    company's *entire* history, which is wrong: a claim extracted from a 10-K's
    MD&A describes that 10-K's own fiscal year, but if a later 10-Q has since
    been filed (a very likely case — 10-Qs come out quarterly), that 10-Q's more
    recent *quarterly* period_end would otherwise get picked as "current"
    instead of the 10-K's own annual period, comparing the wrong two numbers
    entirely. Confirmed against real data: extracting from NVDA's FY2026 10-K
    without this cutoff picked up the already-filed Q1 FY2027 10-Q's quarterly
    revenue as "current", making a correct claim look wildly inconsistent.

    Returns the actual FinancialFact objects (not just period tuples) so the
    caller can read the real reported `.unit` off them — XBRL doesn't use a
    consistent unit per concept type (dollar metrics are "USD", per-share metrics
    are "USD/shares", rate/percentage metrics are "pure" decimal fractions like
    0.19 for 19%) — guessing the unit from the claim's own wording instead of the
    actual matched fact would silently fail to find data, or worse, compare a
    claimed percentage (19.0) against a fraction (0.19) and call it inconsistent.

    `period_hint` (typically the claim's own `ExtractedClaim.period` text) fixes
    a real bug, confirmed against live data (NVDA 10-K, "Income tax expense was
    $21.4 billion and $11.1 billion for fiscal years 2026 and 2025,
    respectively" — extracted as two separate claims, one per stated period):
    without a hint, both claims resolved to the SAME (current, comparison) pair,
    so the FY2025 claim was compared against the FY2026 fact and looked wrong
    even when correct. When `period_hint` names an explicit fiscal year ("fiscal
    year 2025", "FY2025") and a fact for that year exists, that fact becomes
    `current` instead of always the globally most recent one - and the
    comparison is then picked relative to *that* fact, not the global most
    recent, so it doesn't end up "compared against a later period" backwards.
    `period_hint` also distinguishes "sequentially" (comparison = immediately
    preceding period, same length) from "a year ago"/"year-over-year" (comparison
    = ~12 months prior, same length) for growth_pct claims - previously always
    picked the next most recent distinct period regardless of which the text
    actually meant. An empty, unparseable, or non-matching hint falls back to
    the original default behavior unchanged.

    Known limitation, still not attempted: a named quarter in prose ("the
    September quarter", "the third quarter") isn't parsed into a specific
    fiscal_period - real filings name quarters inconsistently (calendar month
    vs. fiscal quarter number) in ways that risk a wrong-but-confident match.
    Left unresolved-by-default rather than guessed.
    """
    eligible = facts if as_of is None else [f for f in facts if f.filed <= as_of]
    matching = sorted(
        (f for f in eligible if f.concept == concept), key=lambda f: f.period_end, reverse=True
    )
    if not matching:
        raise ValueError(f"No facts available for concept '{concept}'" + (f" as of {as_of}" if as_of else ""))

    hinted_year = _extract_fiscal_year(period_hint)
    if hinted_year is not None:
        year_matches = [f for f in matching if f.fiscal_year == hinted_year]
        current = year_matches[0] if year_matches else matching[0]
    else:
        current = matching[0]

    current_idx = next(i for i, f in enumerate(matching) if f is current)
    older = matching[current_idx + 1 :]  # already sorted by period_end descending

    def _is_distinct(f: FinancialFact) -> bool:
        return (f.period_start, f.period_end) != (current.period_start, current.period_end)

    default_comparison = next((f for f in older if _is_distinct(f)), None)

    current_length = _period_length_days(current)
    comparison = default_comparison

    if _SEQUENTIAL_RE.search(period_hint) and current_length is not None:
        same_length = [
            f for f in older
            if _is_distinct(f) and (length := _period_length_days(f)) is not None and _similar_length(length, current_length)
        ]
        if same_length:
            comparison = same_length[0]  # nearest preceding period of a similar length
    elif _YEAR_AGO_RE.search(period_hint) and current.period_start is not None and current_length is not None:
        cur_start = date.fromisoformat(current.period_start)
        try:
            target_start = cur_start.replace(year=cur_start.year - 1)
        except ValueError:
            target_start = cur_start.replace(year=cur_start.year - 1, day=28)  # Feb 29 -> Feb 28
        same_length = [
            f for f in older
            if _is_distinct(f) and f.period_start
            and (length := _period_length_days(f)) is not None and _similar_length(length, current_length)
        ]
        if same_length:
            closest = min(same_length, key=lambda f: abs((date.fromisoformat(f.period_start) - target_start).days))
            if abs((date.fromisoformat(closest.period_start) - target_start).days) <= _YEAR_AGO_TOLERANCE_DAYS:
                comparison = closest

    return current, comparison
