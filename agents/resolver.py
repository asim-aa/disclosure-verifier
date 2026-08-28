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
"""

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


def resolve_periods(
    facts: list[FinancialFact], concept: str, as_of: str | None = None
) -> tuple[FinancialFact, FinancialFact | None]:
    """Pick the FinancialFact for the most recent reporting of `concept` as of
    `as_of`, and one for the next most recent *distinct* period as the
    comparison. Returns (current, comparison); comparison is None if no second
    distinct period is available before the cutoff.

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

    Known limitation, confirmed against real data (NVDA 10-K, "Income tax expense
    was $21.4 billion and $11.1 billion for fiscal years 2026 and 2025,
    respectively" — extracted as two separate claims, one per stated period):
    this function never looks at the claim's own `period` text at all, so both
    claims resolve to the SAME (current, comparison) pair — the FY2025 claim
    ends up compared against the FY2026 fact and looks wrong even when it's
    correct. It also doesn't distinguish "sequentially" from "a year ago" for
    growth_pct claims (prior quarter vs. same quarter last year) — always picks
    the next most recent distinct period regardless. Properly fixing this means
    parsing free-text period phrasing ("fiscal 2025", "the September quarter",
    "sequentially") into actual period matches — real work, out of scope for
    Phase 5's routing/mock-testing focus. Flagged as follow-up, not solved here.
    """
    eligible = facts if as_of is None else [f for f in facts if f.filed <= as_of]
    matching = sorted(
        (f for f in eligible if f.concept == concept), key=lambda f: f.period_end, reverse=True
    )
    if not matching:
        raise ValueError(f"No facts available for concept '{concept}'" + (f" as of {as_of}" if as_of else ""))

    current = matching[0]
    comparison = next(
        (f for f in matching[1:] if (f.period_start, f.period_end) != (current.period_start, current.period_end)),
        None,
    )
    return current, comparison
