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
pairs instead of silently both resolving to the same one. It also now
resolves an *ordinal* named quarter ("the third quarter", "Q3") against
XBRL's own fiscal-quarter field. See its docstring for what "sequentially"
vs. "a year ago" now does, and what's still not attempted (a quarter named
by calendar month, e.g. "the September quarter" - see resolve_periods).
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
    # Added from a direct check of what real hand-labeled claims (eval/labeled_claims*.jsonl)
    # actually failed to resolve on. Two different failure modes, handled differently:
    # (1) a genuinely new concept, confirmed present in real AAPL/MSFT/NVDA/AMZN
    # company-facts data before being added, same discipline as the block above;
    # (2) a plain wording variant of a concept already mapped above - no new
    # verification needed, since the underlying concept is already confirmed present.
    "net sales": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"],  # AAPL's own term for revenue
    "sales": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"],
    "provision for income taxes": ["IncomeTaxExpenseBenefit"],
    "cash provided by operating activities": ["NetCashProvidedByUsedInOperatingActivities"],
    "cash used in investing activities": ["NetCashProvidedByUsedInInvestingActivities"],
    "cash provided by investing activities": ["NetCashProvidedByUsedInInvestingActivities"],
    "cash flow from investing activities": ["NetCashProvidedByUsedInInvestingActivities"],
    "cash used in financing activities": ["NetCashProvidedByUsedInFinancingActivities"],
    "cash provided by financing activities": ["NetCashProvidedByUsedInFinancingActivities"],
    "cash flow from financing activities": ["NetCashProvidedByUsedInFinancingActivities"],
    "long-term debt": ["LongTermDebt"],
    # Added from research/unresolved_claims_audit.py's tally of the tech-vertical
    # specificity check's "unverifiable" bucket (113 of 137 claims) - same discipline
    # as above: wording variants of already-mapped concepts need no new verification;
    # "interest and debt expense" is a genuinely distinct concept, confirmed present
    # in TXN's real company-facts data (separate from plain InterestExpense).
    "operating profit": ["OperatingIncomeLoss"],
    "total revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "total revenues": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "total gross margin": ["GrossProfit"],
    "gross profit margin": ["GrossProfit"],
    "diluted net income per share": ["EarningsPerShareDiluted"],
    "cash provided by operations": ["NetCashProvidedByUsedInOperatingActivities"],
    "common stock repurchase amount": ["PaymentsForRepurchaseOfCommonStock"],
    "interest and debt expense": ["InterestAndDebtExpense"],
    "interest income": ["InvestmentIncomeInterest"],
    "other income (expense), net": ["OtherNonoperatingIncomeExpense"],
    "other income (expense)": ["OtherNonoperatingIncomeExpense"],
    "long-term lease liabilities": ["OperatingLeaseLiabilityNoncurrent"],
    "operating lease liabilities": ["OperatingLeaseLiabilityNoncurrent"],
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
# free-text period.
_FISCAL_YEAR_RE = re.compile(r"fiscal\s+(?:year\s+)?(\d{4})|\bFY\s?(\d{4})\b", re.IGNORECASE)

# A period hint that's *just* a bare 4-digit year ("2024", or "in 2024" -
# extraction phrasing varies run to run for the identical real sentence),
# nothing else - real case, confirmed against TXN's actual MD&A: "...was 12.4%
# in 2025 compared with 12.0% in 2024." extracts as an absolute claim with
# period="2024" (or "in 2024"), but _FISCAL_YEAR_RE requires a "fiscal"/"FY"
# prefix, so this fell through to the default (most recent) fact and got
# checked against 2025's real number instead. Anchored to the whole string
# (not `search`) so this never fires on a longer, ambiguous hint that merely
# contains a year among other words - only a period extracted as literally
# nothing but a (optionally "in "-prefixed) year is unambiguous enough to trust.
_BARE_YEAR_RE = re.compile(r"^\s*(?:in\s+)?(\d{4})\s*$", re.IGNORECASE)

# Matches an *ordinal* quarter reference - "third quarter", "Q3", "3rd fiscal
# quarter" - which maps directly and unambiguously to XBRL's own fiscal_period
# field ("Q1".."Q4", from the source data's `fp`), since that field is itself
# fiscal-quarter-numbered, not calendar-numbered. Deliberately does NOT match a
# quarter named by calendar month ("the September quarter", "the December
# quarter") - that mapping needs the company's fiscal-year-end to translate
# safely (a September-ending Q1 for one company is another's Q3), which isn't
# available at this resolution step, and guessing wrong here would silently
# pick the wrong period instead of the honest "couldn't tell, use the default."
_QUARTER_WORDS = {"first": "Q1", "second": "Q2", "third": "Q3", "fourth": "Q4"}
_QUARTER_RE = re.compile(
    r"\b(first|second|third|fourth)\s+(?:fiscal\s+)?quarter\b|\bQ([1-4])\b", re.IGNORECASE
)

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

# Reference duration for "one standalone fiscal quarter" - used to disambiguate
# a quarter-hint match from a same-fiscal_period cumulative year-to-date fact.
# 10-Qs commonly report a flow concept (Revenues, NetIncomeLoss, ...) twice for
# the same quarter: once as the standalone 3-month figure, once as the 6- or
# 9-month year-to-date cumulative - both tagged with the identical fiscal_period
# ("Q2", "Q3") and the identical period_end, distinguished only by a shorter
# vs. longer period_start. Confirmed against real NVDA data: an unfiltered
# fiscal_period="Q3" match returned a $91.166B "Q3" figure - actually the
# 9-month year-to-date cumulative, not the ~$35B standalone quarter.
_QUARTER_LENGTH_DAYS = 91


def _period_length_days(fact: FinancialFact) -> int | None:
    if fact.period_start is None:
        return None
    return (date.fromisoformat(fact.period_end) - date.fromisoformat(fact.period_start)).days


def _similar_length(a: int, b: int) -> bool:
    return abs(a - b) <= _LENGTH_TOLERANCE_DAYS


def is_relative_year_ago_hint(period_hint: str) -> bool:
    """True when `period_hint` is a relative "a year ago"/"prior year"-style
    phrase (the same pattern `resolve_periods` already uses to pick a precise
    comparison period). Exposed so a caller checking an *absolute* claim's own
    period (not a growth/change claim's delta reference) can tell the two
    apart - see RealVerificationAgent.verify's docstring for why that
    distinction matters."""
    return bool(_YEAR_AGO_RE.search(period_hint))


def _extract_fiscal_year(period_hint: str) -> int | None:
    match = _FISCAL_YEAR_RE.search(period_hint)
    if match:
        return int(match.group(1) or match.group(2))
    bare = _BARE_YEAR_RE.match(period_hint)
    return int(bare.group(1)) if bare else None


def _extract_fiscal_quarter(period_hint: str) -> str | None:
    match = _QUARTER_RE.search(period_hint)
    if not match:
        return None
    word, digit = match.group(1), match.group(2)
    return _QUARTER_WORDS[word.lower()] if word else f"Q{digit}"


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

    `period_hint` also now resolves an *ordinal* named quarter ("the third
    quarter", "Q3") against XBRL's own fiscal_period field ("Q1".."Q4", from
    the source data's `fp`) - safe because that field is itself fiscal-quarter
    numbered, not calendar-numbered, so no fiscal-year-end knowledge is needed
    to interpret it. Combines with an explicit fiscal year when both are
    present ("the third quarter of fiscal 2025"). Still not attempted: a
    quarter named by calendar month ("the September quarter", "the December
    quarter") - translating that safely needs the company's fiscal-year-end,
    which isn't available here, and guessing wrong would silently pick the
    wrong period instead of the honest "couldn't tell, use the default." Also
    worth knowing rather than assuming fixed: many companies only file a
    standalone 10-Q for Q1-Q3 and cover Q4 inside the annual 10-K, so a
    "fourth quarter" hint can still correctly come back unverifiable if no
    discrete Q4 fact was ever separately tagged - not a bug in this parsing,
    a real gap in what some companies report.
    """
    eligible = facts if as_of is None else [f for f in facts if f.filed <= as_of]
    matching = sorted(
        (f for f in eligible if f.concept == concept), key=lambda f: f.period_end, reverse=True
    )
    if not matching:
        raise ValueError(f"No facts available for concept '{concept}'" + (f" as of {as_of}" if as_of else ""))

    hinted_year = _extract_fiscal_year(period_hint)
    hinted_quarter = _extract_fiscal_quarter(period_hint)
    if hinted_quarter is not None:
        quarter_matches = [f for f in matching if f.fiscal_period == hinted_quarter]
        if hinted_year is not None:
            quarter_matches = [f for f in quarter_matches if f.fiscal_year == hinted_year] or quarter_matches
        # Prefer the standalone ~quarter-length fact over a same-fiscal_period
        # cumulative year-to-date one, when both exist - see _QUARTER_LENGTH_DAYS.
        # Instant (point-in-time) facts have no period_start/length at all and
        # aren't affected by this filter (no YTD-vs-quarter ambiguity for those).
        quarter_length_matches = [
            f for f in quarter_matches
            if (length := _period_length_days(f)) is not None and _similar_length(length, _QUARTER_LENGTH_DAYS)
        ]
        if quarter_length_matches:
            quarter_matches = quarter_length_matches
        if not quarter_matches:
            # A recognized ordinal-quarter hint with nothing to match is a real
            # data gap, not "couldn't tell" - many companies (confirmed: CSCO)
            # only file a standalone 10-Q for Q1-Q3 and fold Q4 into the annual
            # 10-K without a separately tagged Q4 fact. Silently falling back to
            # the most recent fact here used to compare a real CSCO claim ("the
            # fourth quarter of fiscal 2025... total revenue increased by 8%")
            # against full-year totals instead, making an accurate quarterly
            # claim look wildly wrong. This was already the documented intent
            # above ("can still correctly come back unverifiable") - just not
            # actually implemented that way.
            raise ValueError(
                f"No '{hinted_quarter}' fact available for concept '{concept}'"
                + (f" as of {as_of}" if as_of else "")
                + " - this company may not separately report this quarter."
            )
        current = quarter_matches[0]
    elif hinted_year is not None:
        # A bare fiscal-year hint (no quarter marker - that's the elif branch
        # above) always means the ANNUAL figure, never a quarter that happens to
        # carry the same fiscal_year number. That distinction matters because
        # `fiscal_year` tagging can genuinely disagree between a company's own
        # annual and quarterly filings for the SAME fiscal year - confirmed
        # against real CRM data: the FY2026 10-K's own annual facts (period
        # 2025-02-01 to 2026-01-31, filed with that 10-K) are tagged
        # fiscal_year=2025, while the Q3 FY2026 10-Q's facts (an earlier,
        # smaller, wrong period) are tagged fiscal_year=2026 - matching the
        # company's own "fiscal 2026" prose by number, but only by accident.
        # Preferring any annual (fiscal_period == "FY") fact over a same-
        # numbered quarterly one - and tolerating the annual fact's number
        # being one less than the hint, not just an exact match - fixes this
        # without weakening the exact-match case every other company already
        # relies on (checked first, unconditionally preferred when present).
        annual_matches = [f for f in matching if f.fiscal_period == "FY"]
        exact_annual = [f for f in annual_matches if f.fiscal_year == hinted_year]
        off_by_one_annual = [f for f in annual_matches if f.fiscal_year == hinted_year - 1]
        if exact_annual:
            current = exact_annual[0]
        elif off_by_one_annual:
            current = off_by_one_annual[0]
        else:
            year_matches = [f for f in matching if f.fiscal_year == hinted_year]
            current = year_matches[0] if year_matches else matching[0]
    else:
        current = matching[0]

    current_idx = next(i for i, f in enumerate(matching) if f is current)
    older = matching[current_idx + 1 :]  # already sorted by period_end descending

    def _is_distinct(f: FinancialFact) -> bool:
        return (f.period_start, f.period_end) != (current.period_start, current.period_end)

    current_length = _period_length_days(current)

    # The plain "next distinct period, sorted by period_end descending" pick
    # below is unsafe on its own: a company that's filed a 10-Q since the
    # current annual fact reports a same-concept year-to-date cumulative fact
    # whose period_end falls between current's and the true prior-year fact's -
    # e.g. a Q3 10-Q's 9-month-YTD revenue sorts ahead of the correct prior
    # full year. Confirmed against real TXN data via research/specificity_check.py:
    # current = FY2025 revenue ($17.682B, period_end 2025-12-31); the naive
    # "next distinct" pick grabbed a $13.259B *9-month* YTD fact (period_end
    # 2025-09-30, from a 10-Q) instead of the correct $15.641B FY2024 annual
    # figure, making a genuinely accurate claim ("increased $2.04B, or 13.0%,
    # compared to fiscal 2024") read as wildly inconsistent. Prefer a distinct
    # older period whose duration matches current's (same discipline already
    # used for the "sequentially"/"a year ago" hinted paths below); only fall
    # back to the unfiltered pick when no similar-length candidate exists at
    # all (e.g. a company with only quarterly history to compare an annual
    # figure against).
    if current.period_start is None:
        # An INSTANT fact (a balance as of one date - RemainingPerformanceObligation,
        # not a duration like revenue) has no length for the same-length check above
        # to match on at all, so it silently never applied here and this concept type
        # kept using the unsafe "next distinct period_end" pick the duration fix above
        # was meant to replace. Real MD&A prose for these metrics is almost always
        # compared year-over-year ("$72.4 billion, an increase of 14 percent
        # year-over-year") - confirmed against real CRM data: current = $72.4B RPO
        # (as of 2026-01-31); the true comparable prior instant is $63.4B (as of
        # 2025-01-31, matching the stated 14% exactly), but the naive "next distinct"
        # pick grabbed a $59.5B intervening quarterly balance (as of 2025-10-31)
        # instead, implying 21.7% and making an accurate claim look wildly wrong.
        # Default to the closest distinct instant ~1 year before current's own date.
        instant_older = [f for f in older if _is_distinct(f) and f.period_start is None]
        closest_year_ago_instant = None
        if instant_older and current.period_end is not None:
            cur_end = date.fromisoformat(current.period_end)
            try:
                target_end = cur_end.replace(year=cur_end.year - 1)
            except ValueError:
                target_end = cur_end.replace(year=cur_end.year - 1, day=28)  # Feb 29 -> Feb 28
            closest = min(instant_older, key=lambda f: abs((date.fromisoformat(f.period_end) - target_end).days))
            if abs((date.fromisoformat(closest.period_end) - target_end).days) <= _YEAR_AGO_TOLERANCE_DAYS:
                closest_year_ago_instant = closest
        default_comparison = closest_year_ago_instant or next(
            (f for f in older if _is_distinct(f)), None
        )
    else:
        same_length_older = [
            f for f in older
            if _is_distinct(f) and current_length is not None
            and (length := _period_length_days(f)) is not None and _similar_length(length, current_length)
        ]
        default_comparison = same_length_older[0] if same_length_older else next(
            (f for f in older if _is_distinct(f)), None
        )

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
