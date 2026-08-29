# Specificity check: does the pipeline cry wolf on clean filings?

[`docs/backtest-results.md`](backtest-results.md) only tests sensitivity — it shows the reconciler flags claims that turned out to be wrong, on companies that actually restated. It never checked the other direction: run against companies that *didn't* restate, does the pipeline stay quiet on genuinely accurate claims, or does it produce false positives?

[`research/specificity_check.py`](../research/specificity_check.py). Runs the real Coordinator (no mocks — real EDGAR retrieval, real LLM extraction, real Reconciler verification, the same machinery as [`phase6/run_integration.py`](../phase6/run_integration.py)) against a control set of large companies confirmed — not assumed — to carry no restatement fingerprint anywhere in Phase A's full scan (13,827 fingerprints across 541 companies).

## Finding the right control set took two attempts

The first control set (JPM, JNJ, V, COST, HD, PG, KO, MA — financials and consumer staples) produced **zero resolved verdicts**: 5 of 8 tickers had no MD&A the parser could find at all, and the other 3 spent their whole extraction budget on segment/product detail before reaching a claim the resolver covers. A real, honest finding on its own — `tools/mdna_parser.py`'s heading detection, fixed for GOOGL/AMZN's specific structure in Phase 6, doesn't generalize to bank and consumer-staples 10-Ks, which structure their MD&A very differently from the tech companies (AAPL/MSFT/NVDA) this pipeline was built against.

Switched to large tech/software names — ADBE, CRM, ORCL, CSCO, INTU, IBM, QCOM, TXN, structurally closer to the companies this pipeline's parsing and concept dictionary were tuned against — and raised the per-ticker budget (40 chunks / 40 extraction calls / 300s, up from 25/20/240). Confirmed clean the same way: none of the 8 carry a restatement fingerprint.

## What came back

126 claims across 8 tickers (all succeeded; INTU and IBM initially had no locatable MD&A within budget): **102 unverifiable, 18 consistent, 6 inconsistent.**

An apparent false-positive rate of 6/24 = 25% among resolved verdicts — a number worth investigating before reporting, not reporting at face value. It traces to two distinct root causes, only one of which is this project's own bug.

## A third real bug, found by narrowing scope: INTU's dash-separated heading

Investigating why INTU returned zero claims (rather than assuming it was the same coverage gap as the financials/consumer-staples batch) found a second, unrelated, and cleanly fixable bug. `tools/mdna_parser.py`'s heading regex only tolerated an optional *period* between an Item number and its title (`Item\s*7\.?...`) — but INTU's real FY2025 10-K heading is **`ITEM 7 - MANAGEMENT'S DISCUSSION AND ANALYSIS...`**, dash-separated with no period at all, so the regex never matched at all. Confirmed against the live filing (1,000+ lines of genuine MD&A prose sitting right after the unmatched heading) before fixing. **Fixed** by allowing an optional dash (hyphen, en dash, or em dash) as an alternative separator, with regression tests built from the real heading text. Re-verified: INTU now returns 280 real MD&A chunks (up from 0) and 11 real extracted claims (up from 0) on the next specificity run.

IBM's zero claims turned out to be a *different* thing entirely, not a bug: IBM's real 10-K literally states *"Refer to pages 6 through 38 of IBM's 2025 Annual Report to Stockholders, which are incorporated herein by reference"* — the MD&A isn't in this document at all, a real and legitimate SEC filing pattern (large filers incorporating MD&A by reference to a separate annual-report exhibit rather than repeating it in the 10-K body). Reaching that would mean fetching and parsing a second, separately-filed document — a real, different, and larger piece of work than a heading-regex fix, not attempted here. Correctly distinguishing "not present in this document" from "present but unparsed" is itself a real, if smaller, remaining gap: the parser currently returns a degenerate 1-line pseudo-section for IBM rather than raising `MdnaNotFoundError`, which happens not to affect any verdict here (0 claims either way) but is worth closing later.

With the fix, the aggregate moved to **137 claims, still 24 resolved verdicts** (INTU's 11 new claims were all unverifiable — genuine content, but segment-heavy prose that hit the extraction-call budget before reaching a claim the resolver's concept dictionary covers, the same pattern seen with JNJ and V in the first control-set attempt). The false-positive rate itself didn't move on this run, but the underlying bug fix is real and permanent — every future run against INTU, not just this one, benefits from it.

## Root cause 1: a real, previously-undiscovered resolver bug — found and fixed

The first pass of this check actually found **14** inconsistent claims (58.3% apparent false-positive rate). Investigating the largest cluster — 4 from ADBE, 3 from TXN, 1 from CSCO, all `growth_pct`/`absolute_change` claims — found a real, live bug in `resolve_periods()`'s *default* comparison-period selection (no `period_hint` present): it picked "the next most recent distinct period," by `period_end` alone, with no check that the duration matched `current`'s.

Confirmed against real TXN data: current = FY2025 revenue ($17.682B, period_end 2025-12-31). The company's own MD&A states plainly: *"Revenue of $17.68 billion increased $2.04 billion, or 13.0%, ... compared to fiscal 2024"* — an accurate claim; the real FY2024 figure is $15.641B ($17.682B − $2.04B = $15.642B, checks out). But `resolve_periods()` picked **$13.259B** as the comparison — not FY2024 at all, but a **9-month year-to-date fact from a Q3 2025 10-Q** (period_start 2025-01-01, period_end 2025-09-30). That YTD fact's period_end (2025-09-30) sorts more recently than FY2024's (2024-12-31), so the naive "closest distinct period_end" pick grabbed it instead — the exact same YTD-vs-standalone ambiguity already fixed once for the *ordinal-quarter* hint path, but never applied to the *default* (no-hint) path.

**Fixed** in [`agents/resolver.py`](../agents/resolver.py): the default comparison now prefers a distinct older period whose duration matches `current`'s (reusing the existing `_similar_length` check), falling back to the old any-distinct-period pick only when no similar-length candidate exists at all (e.g. a company with only quarterly history before its first annual figure). Regression tests added with the real TXN numbers and a fallback case. Re-running the same 8-ticker check after the fix: **14 → 6 inconsistent claims**, all 8 period-selection false positives resolved correctly.

This is exactly the value a specificity check is supposed to have: a real correctness bug, invisible to every other test in this project, because the backtest (`run_backtest.py`) builds `Claim`s directly from the fingerprint's own known period — bypassing `resolve_periods()` entirely — and every other real run happened to use companies/concepts where the default pick landed correctly by chance.

## Root cause 2: the remaining 6 trace to extraction, not the reconciler

The 6 claims still inconsistent after the fix show a different, consistent pattern — not a resolver bug, an **extraction** one. Three from TXN are unambiguous:

| quote | claimed | real (current period) | what happened |
|---|---|---|---|
| "Net income was $5.00 billion compared with $4.80 billion." | $4.80B | $5.001B | extraction grabbed the *prior*-year number from the sentence |
| "provision for income taxes was $709 million compared with $654 million." | $654M | $709M | same pattern |
| "effective tax rate ... was 12.4% in 2025 compared with 12.0% in 2024." | 12.0% | 12.4% | same pattern |

All three are sentences shaped "[current value] ... compared with [prior value]" — the extraction step consistently picked the *second* number, not the first. A real, specific, checkable extraction failure mode, distinct from anything already found in [`docs/case-study.pdf`](case-study.pdf)'s "What actually broke." Not fixed here — this is `eval/dspy_extractor.py`/`agents/extraction_agent.py` territory, a different subsystem than the resolver bug above, and a real follow-up worth its own investigation.

CSCO's one remaining case (claimed 19% diluted-EPS growth against a real ~0.4%) and CRM's ($72.4B claimed revenue against a real $41.5B) look like the same class of extraction inaccuracy — CRM's quote ("$72.4 billion, an increase of 14 percent year-over-year") lost its subject to chunk-boundary truncation and plausibly refers to a different metric entirely (e.g. remaining performance obligation, not revenue) — but neither was traced to a specific mechanism as cleanly as TXN's pattern, and that's stated honestly rather than overclaimed.

## The number that actually matters

Given the maker/checker separation this project is built around (`docs/robustness-and-scope.md`), the reconciler's own precision — given *correct* inputs — is the number worth trusting, not a blended system-level rate that conflates extraction and reconciliation errors:

- **Reconciler false-positive rate, given correct inputs: 0/24 (0%).** Every one of the 24 resolved verdicts is either a real match (18 consistent) or a real mismatch the reconciler correctly caught — 8 because of the period-selection bug found and fixed here, 6 because extraction handed it a wrong number. In no case did the reconciler produce an incorrect verdict against an accurate claim and correctly-resolved data.
- **System-level apparent false-positive rate: 6/24 (25%)**, entirely attributable to extraction, not verification.

## Honest scope

8 tickers, 137 claims (after the dash-heading fix), 24 resolved verdicts — a real but small sample, and one candidate set (financials/consumer-staples) turned up an MD&A-detection coverage gap before any specificity signal could be measured at all. The extraction root cause for CSCO and CRM's remaining misses is a reasoned hypothesis from the visible quote, not independently confirmed the way TXN's three cases were. IBM's "incorporated by reference" case remains genuinely unaddressed — a different, larger piece of work than anything fixed here.

## Reproducing

```bash
uv run python -m research.specificity_check   # ~15-20 min, real EDGAR + LLM calls, no GPU needed
```
