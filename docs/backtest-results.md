# Backtest results — would this have caught a real restatement?

Every other result in this project checks the pipeline in the present tense: does it verify a claim correctly *today*, against whatever EDGAR currently reports? This backtest asks a different, sharper question — has the project's own bitemporal `as_of` machinery ever actually caught something real happening to a real company? SEC 8-K "Item 4.02" disclosures (a company telling investors its own past financial statements can no longer be relied on) supply real ground truth to test that against, using nothing synthetic.

Three stages, each verified before moving to the next.

## Phase A — finding real restatements

[`research/find_restatements.py`](../research/find_restatements.py). SEC's own XBRL company-facts API keeps *both* the original and the corrected value for the same `(concept, period_start, period_end)` when a company amends a filing — a restatement is then just: group a company's facts by that key, and look for groups where an amended filing (`10-K/A`/`10-Q/A`) reports a materially different value than the original. Item 4.02 8-K filings are used only to build the candidate company list, not to determine what was restated — the XBRL diff itself is the ground truth, confirmed by construction, not inferred from prose.

A first pass counted *any* later-filed value as a restatement and found 42,200 "fingerprints" across 655 companies — a number that fell apart on inspection: 82% came from a later regular 10-K/10-Q's comparative column showing a different number, which is routinely just a reclassification or segment realignment, not an error being corrected. Restricting to fingerprints where the later value specifically comes from an amended filing dropped the count to a much more defensible:

- **13,827 fingerprints across 541 companies** (median 14/company — checked directly, not just accepted: the busiest company found, 191 fingerprints, traces back to just 7 distinct amendment filings, each correcting ~20–55 concept/period combinations at once, not 191 independent errors).

Output: [`research/data/restatement_fingerprints.json`](../research/data/restatement_fingerprints.json).

## Phase B — matching fingerprints to real prose claims

[`research/match_prose_claims.py`](../research/match_prose_claims.py). Not every restated XBRL concept gets narrated in prose — a company's MD&A might discuss revenue and margins in words while a balance-sheet line item only ever appears in a table. For a sample of Phase A's fingerprints, this fetches the *original* filing (the one later amended) and runs the project's real extraction agent over its MD&A, keeping claims that match a fingerprint on two signals together:

1. The claim's metric text resolves (via `agents.resolver.METRIC_TO_CONCEPTS`) to the *same* concept the fingerprint restated — not a fuzzy text match.
2. The claim's numeric value is within 5% of the fingerprint's original value — prose commonly rounds ("$1.8 billion" for $1,838,000,000).

**Result: 122 real matches across 16 companies.** Output: [`research/data/phase_b_matches.json`](../research/data/phase_b_matches.json).

**A real reliability bug, found the hard way, and then a real scale-up once it was fixed.** The first full 60-company run stalled for 2h15m+ with no visible progress and, more seriously, no incremental saving — all matches lived only in memory, so what it had already found would have been lost entirely if killed. Root cause was two compounding issues: Python buffers stdout when piped (hiding `print()` progress until process exit), and matches were only written to disk at the very end. Fixed with `flush=True` on every print, a `save_matches()` call after every filing (not just at the end), and a hard per-chunk wall-clock timeout (`EXTRACTION_TIMEOUT_SECONDS=45`, via `ThreadPoolExecutor`) independent of whatever retry/timeout behavior the LLM client does internally. First re-run was deliberately scoped to 2 known-good companies (`TARGET_CIKS`) to confirm the fix worked — 18 matches, completed in under a minute. With the fix confirmed, the full 60-company scan ran for real (moved to an always-on box partway through, since the run outlives a laptop staying open): **409 filings tried, 122 matches found across 16 companies** — Discover Financial Services and Rithm Capital Corp. (the original two) plus AZEK, Seneca Foods, CIM Opportunity Zone Fund, Camber Energy, Astrana Health, Sanmina, Compass Minerals, and seven more.

## Phase C — the backtest itself

[`research/run_backtest.py`](../research/run_backtest.py). For each matched claim, this builds a `Claim` directly from the fingerprint's own known `period_start`/`period_end` and the matched prose's `claimed_value`, then calls the actual `tools.reconciler.reconcile()` — the same function every other result in this project uses — twice:

- Once with the bitemporal `as_of` cutoff set to the claim's own filing date: what did the reconciler know when the claim was made?
- Once with `as_of` set to the restated filing's date: what does the reconciler know now?

No LLM calls in this step — `reconcile()` is pure arithmetic against already-known claim values, so this runs in seconds.

**Deliberately bypasses `agents.resolver.resolve_periods()`** and the higher-level `RealVerificationAgent.verify()`. `resolve_periods()` picks the "most recent reported period" by sorting facts by `period_end` descending — correct for a live claim about "the latest quarter," but wrong here: by the time a restatement happens, the company has always filed later fiscal years, so "most recent period" would silently jump to the wrong fiscal year entirely, defeating the point of checking a *specific* historical period. Each fingerprint already carries its own correct period, so the claim is built directly from data already known to be right.

### Results

| outcome | count | what it means |
|---|---|---|
| **expected flip** | 102 | consistent at filing → inconsistent once the restatement existed — the clean target pattern |
| **already inconsistent at filing** | 16 | the prose's own rounding missed tolerance before any restatement existed — a real catch, for an unrelated reason |
| **unverifiable (both sides)** | 3 | `absolute_change` claims ("decreased $33.4 million") — Phase B's matching doesn't capture the comparison period this claim type needs, so `reconcile()` correctly declines rather than guessing one |
| **restatement too small for tolerance** | 1 | a real restatement existed but was smaller than the prose's own rounding — a genuine miss |

**118 of the 119 checkable matches (99.2%) would have been flagged inconsistent by the time each restatement existed, using only data that existed at each point in time** — 118 of all 122 matches (96.7%), counting the 3 structurally-unverifiable ones against the total rather than excluding them. 102 of those show the full target pattern — the claim looked fine on filing day and only became detectably wrong once the SEC amendment landed, exactly what the bitemporal `as_of` design exists to catch.

A few examples of the clean flip, from [`research/data/backtest_results.json`](../research/data/backtest_results.json), spanning companies well beyond the original two:

- Discover Financial Services, net income, Q3 2023: claimed $683M, consistent as filed (0.0% off) → inconsistent once restated (16.55% off, EDGAR now reports a materially different figure).
- Rithm Capital Corp., total assets, Q1 2024: claimed $42.1B, consistent as filed (0.05% off) → inconsistent once restated (12.17% off).
- Camber Energy, net income, Q2 2023: claimed $1.92M, consistent as filed (0.0% off) → inconsistent once restated (250.53% off — one of the more dramatic corrections in the set).
- AZEK Co, net income, FY2023: claimed $25.3M, consistent as filed (0.05% off) → inconsistent once restated (9.38% off).

The one genuine miss, reported rather than hidden: Discover's FY2021 net income claim ("$5.4 billion") sat within 1% tolerance of *both* the original ($5.449B) and restated ($5.388B) figures, because the 1.12% restatement was smaller than the prose's own rounding — a real limitation of a fixed percentage tolerance, not a bug.

Not every match is independent: some periods were restated more than once (CIM Opportunity Zone Fund's interest expense fact for one quarter has two separate amended values across two different amendments), so the same prose claim correctly appears as more than one fingerprint match. Counted as what it is — a real characteristic of companies that restate repeatedly — not deduplicated to make the headline number look tidier.

## Honest scope

122 matches come from 16 companies out of the 541 with real restatement fingerprints, drawn from the same 60-company candidate pool used throughout (`MAX_COMPANIES` in `research/match_prose_claims.py`, sorted by fingerprint magnitude) — not the full 541. The number to trust as the total real-restatement population is 13,827 fingerprints / 541 companies, counted directly from real SEC data; 122 is what was actually pushed through prose-matching and backtesting from the top 60 of that population by magnitude. No LLM extraction runs on prose that hasn't been checked; no reconciliation runs on a claim this pipeline didn't actually extract and match.

## Reproducing

```bash
uv run python -m research.find_restatements     # ~15 min, hits EDGAR full-text search + company-facts API
uv run python -m research.match_prose_claims     # full 60-company scan by default; real LLM extraction calls, ~10-40 min
uv run python -m research.run_backtest           # seconds — pure arithmetic, no LLM calls
```
