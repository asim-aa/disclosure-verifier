# Phase 6 results — end-to-end integration at scale

Everything before this ran the real Coordinator against exactly one hand-picked
filing (NVDA's FY2026 10-K, in the README/case study). This phase runs it for
real — no mocks — against five companies, and reports what actually happened,
including the two real gaps it surfaced — both now fixed where they could
honestly be fixed, and precisely bounded where they couldn't, not just
diagnosed. Script: [`phase6/run_integration.py`](../phase6/run_integration.py).

## Setup

Real `RetrievalAgent` (live EDGAR), real `ExtractionAgent` (live DSPy + LLM
extraction), real `VerificationAgent` (the actual Reconciler) — the same
`Coordinator` used everywhere else, run against:

- **AAPL, MSFT, NVDA** — the three companies this project was built and tested against throughout
- **GOOGL, AMZN** — deliberately new, to check whether the pipeline generalizes or was quietly tuned to the three it had already seen

Each ticker capped at a `Budget(max_chunks=25, max_extraction_calls=20, max_seconds=240)` — real filings can run 100+ MD&A chunks, each an LLM call; this bounds cost/time per company rather than letting one large filing dominate the batch.

## Run 1 — baseline, before any fix

```
n_tickers: 5, n_succeeded: 5, n_failed: 0
total_claims_verified: 16
by_verdict: {"unverifiable": 16}
total_tool_calls: 86, total_reasoning_steps: 76
total_elapsed_seconds: 103
```

**Zero crashes across all five companies.** The harness (budget enforcement, per-ticker error isolation) held up exactly as designed — no exception propagated out of a single run, every ticker returned a `Report` regardless of what it found.

**But every single verdict was "unverifiable," and two tickers produced zero claims at all.** That's not a clean result, and reporting it as one would be exactly the kind of thing this project has spent its whole evaluation methodology trying not to do. Two distinct, well-diagnosed causes, addressed in the order they were found.

## Fix 1 — MD&A heading detection didn't generalize past the three companies it was built against

`GOOGL` and `AMZN` produced zero claims — not because their MD&A was checked and found empty, but because `tools/mdna_parser.py` couldn't locate the section at all (`MdnaNotFoundError`, caught and skipped per-filing rather than crashing the run — see `agents/retrieval_agent.py`).

The parser's heading match assumed the real "Item 7. Management's Discussion..." heading always appears as one combined text node, while a table-of-contents entry splits "Item 7." from the title across separate nodes — matching only the combined form was how it told the real section from the ToC. That assumption held for AAPL/MSFT/NVDA's filing HTML. It did **not** hold for GOOGL or AMZN: both render the real body heading as an isolated `"Item 7."` node, with the title text following separately — structurally identical to what the parser was treating as a ToC-only pattern. Confirmed directly by pulling the raw text around every "Item 7" occurrence in both filings, not assumed.

Three companies were enough to make the heuristic look solid and enough to make it wrong — a real, previously-invisible generalization gap. Fixed in `tools/mdna_parser.py`: instead of requiring the item number and title to share one text node, every window of up to 3 consecutive lines is treated as a *candidate* heading, and — since that alone would now also match every ToC entry — each candidate is paired with the nearest matching end-of-section marker after it, and the (start, end) pair with the **largest gap** wins. A real section runs for hundreds of lines of prose before the next Item heading; a ToC entry or an incidental cross-reference is only ever a few lines from whatever comes next. That distinction generalizes in a way "which text node" never did.

Writing the fix surfaced a second, subtler bug in the first draft of it: a candidate window could "absorb" a heading that actually started on a later line, silently stealing one line of real body content at the section boundary — caught by a synthetic regression test (`tests/test_mdna_parser.py`) checking that every body paragraph survives extraction, not just that extraction succeeds at all. Fixed by requiring a candidate's match to actually begin within the window's first line, not merely appear somewhere in the joined text.

### Run 2 — after the parser fix

```
n_tickers: 5, n_succeeded: 5, n_failed: 0
total_claims_verified: 18
by_verdict: {"unverifiable": 18}
total_tool_calls: 128, total_reasoning_steps: 119
total_elapsed_seconds: 106
```

GOOGL and AMZN both now retrieve real MD&A chunks (AMZN: 2 claims found; GOOGL: chunks retrieved successfully but 0 checkable claims within the budget cap — see the caveat below). The fix is confirmed working directly, not just by an improved aggregate number: re-extracting all 5 companies' real filings after the fix recovered 1–2 previously-dropped final lines in AAPL, NVDA, and GOOGL's sections too (the boundary bug above affected everyone, not only the two companies that failed outright).

Every verdict was *still* "unverifiable" here — expected, and it strengthens the second finding rather than undercutting it: with the parsing gap closed, all 5 companies could now be checked against the resolver's coverage, not just the 3 that could even be tested before, and the same narrow-scope result held across the full set. That's the subject of the second fix, below.

## Fix 2 — the resolver's scope is narrower than real MD&A prose, by a lot — partly fixed, partly a genuine data-source limit

Pulling NVDA's actual unresolved claims:

```
H20 charge                              | absolute | Could not resolve metric 'H20 charge'...
H20 revenue                             | absolute | Could not resolve metric 'H20 revenue'...
tariff rate for H200 products           | absolute | Could not resolve metric 'tariff rate for H200 products'...
Private company and infrastructure fund investments | absolute | Could not resolve...
land, power, and shell guarantees       | absolute | Could not resolve...
```

And MSFT's:

```
Microsoft Cloud revenue | absolute/growth_pct | unverifiable
Commercial remaining performance obligation | absolute/growth_pct | unverifiable
Microsoft 365 Commercial cloud revenue | growth_pct | unverifiable
LinkedIn revenue | growth_pct | unverifiable
Azure and other cloud services revenue | growth_pct | unverifiable
XBOX content and services revenue | growth_pct | unverifiable
```

Every one of these is a genuine, timely, company- or segment-specific line item (H20/H200 are NVDA chip products caught up in export-control tariffs; Azure/LinkedIn/XBOX are Microsoft product segments) — not something `agents/resolver.py`'s fixed 14-concept top-level GAAP mapping was ever meant to cover, and the system is doing exactly what it's supposed to: declining rather than guessing (see `docs/robustness-and-scope.md`'s maker/checker and "never guess" principles). What's new here isn't a bug — it's the discovery that **this is the common case, not the edge case**, once you look at more than one hand-picked filing. Real MD&A prose near the start of a 10-K is dominated by segment/product/event-specific figures; the resolver's safe-but-narrow scope means most of it is currently unverifiable by design.

**Investigated rather than assumed fixable.** Checking the raw SEC company-facts data directly (not the taxonomy, the actual JSON this project's Filing Retriever consumes) confirmed why segment figures specifically can't resolve here: every data point is a flat `{start, end, val, accn, fy, fp, form, filed, frame}` record — there is no segment/member/axis field anywhere in this data source. Azure revenue, LinkedIn revenue, and Data Center revenue simply aren't present at any granularity in the data this pipeline reads. Reaching them would mean parsing a filing's raw XBRL instance document instead — a genuinely different, much larger effort, not attempted here.

**But not everything in the unresolved list was actually a segment metric — some of it was just a missing dictionary entry.** "Commercial remaining performance obligation" turned out to map directly to `RevenueRemainingPerformanceObligation`, a real, standard, non-dimensional us-gaap concept — confirmed present with plausible values ($375B→$684B, growing, matching Microsoft's known real backlog disclosures) in cached company-facts data before adding it, not guessed from the taxonomy's existence alone. 17 more concepts (net income, EPS, total assets, R&D expense, capital expenditures, interest expense, and others) were verified present across AAPL/MSFT/NVDA the same way and added to `METRIC_TO_CONCEPTS`.

### Run 3 — after the dictionary expansion

```
n_tickers: 5, n_succeeded: 5, n_failed: 0
total_claims_verified: 18
by_verdict: {"unverifiable": 16, "consistent": 1, "inconsistent": 1}
```

MSFT's `RevenueRemainingPerformanceObligation` claims now resolve to real verdicts:

```
Commercial remaining performance obligation (absolute, claimed $678.0B)
  -> consistent: EDGAR reports $684.0B for the period ending 2026-06-30 (0.88% off)

Commercial remaining performance obligation (growth_pct, claimed +84.0%)
  -> inconsistent: EDGAR data implies +8.06% (633.0B -> 684.0B), a 75.94-point difference
```

The `inconsistent` result is itself informative rather than a false alarm: `resolve_periods`'s own documented limitation (it can't tell "sequentially" from "a year ago" when picking a comparison period — see its docstring) means this specific growth claim may be comparing against the wrong prior period, not that the claim itself is necessarily wrong. Flagged here rather than presented as a clean catch.

Every other previously-unresolved claim across all 5 companies (Azure, LinkedIn, XBOX, H20, land/power/shell guarantees, infrastructure fund investments) is still unverifiable — correctly, and for the confirmed data-source reason above, not a dictionary gap. Expanding `METRIC_TO_CONCEPTS` further will keep finding a few more of these; it will never reach the genuinely segment-level ones without a different data source.

## One more honest caveat, present in both Run 2 and Run 3

AAPL and GOOGL returned **zero claims at all** within the per-ticker budget (`max_chunks=25`) — not zero *resolvable* claims, zero claims *extracted*. That means the first ~25 MD&A paragraphs of these two companies' filings, in document order, happened to be more qualitative/narrative than numeric. This is a budget-vs-coverage tradeoff, not a bug: a tighter budget bounds cost and time (the whole reason it exists), at the cost of not reaching the numbers-dense parts of a filing that starts with scene-setting prose. Stated explicitly rather than left to look like AAPL and GOOGL simply "had nothing to say" — they weren't read far enough to know. Neither fix in this document changes which chunks get read, only what happens once they are — this caveat is independent of both.

## Checkpoint/resume, exercised against a real filing

`run_integration.py` always calls `coordinator.run(..., resume=False)` — every run above proves the *budget* half of the harness claim (every ticker above hit `[PARTIAL - budget hit]`), but none of them ever exercised *resume*, only the mock-agent scenario tests in `tests/test_coordinator_checkpoint.py` did. [`phase6/exercise_checkpoint_resume.py`](../phase6/exercise_checkpoint_resume.py) closes that gap directly against MSFT's real 215-chunk filing:

```
--- Step 1: tiny-budget run (max_chunks=3), resume=False, MSFT ---
  partial=True  reason=max_chunks (3) reached
  checkpoint on disk: 3 chunks processed, 0 claims saved

--- Step 2: full-budget run, resume=True (default), MSFT ---
  resumed-from-checkpoint trace events found: 1
    TraceEvent(action='resumed_from_checkpoint', detail='3 of 215 chunks already processed')
  final claims: 11 (first run had 0)
  re-checkpointed after resume: 23 chunks processed (was 3 before resume)
```

Real, on-disk checkpoint written after a real budget stop; real resume that loaded it, correctly skipped re-extracting the 3 already-processed chunks (no duplicate LLM calls), and continued forward to chunk 23, picking up 11 real claims along the way. The second call hit its own budget before finishing all 215 chunks — expected, not a bug: a real 10-K this size needs several budgeted hops to finish, which is exactly the scenario checkpointing exists for. Both halves of the harness claim — budget enforcement and checkpoint/resume — are now demonstrated against real data, not just mock scenarios.

## Why this is the right Phase 6 outcome, not a failed one

The harness claim — bounded cost, no crashes, graceful per-filing degradation — held up under real, previously-untested conditions, across all three runs. The coverage claim was never made this precisely before: at scale, on real prose, the resolver and the MD&A parser both had concrete, now-diagnosed edges. The parser gap is fixed and re-verified against live data. The resolver gap turned out to be two different things wearing one label — a handful of genuinely missing dictionary entries (fixed, verified) and a real data-source limit (confirmed with evidence, not fixed, correctly still out of scope) — and telling those apart is itself the finding. That's what "integration at scale" is for.

## Reproducing

```bash
python -m phase6.run_integration
```

Requires live EDGAR access and a reachable `LLM_BASE_URL` (see `.env`) — not part of the offline test suite. Full per-ticker output saved to `phase6/results/run.json`.
