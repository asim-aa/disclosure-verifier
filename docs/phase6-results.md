# Phase 6 results — end-to-end integration at scale

Everything before this ran the real Coordinator against exactly one hand-picked
filing (NVDA's FY2026 10-K, in the README/case study). This phase runs it for
real — no mocks — against five companies, and reports what actually happened,
including the two real gaps it surfaced. Script: [`phase6/run_integration.py`](../phase6/run_integration.py).

## Setup

Real `RetrievalAgent` (live EDGAR), real `ExtractionAgent` (live DSPy + LLM
extraction), real `VerificationAgent` (the actual Reconciler) — the same
`Coordinator` used everywhere else, run against:

- **AAPL, MSFT, NVDA** — the three companies this project was built and tested against throughout
- **GOOGL, AMZN** — deliberately new, to check whether the pipeline generalizes or was quietly tuned to the three it had already seen

Each ticker capped at a `Budget(max_chunks=25, max_extraction_calls=20, max_seconds=240)` — real filings can run 100+ MD&A chunks, each an LLM call; this bounds cost/time per company rather than letting one large filing dominate the batch.

## What happened

```
n_tickers: 5, n_succeeded: 5, n_failed: 0
total_claims_verified: 16
by_verdict: {"unverifiable": 16}
total_tool_calls: 86, total_reasoning_steps: 76
total_elapsed_seconds: 103
```

**Zero crashes across all five companies.** The harness (budget enforcement, per-ticker error isolation) held up exactly as designed — no exception propagated out of a single run, every ticker returned a `Report` regardless of what it found.

**But every single verdict was "unverifiable," and two tickers produced zero claims at all.** That's not a clean result, and reporting it as one would be exactly the kind of thing this project has spent its whole evaluation methodology trying not to do. Two distinct, well-diagnosed causes:

### 1. The resolver's scope is narrower than real MD&A prose, by a lot

Pulling NVDA's actual unresolved claims:

```
H20 charge                              | absolute | Could not resolve metric 'H20 charge'...
H20 revenue                             | absolute | Could not resolve metric 'H20 revenue'...
tariff rate for H200 products           | absolute | Could not resolve metric 'tariff rate for H200 products'...
Private company and infrastructure fund investments | absolute | Could not resolve...
land, power, and shell guarantees       | absolute | Could not resolve...
```

Every one of these is a genuine, timely, company-specific line item (H20/H200 are NVDA chip products caught up in export-control tariffs) — not something `agents/resolver.py`'s fixed 14-concept top-level GAAP mapping was ever meant to cover, and the system is doing exactly what it's supposed to: declining rather than guessing (see `docs/robustness-and-scope.md`'s maker/checker and "never guess" principles). What's new here isn't a bug — it's the discovery that **this is the common case, not the edge case**, once you look at more than one hand-picked filing. Real MD&A prose near the start of a 10-K is dominated by segment/product/event-specific figures; the resolver's safe-but-narrow scope means most of it is currently unverifiable by design. The system is correct and honest; it just isn't broadly *useful* yet at this coverage level. Expanding `METRIC_TO_CONCEPTS` (or handling dimensional/segment XBRL, which the Filing Retriever doesn't currently parse — see `agents/resolver.py`'s own module docstring) is the concrete next step this finding points to, not a fix to the verdict logic itself.

### 2. MD&A heading detection doesn't generalize past the three companies it was built against

`GOOGL` and `AMZN` produced zero claims — not because their MD&A was checked and found empty, but because `tools/mdna_parser.py` couldn't locate the section at all (`MdnaNotFoundError`, caught and skipped per-filing rather than crashing the run — see `agents/retrieval_agent.py`).

The parser's heading match assumes the real "Item 7. Management's Discussion..." heading always appears as one combined text node, while a table-of-contents entry splits "Item 7." from the title across separate nodes — matching only the combined form is how it tells the real section from the TOC. That assumption holds for AAPL/MSFT/NVDA's filing HTML. It does **not** hold for GOOGL or AMZN: both render the real body heading as an isolated `"Item 7."` node, with the title text following separately — structurally identical to what the parser was treating as a TOC-only pattern. Confirmed directly by pulling the raw text around every "Item 7" occurrence in both filings, not assumed.

This is a real, previously-invisible generalization gap: three companies were enough to make the heuristic look solid and enough to make it wrong. A proper fix needs a sturdier way to tell a body heading from a TOC entry than "combined vs. split text nodes" (e.g. checking whether the match sits inside a `<table>`/anchor-link TOC container, rather than inferring it from node boundaries) — flagged here with the exact root cause rather than patched with a narrower regex that would just move the same fragility to the next untested filer.

## Why this is the right Phase 6 outcome, not a failed one

The harness claim — bounded cost, no crashes, graceful per-filing degradation — held up under real, previously-untested conditions. The coverage claim was never made this precisely before: at scale, on real prose, the resolver and the MD&A parser both have concrete, now-diagnosed edges, not vague ones. That's what "integration at scale" is for — surfacing exactly this, honestly, rather than declaring victory off one filing that happened to work.

## Reproducing

```bash
python -m phase6.run_integration
```

Requires live EDGAR access and a reachable `LLM_BASE_URL` (see `.env`) — not part of the offline test suite. Full per-ticker output saved to `phase6/results/run.json`.
