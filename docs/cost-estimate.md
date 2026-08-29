# What would full coverage actually cost?

The README's "why now" argument is directional: SEC's XBRL mandate plus falling LLM inference cost make full-filing coverage "a cost problem now, not a capability one." This turns that into a real dollar figure — measured, not guessed.

[`research/cost_estimate.py`](../research/cost_estimate.py). Verification is free: `tools/reconciler.py` is pure Python arithmetic against already-fetched XBRL facts, no LLM call. Retrieval is free: plain HTTP GETs to EDGAR. **Extraction is the only step with a real dollar cost**, so it's the only one measured.

## Method

Real, cache-disabled extraction calls (a unique nonce appended per chunk, so a prior run's cached response can't silently substitute for a genuine network call) against MSFT's real 10-K MD&A — 215 chunks, the same real number already used elsewhere in this project (the Phase 6 checkpoint/resume exercise). 15 chunks measured directly; dspy's own LM call history gives the actual `prompt_tokens`/`completion_tokens` per call, not an estimate from character counts.

**Measured: 1,537 prompt tokens, 454 completion tokens per extraction call, averaged across 15 real chunks.** The prompt cost is dominated by the fixed instruction/schema overhead (~1,500 tokens), not the chunk text itself (chunks averaged well under 1,000 characters) — meaning cost scales with chunk *count*, not chunk *size*, for filings in this range.

Pricing: **gpt-oss-20b — the exact model this project's `LLM_MODEL` points to** — on OpenAI's direct API, $0.030 / 1M input tokens and $0.130 / 1M output tokens (checked directly, 2026, not assumed from memory). Third-party hosts range roughly $0.02–0.10 per 1M input and $0.10–0.50 per 1M output depending on provider, so this is a mid-to-low estimate for a *comparable commercial rate*, not the cheapest possible one this exact model could be run at.

## The number

| | |
|---|---|
| Cost per extraction call | $0.000105 |
| Cost per full 10-K MD&A (215 chunks) | $0.0226 |
| **Cost for all 500 S&P 500 companies' annual 10-Ks** | **$11.31** |

Eleven dollars a year to run extraction across every S&P 500 company's full annual disclosure — not per company, the entire index. The majority of S&P 500 companies carry calendar (December 31) fiscal years, so a large share of that filing volume — and that $11.31 — concentrates in the January–March 10-K season rather than spreading evenly across the year; this project didn't verify the exact split, so it's noted as a real skew rather than assumed away.

## What this number does and doesn't say

It's the extraction cost only, at this project's current, un-fine-tuned model. It doesn't include:

- **The reasoning-quality gap Phase 7 measured.** The zero-shot model used here has real failure modes (a false-consistent rate the RLVR-trained model cut from 10.8% to 2.0%) — full coverage at this dollar cost doesn't mean full coverage at production-grade accuracy without the training step `docs/phase7-results.md` already ran.
- **The coverage gaps `docs/robustness-and-scope.md` and `docs/specificity-check-results.md` already found** — several real filer types (financials, consumer staples) have MD&A structures this project's parser doesn't yet handle, and roughly 4 in 10 real claims resolve to a checkable concept today. $11.31 buys running the pipeline over every chunk, not verifying every claim a filing makes.
- A stronger commercial model, which would cost more per token but might need fewer retries or produce better extraction quality — a real tradeoff, not evaluated here.

The honest claim this number supports: the *inference cost* of full-index coverage is genuinely negligible at current small-open-model pricing. What full, production-grade coverage would actually require is the accuracy and coverage work already documented elsewhere in this project, not a bigger compute budget.

## Reproducing

```bash
uv run python -m research.cost_estimate   # ~1 min, real LLM calls, no GPU needed
```
