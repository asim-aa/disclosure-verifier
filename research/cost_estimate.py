"""Turns the README's "why now" cost argument into a real number: what would it
actually cost, in dollars, to run this pipeline's extraction step across every
S&P 500 company's annual 10-K?

Measured, not guessed. Runs real (cache-disabled, so no stale/free response
skews the count) extraction calls against a real company's real MD&A chunks
and reads the actual prompt/completion token counts off dspy's LM call
history - not an assumed average.

Verification (reconciliation) is free: `tools/reconciler.py` is pure Python
arithmetic against already-fetched XBRL facts, no LLM call. Retrieval is free:
plain HTTP GETs to EDGAR. Extraction is the only step with a real dollar cost,
so it's the only one measured here.

Pricing: gpt-oss-20b (the exact model this project's own LLM_MODEL points to)
on OpenAI's direct API, $0.030 / 1M input tokens and $0.130 / 1M output
tokens - the standard/reference tier; third-party hosts range from ~$0.02-0.10
per 1M input and ~$0.10-0.50 per 1M output depending on provider, so this is a
mid-to-low estimate for a *comparable* commercial rate, not the cheapest
possible one. Checked directly (2026), not assumed from memory.

Chunk count: 215 chunks for MSFT's real 10-K MD&A - the same real number
already used elsewhere in this project (docs/phase6-results.md's checkpoint/
resume exercise), not a separate guess.

Run: python -m research.cost_estimate
"""

import time

import dspy

from agents.extraction_agent import RealExtractionAgent
from agents.retrieval_agent import RealRetrievalAgent
from eval.llm_config import get_lm

TICKER = "MSFT"
SAMPLE_SIZE = 15  # real chunks measured directly; enough to average out per-chunk variance

PRICE_PER_1M_INPUT_USD = 0.030
PRICE_PER_1M_OUTPUT_USD = 0.130

N_SP500_COMPANIES = 500


def measure_real_token_usage(chunks) -> tuple[float, float]:
    """Runs real extraction calls (cache disabled, unique nonce per chunk so a
    prior run's cached response can't silently skip the real network call) and
    returns (avg_prompt_tokens, avg_completion_tokens) from dspy's own recorded
    usage - not computed from character counts."""
    lm = get_lm()
    lm.cache = False
    dspy.configure(lm=lm)
    agent = RealExtractionAgent()

    prompt_tokens, completion_tokens = [], []
    sample = [c for c in chunks if len(c.text) > 400][:SAMPLE_SIZE]
    for i, chunk in enumerate(sample):
        text = chunk.text + f"\n<!-- nonce:{time.time()}:{i} -->"  # forces a real, uncached call
        agent.extract(text)
        usage = dspy.settings.lm.history[-1].get("usage") or {}
        if usage.get("prompt_tokens"):
            prompt_tokens.append(usage["prompt_tokens"])
            completion_tokens.append(usage["completion_tokens"])
        print(f"  [{i + 1}/{len(sample)}] {len(chunk.text)} chars -> "
              f"{usage.get('prompt_tokens')} prompt / {usage.get('completion_tokens')} completion tokens",
              flush=True)

    return sum(prompt_tokens) / len(prompt_tokens), sum(completion_tokens) / len(completion_tokens)


def main() -> None:
    retrieval = RealRetrievalAgent()
    chunks = retrieval.get_mdna_chunks(TICKER, form_type="10-K", limit=1)
    n_chunks = len(chunks)
    print(f"{TICKER}'s real 10-K MD&A: {n_chunks} chunks (the actual extraction-call count "
          f"a full, un-budget-capped run would make)\n", flush=True)

    print(f"Measuring real token usage over {SAMPLE_SIZE} real chunks (cache disabled)...", flush=True)
    avg_prompt, avg_completion = measure_real_token_usage(chunks)
    print(f"\nMeasured average: {avg_prompt:.0f} prompt tokens, {avg_completion:.0f} completion "
          f"tokens per extraction call", flush=True)

    cost_per_chunk = (avg_prompt / 1_000_000 * PRICE_PER_1M_INPUT_USD
                       + avg_completion / 1_000_000 * PRICE_PER_1M_OUTPUT_USD)
    cost_per_filing = cost_per_chunk * n_chunks
    cost_all_sp500 = cost_per_filing * N_SP500_COMPANIES

    print(f"\nCost per extraction call:  ${cost_per_chunk:.6f}")
    print(f"Cost per full 10-K MD&A ({n_chunks} chunks): ${cost_per_filing:.4f}")
    print(f"Cost for all {N_SP500_COMPANIES} S&P 500 companies' annual 10-Ks: ${cost_all_sp500:.2f}")


if __name__ == "__main__":
    main()
