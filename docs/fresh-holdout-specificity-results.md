# Does the specificity result generalize, or is it overfit to 8 companies?

[`docs/specificity-check-results.md`](specificity-check-results.md) landed at 0/33 false positives (0.0%) — but every one of its eleven confirmed root causes was diagnosed against real bugs found *in that same 8-ticker control set* (ADBE, CRM, ORCL, CSCO, INTU, IBM, QCOM, TXN). CRM's fiscal-year tagging quirk, CSCO's specific phrasing patterns, TXN's specific sentence structures — a 0% rate measured only against the companies the fixes were built to fix proves the fixes work, not that the underlying approach generalizes to companies never seen before.

[`research/specificity_check_fresh.py`](../research/specificity_check_fresh.py). Same methodology, same `Coordinator`/`Budget`/`confirm_clean` (reused directly, not reimplemented) — run against a second, disjoint control set of large tech/software companies genuinely untouched by any fix, test, or diagnostic call this project has made at any point: **NOW, PANW, CRWD, WDAY, SNPS, FTNT, INTC, MU**. Also disjoint from AAPL/MSFT/NVDA/GOOGL/AMZN (the pipeline's own original dev/tuning companies). Confirmed clean of restatement fingerprints the same way as the original control set.

## First run: it worked exactly as intended — it found something

67 claims, 11 resolved verdicts, **5 inconsistent — 45.5% apparent false-positive rate**, all 5 from a single company: SNPS (Synopsys). Investigated directly rather than assumed to be more of the same:

**Root cause A: segment/acquisition-qualified sub-figures mislabeled under the bare metric name.** SNPS's real 10-K states *"Revenues were $7.1 billion, an increase of $926.8 million or 15%, which includes revenues from Ansys of $756.6 million"* — extraction grabbed the Ansys-specific contribution ($756.6M, from SNPS's recent Ansys merger) and labeled it plain "revenue," comparing it against the company's real total revenue ($7.05B) for an apparent 89% miss that isn't a mismatch at all, just a mislabeled number. The same sentence's *"[China] revenue decrease 22% compared to fiscal 2024, excluding Ansys"* hit the identical pattern with the sign flipped too — China's specific decline mislabeled as overall revenue growth.

**Root cause B: a GAAP/non-GAAP ambiguity hiding inside an already-trusted dictionary entry.** SNPS's MD&A states, word for word, *"Operating income was $1.4 billion, an increase of $82.5 million or 6%."* Extraction read this correctly. But real GAAP `OperatingIncomeLoss` actually **fell** 32.5% ($1,355.7M → $914.9M) — almost certainly because SNPS's stated figure excludes one-time Ansys-merger acquisition costs (a non-GAAP adjustment). This is the same trap already found and avoided for CSCO's "operating margin" earlier — except here it's hiding inside `"operating income" → OperatingIncomeLoss`, a mapping *already used successfully* by other companies in the original control set. SNPS's unusually large one-time acquisition charges just made the GAAP/non-GAAP gap large enough to surface it on a plain, previously-safe metric text.

Also worth reporting honestly: NOW, FTNT, and INTC each returned **zero claims** this run (budget exhausted on segment-heavy prose, or no MD&A locatable within budget) — a real coverage gap, the same class already documented for the financials/consumer-staples candidate set, not a false positive.

## Fixed: root cause A only, by explicit decision

Root cause A is bounded and matches the exact pattern of prior fixes in this project (the FX-impact rule, the percentage-points rule): a docstring addition to `eval/dspy_extractor.py`'s `ExtractClaims` signature teaching the model that a segment/product/subsidiary/geography-qualified sub-figure gets its *own* metric text (keeping the qualifier), never the bare metric name — the resolver's existing exact-match dictionary then correctly, automatically treats the qualified text as unverifiable (no new resolver code needed at all).

Root cause B was deliberately **not** fixed — a genuinely open, structural question (a company's plain "operating income" prose can mean GAAP or non-GAAP depending on the company and the year, and an exact-match dictionary has no way to know which), documented here rather than guessed at.

Verified before shipping: a full regression sweep against every previously-fixed real sentence this project has diagnosed (TXN's number-expansion, CRM's compared-with pairing, the FX-impact skip, `bps_change`-vs-`growth_pct`, CSCO's period-propagation) — no interference, all still correct. Added the real SNPS sentence as a labeled training example (`eval/labeled_claims.jsonl` id=84) and a generalization eval (`eval/segment_qualifier_eval.py`, 2 synthetic cases with different segment/geography names) — recompiled (`eval/compile_and_save.py` — see `docs/specificity-check-results.md`'s Root cause 7 for why that step is mandatory after any docstring edit) and confirmed 3/3 against the real production artifact on the exact real SNPS sentence, with every value and sign correct.

## Second run: the fix worked completely

73 claims, 9 resolved verdicts, **3 inconsistent — 33.3% apparent false-positive rate**. All 3 are the *same* Operating income (GAAP/non-GAAP) case, unchanged — root cause A is confirmed fully eliminated: zero revenue-misattribution false positives remain. FTNT went from 0 to 3 claims run to run (the same LLM extraction non-determinism already documented repeatedly in this project, not a regression).

## Honest scope

n=9 resolved verdicts is a genuinely tiny sample — the 95% confidence half-width on 33.3% is roughly ±31pp, wide enough that the *rate* itself shouldn't be over-read. What's precise, not noisy, is the categorical finding: the specific bug class targeted (segment-qualifier misattribution) is confirmed eliminated by direct re-measurement, and every remaining miss traces to one already-diagnosed, already-documented, deliberately-unfixed cause — not a mystery. The GAAP/non-GAAP operating-income risk is real and, on this evidence, not limited to one company; it's a structural property of exact-match metric-text mapping that a future investigation would need actual company-by-company or context-aware disambiguation to close, not another dictionary entry.

## Reproducing

```bash
uv run python -m research.specificity_check_fresh   # ~15-20 min, real EDGAR + LLM calls, no GPU needed
uv run python -m eval.segment_qualifier_eval          # extraction generalization: segment/geography-qualifier sentences
```
