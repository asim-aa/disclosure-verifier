# Naive-LLM-judge baseline: does the deterministic pipeline actually earn its keep?

Every other result in this project measures whether *this* pipeline works. This measures whether the *architecture choice* did — structured extraction, exact concept resolution, then deterministic bitemporal arithmetic (`tools/reconciler.py`), instead of the obvious alternative: hand a general LLM the claim and the raw numbers and let it decide.

[`research/naive_llm_baseline.py`](../research/naive_llm_baseline.py). Two test sets, both reused from already-verified ground truth — nothing hand-labeled fresh for this comparison. Run against this project's own LLM endpoint (the same model used everywhere else here), not a stronger commercial model — isolates "does structure help, holding model capability constant" as the cleanest version of the comparison.

## Test set 1 — the reconciler audit's 15 adversarial cases

[`eval/reconciler_audit.py`](../eval/reconciler_audit.py)'s cases, already the project's own tool for stress-testing the *deterministic* reconciler (100% correct on all 15, by construction — that's what "deterministic" means). Deliberately isolates arithmetic-reasoning from concept-resolution: each case's `Claim.metric` is already the exact XBRL concept string, so there's no metric-to-concept matching for the naive baseline to get right or wrong, only the sign-flip/magnitude-confusion/tolerance-boundary probes the audit was built to catch.

**Naive LLM baseline: 11/15 (73.3%)**, against the deterministic reconciler's 15/15 (100%) on the identical cases. The 4 misses:

| case | expected | naive got |
|---|---|---|
| magnitude confusion (claim off by 1,000,000×) | inconsistent | **consistent** |
| sign flip on `absolute_change` | inconsistent | **consistent** |
| real `bps_change` case, correct math | consistent | inconsistent |
| tolerance boundary, exactly at the 1% edge | consistent | inconsistent |

The magnitude-confusion miss is the one worth sitting with: a claim stated in millions against facts reported in raw dollars — off by six orders of magnitude — read as "consistent" to the naive baseline. This is exactly the dangerous failure mode the whole project is built around (a false "consistent" actively endorses a wrong number, worse than a false "inconsistent"), and it's the deterministic reconciler's `_reconcile_absolute()` doing plain division that never gets it wrong, not a prompting trick.

A real harness bug was caught and fixed while building this: the first pass omitted the denominator concept's facts for `bps_change` cases (the ratio's numerator concept was included, the denominator wasn't), so the naive baseline correctly said "unverifiable" on data it was never shown — not a fair test. Fixed by including both `metric` and `denominator_metric`'s facts; the corrected re-test is what's reported above.

## Test set 2 — the 122 real backtest matches, bitemporal

The harder, more realistic test. For each of the 122 real matches from [`docs/backtest-results.md`](backtest-results.md), the naive baseline is handed the *same* already-correctly-resolved concept name `reconcile()` would use, but the **raw, unfiltered fact history** for that concept — every filed value across every amendment, not the one bitemporally-correct value `_find_fact()` would have picked. It's told the claim's own filing date and asked to judge consistency "as of" that date, then asked again "as of" the restated filing's date — exactly mirroring the two `reconcile()` calls `run_backtest.py` makes, but via prompting instead of an `as_of` cutoff in code.

**Naive baseline reproduced the correct pattern in only 50 of the 102 cases (49%) where the deterministic pipeline got it right** — barely better than a coin flip on the exact task the bitemporal design exists to do.

The failure mode is precise and consistent, not scattered: of the 100 real cases where the claim was genuinely consistent as of its own original filing date, the naive baseline incorrectly called **49 of them "inconsistent"** — a 49% false-positive rate at exactly the point bitemporal correctness matters. Full breakdown:

| naive verdict pattern | count | what it means |
|---|---|---|
| consistent → inconsistent (correct flip) | 54 | reproduced the right pattern |
| inconsistent → inconsistent | 67 | called it wrong from the start — but only 16 of these are cases where it was *actually* already wrong; the rest are false positives |
| consistent → consistent | 1 | missed the restatement entirely |

The naive baseline is systematically bleeding later information backward — seeing a company's eventual restated value in the raw fact list and judging the *original* claim against it, exactly the bug this project's own deterministic code had and fixed once already (`tools/reconciler.py`'s `as_of` threading, see `docs/robustness-and-scope.md`'s bitemporal-correctness section). Telling a general LLM "only use data filed on or before this date" in a prompt does not reliably make it do that; writing the cutoff into the fact-lookup code does.

## The honest comparison

| | deterministic reconciler | naive LLM baseline |
|---|---|---|
| audit set (15 adversarial cases) | 15/15 (100%) | 11/15 (73.3%) |
| backtest set, correct flip pattern | 102/122 (83.6%) | 50/122 (41.0%) |
| backtest set, false-positive rate at the original filing date | 0% (by construction) | 49% (49/100) |

This isn't a claim that LLMs can't reason about numbers — the naive baseline gets plenty right, and this project's own Phase 7 result shows a *fine-tuned* small model closing much of the arithmetic gap. It's a specific, measured answer to a specific question: given the same data, does skipping the deterministic scaffolding cost real accuracy? Yes, and the cost concentrates exactly where you'd predict — magnitude/unit errors and bitemporal reasoning, the two places this project's own development history already found real bugs in more careful, code-based attempts at the same problem.

## Honest scope

Run against this project's own self-hosted endpoint (`gpt-oss-20b`), a fixed model held constant to isolate "does structure help" from "is this specific model good at this." A stronger commercial model (GPT-4-class or Claude) would very plausibly narrow this gap, especially on the audit set's arithmetic — that's a real, worthwhile follow-up, not run here. One prompt attempt per case, no retries or self-consistency voting, which could also change the naive baseline's numbers somewhat. The comparison that's robust regardless: the *pattern* of failure (magnitude/unit errors, bitemporal information leakage) is a structural property of prompting an LLM with raw, unfiltered data, not an artifact of this one model or one prompt.

## Reproducing

```bash
uv run python -m research.naive_llm_baseline   # ~15-25 min, real LLM calls, no GPU needed
```
