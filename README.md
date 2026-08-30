# The Disclosure Verifier

[![CI](https://github.com/asim-aa/disclosure-verifier/actions/workflows/ci.yml/badge.svg)](https://github.com/asim-aa/disclosure-verifier/actions/workflows/ci.yml)

**An agentic system that checks whether a company's own words match its own numbers.**

Public companies make thousands of quantitative claims a year in MD&A prose — *"revenue increased 27% to $214.4 billion."* SEC EDGAR also publishes the exact structured data (XBRL) those claims should be derived from. Almost nobody actually cross-checks the two. This project retrieves both, extracts discrete claims from the prose with an optimized LLM pipeline, and reconciles each one against the real reported numbers — with a citation back to the exact filing.

```
"Revenue for fiscal year 2026 was $215.9 billion, up 65% from a year ago."
   -> extracted:  {metric: Revenue, value: 215_900_000_000, type: absolute}
                  {metric: Revenue, value: 65.0, type: growth_pct}
   -> reconciled:  consistent   (EDGAR: $215,938,000,000 — 0.02% off)
                   inconsistent (EDGAR implies 46.09% growth, not 65%)
```

Built as the capstone for the SupportVectors AI Agents Bootcamp, against four required pillars: real MCP tools, a DSPy-optimized prompt with a measured baseline delta, a justified multi-agent architecture, and RLVR/GRPO fine-tuning with a noise-floor-checked before/after result.

**Why now.** SEC's XBRL mandate (phased in 2009–2011, effectively universal today) means the structured ground truth for every claim in this project's scope has always been sitting a few API calls away — cross-checking prose against it by hand just never scaled to thousands of claims a year across thousands of filers. What's changed recently is the cost side: extracting and reasoning through a full filing's worth of claims used to mean either analyst hours or an LLM bill that didn't pencil out at that volume. This project's own Phase 7 result is a small data point for that shift — a 7B-class open model, LoRA fine-tuned on a single consumer GPU, learns the reconciliation-reasoning step well enough (false-consistent rate 0.108→0.020, real and noise-floor-clearing) to make full-filing coverage a cost problem now, not a capability one. Measured directly, not estimated: real, cache-disabled extraction calls against MSFT's real 215-chunk 10-K MD&A average 1,537 prompt / 454 completion tokens each; at gpt-oss-20b's published API pricing, that's **$11.31 to run extraction across every S&P 500 company's full annual 10-K** — the whole index, not one company. Full methodology: [`docs/cost-estimate.md`](docs/cost-estimate.md).

📄 **[Read the full case study (PDF)](docs/case-study.pdf)** — results, a live example, and the real bugs found along the way.

📒 **[The Full Ledger (PDF)](docs/field-guide.pdf)** — a complete reference: every required pillar mapped to where it's implemented, every phase (0–7) explained, every post-phase hardening pass, and the load-bearing concepts and likely questions behind all of it.

---

## Architecture

```mermaid
flowchart LR
    subgraph Retrieval
        A["Filing Retriever"] -->|"filings + XBRL facts"| D
        B["MD and A Extractor"] -->|"cited prose chunks"| D
    end
    D["Coordinator"] --> E["Extraction Agent"]
    E -->|"DSPy ChainOfThought, optimized"| F["ExtractedClaim"]
    F --> G["Verification Agent"]
    G -->|"resolver: text to XBRL concept"| H["Numerical Reconciler"]
    H -->|"consistent / inconsistent / unverifiable"| I["Cited Report"]
    D -. "budget + checkpoint" .-> D
```

Three independent tools (Pillar 1) wired into a hierarchical pipeline (Pillar 3), not a single autonomous loop — deliberately. The *sequence* here never needs an LLM's judgment (you always retrieve before extracting, always extract before verifying); what needs deciding is narrower — skip a chunk with no claims, flag a claim that can't be resolved — and that's ordinary code, not a reasoning task. The one stage that genuinely needs the model (claim extraction) is the one DSPy-optimized signature in the system, run as `dspy.ChainOfThought` so the model reasons through which spans are checkable claims before committing to structured output.

| Layer | What it does | Real / Mock |
|---|---|---|
| **Filing Retriever** | Pulls 10-K/10-Q filings + XBRL facts from EDGAR | [`tools/filing_retriever.py`](tools/filing_retriever.py) |
| **MD&A Extractor** | Extracts prose paragraphs from filing HTML, cited to accession number | [`tools/mdna_extractor.py`](tools/mdna_extractor.py) |
| **Numerical Reconciler** | Checks a structured claim against XBRL facts (4 comparison types) | [`tools/reconciler.py`](tools/reconciler.py) |
| **Extraction Agent** | DSPy-optimized signature — prose → structured claims | [`eval/dspy_extractor.py`](eval/dspy_extractor.py) |
| **Verification Agent** | Resolves free-text claims to exact XBRL concepts, then reconciles | [`agents/verification_agent.py`](agents/verification_agent.py) |
| **Coordinator** | Routes the pipeline; enforces a budget; checkpoints for resume | [`agents/coordinator.py`](agents/coordinator.py) |
| **RLVR/GRPO fine-tuning** | Trains reconciliation-reasoning via the Reconciler-derived reward | [`phase7/train_grpo.py`](phase7/train_grpo.py) |

## Results

**Pillar 2 — DSPy prompt optimization**, measured on a held-out test set (hand-labeled *before* any DSPy code was written, to keep the ground truth honest):

| | Precision | Recall | F1 |
|---|---|---|---|
| Hand-written baseline prompt | 0.711 | 0.750 | 0.730 |
| DSPy zero-shot (`ChainOfThought`, no demos) | 0.763 | 0.806 | 0.784 |
| **DSPy optimized** (BootstrapFewShot, 4 demos) | **0.763** | **0.806** | **0.784** |

Switching the DSPy signature from `Predict` to `ChainOfThought` — reasoning before committing to structured output — took zero-shot DSPy from *underperforming* the hand-written baseline (0.676 F1) to clearly *beating* it (0.784 F1) before any optimization ran. `BootstrapFewShot` on top of that produced **exactly identical numbers** — verified this wasn't an evaluation bug (the compiled program is a genuinely separate instance, and the optimizer log confirms 4 real demonstrations were bootstrapped). Reasoning captured essentially all the available signal; the few-shot demos added nothing further.

**But the pooled F1 delta doesn't survive closer scrutiny, and that's worth reporting rather than smoothing over.** Two checks that changed the honest headline:

1. **Noise floor.** At n=45 claim-level decisions (the actual sample size the precision/recall proportions are computed over — not just the 16 test paragraphs), the 95% noise floor is **±0.120 F1**. The measured +0.054 delta over baseline is *inside* that noise floor — not statistically distinguishable from chance at this sample size. `eval/run_comparison.py` computes and prints this check on every run rather than reporting a bare delta.
2. **Stratified breakdown by `comparison_type` exposes a masked regression.** The pooled number hides a real split:

   | comparison_type | baseline F1 | optimized F1 |
   |---|---|---|
   | `growth_pct` | 1.000 | 1.000 |
   | `absolute` | 0.625 | 0.788 |
   | `absolute_change` | 0.429 | **0.308** (worse) |

   `growth_pct` is trivially easy for this task and perfect in every condition, propping up the pooled score. `absolute` genuinely improved with optimization. `absolute_change` got *worse* — a real regression that a pooled average completely hides, because it happened at the same time `absolute` improved and the two roughly canceled out. This is precisely the "aggregate smiles while a stratum bleeds" failure mode: reporting only the pooled F1 would have missed it entirely.

The honest conclusion: reasoning (`ChainOfThought`) is a real, large improvement over a bare `Predict` call. Whether `BootstrapFewShot` optimization helped, hurt, or did nothing net is **not resolved by this test set** — it's too small to tell, and what data exists suggests it may have *traded* accuracy from one claim type for another rather than improving overall. Ground truth: 78 hand-labeled examples / 161 claims / 9 true negatives, drawn from real MSFT and NVDA filings.

*A further caveat, now closed rather than just named:* `eval/dataset.py`'s split (train ~80% / test ~20%, by index) is a clean single-touch holdout in structure, but the same 20% test slice was scored and reported on repeatedly during this project's own development (baseline, zero-shot, optimized, the stratified breakdown, the noise-floor check above, GEPA) — every delta above was better read as "the best available estimate from a set we've looked at many times" than a clean holdout result. [`eval/run_fresh_holdout.py`](eval/run_fresh_holdout.py) fixes that directly: 25 examples / 59 claims hand-labeled from real AMZN and AAPL 10-K MD&A text — two companies absent from `eval/labeled_claims.jsonl` entirely, so this set had never been scored by anything before that run. Full write-up: [`docs/fresh-holdout-results.md`](docs/fresh-holdout-results.md). The headline number actually changes here: baseline collapsed on genuinely new companies (0.730 → 0.344 F1 — 44 false positives in the `absolute` category alone), while zero-shot DSPy held up (0.784 → 0.752). DSPy vs. the hand-written baseline is now a real, **noise-floor-clearing** result (+0.355 F1 at n=82, noise floor ±0.099) — the first time this specific question has been resolved rather than left "inconclusive at this sample size." A second, new finding came with it: `BootstrapFewShot` didn't just fail to help on the fresh set, it measurably hurt (0.752 → 0.698) — directionally consistent with the 4 bootstrapped demos overfitting to MSFT/NVDA's phrasing, though at n=25 that specific drop is itself still inside its own noise floor (±0.096–0.099), so it's a real, evidence-backed open question, not yet a proven regression.

Separately: `eval/run_comparison.py`'s original DSPy optimization metric (`dspy_metric`, still what `BootstrapFewShot` uses) returns a bare F1 float, no diagnostic feedback text. That matters for one specific decision — `dspy.GEPA` (a reflective optimizer, a candidate next step above `BootstrapFewShot`) only out-performs simpler optimizers when its metric can describe *why* an attempt failed; against a purely scalar pass/fail-shaped metric, GEPA's reflective-mutation step has nothing to read and its usual edge doesn't apply. The prerequisite is now built rather than left as a gap: `dspy_metric_with_feedback` (backed by `eval/metrics.py`'s `feedback_for_example`) returns the same score plus a diagnostic string naming exactly which claims were missed or extraneous, and accepts GEPA's calling convention directly. Trying GEPA is now a metric swap away, not a metric redesign, and it has since actually been run — see [`eval/run_gepa.py`](eval/run_gepa.py). Honest caveat stated up front there, not discovered after the fact: GEPA's reflective step is meant to be guided by a model *stronger* than the one being optimized, and this project has exactly one LLM endpoint (gpt-oss-20b) — it reflects on its own failures here, not a stronger model's read on them, a real limitation on what this run can show. Result: `auto="light"` (444 metric calls, ~37 min on the local single-GPU server) moved F1 from 0.784 (zero-shot) to **0.800** — a +0.016 delta that's *inside* the ±0.117 noise floor at this test set size, so not distinguishable from chance, same honest conclusion as the `BootstrapFewShot` result above. The per-category breakdown is more interesting than the pooled number: GEPA's rewritten instruction reached a perfect 1.000 F1 on `growth_pct` and improved `absolute` to 0.857, but `absolute_change` collapsed to 0.167 — a real trade-off, not a clean win, echoing the same "may have traded accuracy between claim types rather than improving overall" pattern already seen with `BootstrapFewShot`. Run again with a stronger reflection model or a larger `auto` budget and this could look different; reported as what actually happened with the one endpoint available, not what a better-resourced run might show.

**Pillar 4 pre-flight — the Reconciler's own correctness, isolated from extraction.** Before the Reconciler is ever trusted as an RLVR reward, `eval/reconciler_audit.py` runs it against a battery of known-good, known-bad, and adversarial cases (sign flips, order-of-magnitude confusion, mislabeled comparison types, exact tolerance-boundary probes) — 15/15 matched the hand-computed expected verdict, and critically, **zero false-"consistent" results**, the dangerous failure mode for a reward signal (a false "inconsistent" just costs training signal; a false "consistent" actively teaches a policy that a wrong answer was right). Enforced permanently in CI via `tests/test_reconciler_audit.py`. Every verdict also carries a machine-readable `reason_code` (`near_miss`, `missing_fact`, `zero_denominator`, ...) instead of only free-text explanation — usable both as reward-shaping material for Phase 7 and as structured error-analysis output today.

The audit also lets us apply the maker/checker precision-ceiling formula — `Pr(correct | accepted) = pr / (pr + (1-p)f)`, where `p` is extraction precision, `r` is verifier recall, and `f` is verifier false-positive rate — to state what actually bounds end-to-end system accuracy. From the audit's cases: `r = 1.000` (6/6), `f = 0.000` (0/8, ~0.375 95% upper bound by the rule of three at this sample size). With the DSPy-optimized extraction precision (`p = 0.763`), the formula gives a ceiling of **1.000** — a f=0 verifier means every claim it accepts as "consistent" is trustworthy regardless of upstream extraction precision, on this test surface. The small n keeps this illustrative rather than statistically tight, same caveat as the noise-floor finding above.

Reward-shaping design for Phase 7 was recorded in [`docs/phase7-reward-design.md`](docs/phase7-reward-design.md) before any GPU time was spent on it — full results below.

**Pillar 4 — RLVR/GRPO fine-tuning**, measured the same way as Pillar 2: baseline vs. trained, checked against the noise floor, not just reported as a raw delta. `Qwen2.5-7B-Instruct` (QLoRA, LoRA rank 32, via Unsloth) trained on a single RTX 4090, using the shaped Reconciler-derived reward to learn reconciliation-*reasoning* — given already-resolved claim values, reason to a verdict, rather than calling `reconcile()` directly:

| | baseline (zero-shot) | trained (1 epoch, 1,300 GRPO steps) |
|---|---|---|
| accuracy | 0.766 | **0.855** |
| mean reward | 0.633 | **0.826** |
| false-consistent rate (dangerous case) | 0.148 | **0.036** |
| format failures | 1/304 | **0/304** |

At n=304, the accuracy delta (+0.089) clears its 95% noise floor (±0.062) and the false-consistent-rate drop (−0.112) clears its own (±0.045) — both are real, not sampling noise. The `absolute_change` claim category shows the largest, also noise-floor-clearing gain (0.589→0.821). `bps_change` — the hardest category (two ratios, then a basis-point difference) — moved (0.233→0.400) but not provably at its small n=30, traced to a data-generator bug: the training set only had 2 real distinct ratios to draw from. Fixed (5 more ratio pairs added, `bps_change`'s share of the dataset 8.7%→13.0%) and retrained for real on the larger dataset — the honest result there was a **null**, not a win: on a fresh 706-example holdout, `bps_change` moved from 0.436→0.372, inside its own noise floor either direction. The leading hypothesis was that the GRPO step budget (still 1,300 steps) hadn't scaled with the 2.1×-larger dataset — **tested directly** by retraining a third time with steps scaled to match (2,730), and the hypothesis didn't hold: `bps_change` moved to 0.404, itself not a real change from either prior run, still below the 0.436 baseline. More steps didn't fix it — whatever's actually holding this category back isn't a step-budget problem. Everything else kept improving on this third run: accuracy (0.820→0.865) and the false-consistent rate (0.108→0.024) are both real again, and `absolute_change` reached the largest clean win across all three runs (0.667→0.867). Full breakdown, all three runs, a worked before/after example, and the three real upstream packaging bugs found getting here: [`docs/phase7-results.md`](docs/phase7-results.md).

**A real, current end-to-end run** (NVDA's FY2026 10-K, live EDGAR + live LLM, no cached answers):

```
consistent     Revenue                             $215,900,000,000   (EDGAR: $215,938,000,000 — 0.02% off)
inconsistent   Revenue growth                       65.0%             (EDGAR implies 46.09%)
consistent     Income tax expense                  $21,400,000,000    (EDGAR: $21,383,000,000 — 0.08% off)
inconsistent   Income tax expense (FY2025 figure)  $11,100,000,000    (compared against wrong period — known limitation, see below)
unverifiable   Data Center revenue                  68.0% growth      (segment-level, no top-level XBRL tag — correctly declined, not guessed)
```

**Engineering rigor:** 216 automated tests (unit + live-network + live-LLM tiers), green on every push via GitHub Actions.

**A real backtest: would this have caught an actual restatement?** Every result above checks the pipeline in the present tense. A sharper question: has this project's own bitemporal `as_of` machinery ever caught something real happening to a real company? SEC 8-K "Item 4.02" filings (a company disclosing its own past financials can't be relied on) supply real ground truth. Scanning real XBRL company-facts data for `(concept, period)` groups where an *amended* filing (`10-K/A`/`10-Q/A`) reports a materially different value than the original found **13,827 real restatement fingerprints across 541 companies** (after rejecting a looser method that was 82% routine reclassification noise, not corrections). Matching those to real prose claims in the original MD&A — via the project's real extraction agent, keeping claims whose metric resolves to the restated concept and whose value is within 5% — found **122 real matches across 16 companies**, from a full scan of the top 60 (by fingerprint magnitude) of those 541. Running the actual `reconcile()` function twice per match (once `as_of` the original filing date, once `as_of` the restated filing date, no LLM calls): **118 of 122 (96.7%) would have been flagged inconsistent by the time each restatement existed, using only data that existed at each point in time** — 102 as the clean "consistent when made, inconsistent once restated" pattern, 16 already caught by prose-rounding tolerance alone, 3 structurally unverifiable (a claim type the matching step doesn't capture enough context for, correctly declined rather than guessed), 1 genuine miss reported rather than hidden. Not a synthetic eval number — a real outcome across 16 real companies' real regulatory history. Full methodology and honest scope caveats: [`docs/backtest-results.md`](docs/backtest-results.md).

**Why build a deterministic reconciler instead of just asking an LLM?** Tested directly, not assumed. A naive baseline — same LLM endpoint, same claims, but the *raw, unfiltered* XBRL fact history instead of the bitemporally-correct value `reconcile()` would pick, asked to judge consistency "as of" a given date via prompting alone — was run against two already-verified test sets. On `eval/reconciler_audit.py`'s 15 adversarial cases: **11/15 (73.3%)**, against the deterministic reconciler's 15/15 — including a claim off by 1,000,000× (millions vs. raw dollars) that the naive baseline called "consistent." On the 122-match real backtest, testing bitemporal reasoning specifically: the naive baseline reproduced the correct "consistent when made, inconsistent once restated" pattern in only **50/102 (49%)** of the cases the deterministic pipeline got right, and **falsely called 49 of the 100 genuinely-consistent-at-filing claims "inconsistent"** — bleeding the later restatement's information backward into a judgment about the past, the exact failure mode the bitemporal `as_of` design exists to prevent. Full results: [`docs/naive-baseline-results.md`](docs/naive-baseline-results.md).

**Does it cry wolf on clean filings?** The backtest above only tests sensitivity. Run against 8 large tech companies confirmed to carry no restatement fingerprint (ADBE, CRM, ORCL, CSCO, INTU, IBM, QCOM, TXN — the real Coordinator, no mocks), 126 real claims produced 18 consistent, 6 inconsistent, an apparent 25% false-positive rate among resolved verdicts. Investigating rather than reporting that number at face value found two real, distinct causes: this specificity check itself surfaced a genuine, previously-undiscovered bug in `resolve_periods()`'s default comparison-period selection — confirmed against live TXN data, it picked a Q3 10-Q's 9-month year-to-date fact over the correct prior full year because nothing checked that the durations matched, making an accurate claim ("increased $2.04B, or 13.0%, compared to fiscal 2024") read as wildly inconsistent. **Fixed**, with regression tests from the real numbers — 8 of the original 14 false positives resolved correctly afterward. The remaining 6 trace to extraction, not the reconciler: three TXN claims show the identical pattern of grabbing the *prior*-year number from a "$X ... compared with $Y" sentence instead of the current one. Given correct inputs, the reconciler's own false-positive rate across all 24 resolved verdicts is **0/24** — every miss traces to either the now-fixed resolver bug or an upstream extraction error, never an incorrect verdict against accurate, correctly-resolved data. A third bug turned up narrowing this same control set to one vertical: INTU's real 10-K returned zero claims because its heading ("ITEM 7 - MANAGEMENT'S DISCUSSION...") uses a dash where the parser only tolerated a period — **fixed**, with regression tests, and INTU now returns 280 real MD&A chunks instead of 0. IBM's zero claims turned out not to be a bug at all: its 10-K literally incorporates MD&A by reference to a separate annual-report exhibit rather than including it inline — a real, different, larger gap, not addressed here. Full write-up: [`docs/specificity-check-results.md`](docs/specificity-check-results.md).

## What actually broke, and how it got caught

The interesting engineering in this project wasn't writing the happy path — it was what real SEC data did to the happy path. A few examples, each caught by testing against *live* data rather than trusting the design on paper:

- **A dangerous string-matching shortcut.** Metric resolution originally tolerated wording variance via substring matching — but `"revenue"` is a substring of `"Azure and other cloud services revenue"`, so it would have silently verified a *segment's* revenue against *total company* revenue and produced a confidently wrong "consistent" verdict. Fixed to exact-match-plus-explicit-aliases only. [`agents/resolver.py`](agents/resolver.py)

- **XBRL doesn't tag percentages as `percent`.** Rate concepts like effective tax rate are reported as `unit="pure"` — a decimal fraction (`0.19`, not `19`). A claim stating "19%" would have either never matched the fact at all, or (once that was fixed) been compared as `19.0` against `0.19` and read as wildly inconsistent even when correct. [`agents/verification_agent.py`](agents/verification_agent.py)

- **Companies retag their own XBRL concepts over time.** NVDA reported revenue under `RevenueFromContractWithCustomerExcludingAssessedTax` through FY2022, then switched to plain `Revenues` — the old tag never disappears from the company's concept list, it just stops getting new data. Picking "first candidate that exists at all" silently locked onto a tag with no data newer than 2022. Fixed to pick by data recency.

- **A later filing can corrupt an earlier one's claims.** A claim extracted from a 10-K's MD&A describes *that 10-K's* fiscal year — but if a newer 10-Q has since been filed (nearly always true), its more recent quarterly `period_end` would otherwise get picked as "current," comparing the wrong two numbers entirely. Fixed with an `as_of` cutoff anchoring resolution to the claim's own source filing.

- **A metric-matching bug caught by its own unit test before it ever ran on real data**: an early precision/recall scorer would have counted generic `"Revenue"` as matching segment-specific `"Microsoft Cloud revenue"` — a bug in the *measurement*, which is worse than a bug in the thing being measured, since it makes bad results look good. Caught and fixed before the real comparison ran.

- **Chain-of-thought reasoning silently corrupted its own numeric output.** Switching to `dspy.ChainOfThought` made the model write dollar amounts in shorthand inside its `reasoning` field — *"$215.9 billion"* — and then just echo that literal `215.9` into the structured `value` field instead of expanding it to `215900000000`, consistently, every run. Plain `Predict` never had this failure mode because it never generates that intervening shorthand text to anchor on. Fixed by instructing the signature to write the fully-expanded number in the reasoning itself, so the structured step has nothing left to get wrong. [`eval/dspy_extractor.py`](eval/dspy_extractor.py)

- **A claim could be marked wrong for being correct — at the time it was made.** SEC filings restate: a later 10-K/A amendment can revise an XBRL figure after the original filing. Without tracking *when* a fact was filed relative to the claim's own source filing, the reconciler's restatement-handling logic (correctly preferring the most recent value when duplicates disagree) could compare an old, accurate claim against a *later* restatement and call it "inconsistent" — the claim was right when it was written, and the world's record of that period simply changed afterward. Fixed by threading the claim's own filing date (`as_of`) into the reconciler's fact-matching itself, not just its period-selection layer (which Phase 5 already handled). [`tools/reconciler.py`](tools/reconciler.py)

- **The same bitemporal bug, found a second time via a tool-contract audit.** Reading `tools/numerical_reconciler.py`'s MCP tool docstring as if deciding how to call it (the "read it like the calling model would" discipline) surfaced that `reconcile_claim` — the literal Pillar 1 tool, callable independently of the Coordinator — never exposed an `as_of` parameter at all, so a direct caller had no bitemporal protection even after the agent-path fix above landed. Fixed by adding `as_of` to the tool's signature and threading it into `reconcile()`. [`tools/numerical_reconciler.py`](tools/numerical_reconciler.py)

- **A claim's own stated period was being ignored entirely.** `resolve_periods()` always picked the globally most-recent fact as "current," regardless of what the claim's `period` text actually said — so two claims from the same sentence naming two different fiscal years ("Income tax expense was $21.4 billion and $11.1 billion for fiscal years 2026 and 2025, respectively") both resolved to the *same* (current, comparison) pair, and the FY2025 claim ended up checked against the FY2026 fact. Confirmed against this exact live NVDA sentence: without a hint, the FY2025 claim's "current" resolved to the FY2026 annual figure ($21.38B) against a $13.9B quarterly comparison — nonsensical. Fixed by threading the claim's own period text through as a hint: an explicit fiscal year ("fiscal year 2025", "FY2025") now picks that year's fact as current; "sequentially" vs. "a year ago"/"year-over-year" now correctly select the immediately-preceding period vs. the same period ~12 months back for `growth_pct` claims, instead of always picking "next most recent" regardless of which the text meant. Re-run against the same live sentence: both the FY2026 and FY2025 income-tax claims now independently resolve `consistent` against their own correct periods. [`agents/resolver.py`](agents/resolver.py)

- **A named quarter resolved to the wrong duration entirely.** An *ordinal* quarter reference ("the third quarter", "Q3") is safe to resolve — XBRL's own `fiscal_period` field is itself fiscal-quarter-numbered, so "the third quarter" maps directly to `fp="Q3"` with no fiscal-year-end guessing involved, unlike a calendar-named quarter ("the September quarter"). Implementing it surfaced a second, sharper bug before it shipped: a 10-Q commonly tags *both* the standalone 3-month figure and a 6-/9-month year-to-date cumulative with the identical `fiscal_period` and `period_end` — an unfiltered match against real NVDA data returned a $91.166B "Q3" figure that was actually the 9-month cumulative, not the $35.082B standalone quarter. Fixed by preferring the ~91-day-length fact when both exist. [`agents/resolver.py`](agents/resolver.py)

- **A handful of real unresolved claims traced to missing dictionary keys, not missing data.** Checking exactly which metric texts in `eval/labeled_claims*.jsonl` failed to resolve (rather than assuming the gap was all segment-level) found two different things mixed together: genuine segment claims with no fix available, and plain wording variants ("sales" for revenue, "provision for income taxes" for income tax expense) or new-but-standard concepts (`LongTermDebt`, `InvestmentIncomeInterest`, `NetCashProvidedByUsedInFinancingActivities`, `OtherNonoperatingIncomeExpense`, `OperatingLeaseLiabilityNoncurrent`) confirmed present in real AAPL/MSFT/NVDA/AMZN data before being added. Resolution rate on the fresh AMZN/AAPL holdout: **13.6% → 44.1%**. [`agents/resolver.py`](agents/resolver.py)

One limitation is still open and documented rather than hidden: a quarter named by *calendar month* ("the September quarter", "the December quarter") still isn't parsed into a specific fiscal period — translating that safely needs the company's own fiscal-year-end, which isn't available at this resolution step, so that case still falls back to the default "most recent" pick rather than guessing. Flagged in [`agents/resolver.py`](agents/resolver.py) as follow-up, not silently left broken. What the resolver's scope actually covers, measured directly against 676 real companies' filings and against real hand-labeled claims rather than left as an implicit gap: [`docs/robustness-and-scope.md`](docs/robustness-and-scope.md#coverage-how-much-of-a-real-filing-is-actually-resolvable). A fuller accounting of what's explicitly in and out of scope — maker/checker independence, idempotency, why doom-loop detection doesn't apply here — is in the same doc.

## Harness: budget and checkpointing

A real 10-K MD&A can run 100+ paragraphs, each needing its own LLM call. The coordinator enforces a `Budget` (max chunks / max LLM calls / max wall-clock seconds) and checkpoints progress after every chunk — a crash or an intentional stop loses at most the one chunk in flight, not the whole run. Verified against the live pipeline: a budget-interrupted run picks up exactly where it left off on the next call, without re-processing anything already done. [`agents/checkpoint.py`](agents/checkpoint.py)

## Structure

```
tools/    MCP servers — Filing Retriever, MD&A Extractor, Numerical Reconciler (Pillar 1)
eval/     DSPy signature, hand-labeled test set, baseline-vs-optimized harness (Pillar 2)
agents/   Coordinator + retrieval/extraction/verification agents, budget, checkpointing (Pillar 3)
phase6/   End-to-end integration-at-scale run (real Coordinator, 5 companies, no mocks)
phase7/   RLVR/GRPO fine-tuning — dataset builder, reward, training/eval scripts (Pillar 4, GPU-only)
research/ Real-restatement backtest — find fingerprints, match prose, reconcile bitemporally
docs/     Design docs — reward design, results, robustness & scope
data/     Local cache / checkpoints (gitignored)
tests/    216 tests — 208 unit (always run) + 8 live-network/live-LLM (opt-in via -m)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # then fill in EDGAR_USER_AGENT and LLM_* config
```

## Test

```bash
pytest -v                 # 208 unit tests, no network/LLM required
pytest -v -m network       # + live SEC EDGAR checks
pytest -v -m llm           # + live LLM checks (requires LLM_BASE_URL reachable)
```

## Status

**All phases (0–7) complete.** Phase 6 (end-to-end integration at scale — the real Coordinator, no mocks, run against 5 companies including 2 never used during development) surfaced two honest, well-diagnosed coverage gaps rather than a clean pass. Both are now addressed, each to the extent it honestly could be: MD&A heading detection didn't generalize past the 3 companies it was built against, root-caused and fixed in `tools/mdna_parser.py` (all 5 companies now retrieve real MD&A text). The resolver's 14-concept scope turned out to be two different problems: part was just missing dictionary entries for real, standard concepts (18 more added and verified against live AAPL/MSFT/NVDA data — MSFT's "remaining performance obligation" claims now resolve to real consistent/inconsistent verdicts instead of "unverifiable"), and part is a genuine data-source limit confirmed by inspecting the raw SEC data directly: segment/product figures (Azure, LinkedIn, XBOX revenue) have no dimensional breakdown anywhere in the company-facts API this pipeline reads, so those correctly stay unresolved rather than guessed. Zero crashes across the batch throughout; the harness held up, the coverage claims are now precisely bounded rather than vague, and that distinction is the actual finding. The harness's other claim — checkpoint/resume — was also exercised against a real filing (not just mock scenarios): a real budget stop mid-document, a real on-disk checkpoint, a real resume that skipped the already-processed chunks and continued forward — see [`docs/phase6-results.md`](docs/phase6-results.md).
