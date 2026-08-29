# Phase 7 results — RLVR/GRPO fine-tuning

The reward design in [`docs/phase7-reward-design.md`](phase7-reward-design.md) was written before any GPU time was spent. This records what actually happened once it was.

## Setup

- **Model:** `Qwen/Qwen2.5-7B-Instruct`, QLoRA (4-bit, LoRA rank 32) via Unsloth
- **Hardware:** a single RTX 4090 (24GB) — not the RTX 5090/32GB the original planning doc assumed, but sufficient
- **Reward:** the shaped Reconciler-derived reward from [`phase7/reward.py`](../phase7/reward.py) — verdict match, magnitude shaping via `reason_code`, and a penalty for the dangerous false-"consistent" case
- **Data:** 1,612 examples (1,308 train / 304 test), built entirely offline from cached XBRL facts already on disk — see [`phase7/build_dataset.py`](../phase7/build_dataset.py). Every example's ground truth comes from literally calling `tools.reconciler.reconcile()`, never hand-labeled
- **Task:** reconciliation-reasoning. The model sees an already-resolved claim plus the raw reported values (not asked to search for them — concept/period resolution is deterministic code, not a reasoning task) and must reason to a verdict. This isolates the one part of the pipeline that's a genuine capability question for a small model: does it reliably do the *arithmetic*, not whether the prompt asks nicely enough

Before any of this ran, `eval/reconciler_audit.py`'s 15-case adversarial audit confirmed 0 false-"consistent" results in the reward function itself — the pre-flight check this project's own reward-design doc called for before trusting a reward signal with GPU time.

## Two runs, two very different stories

**Run 1 — 150 steps (~31 min, ~11.5% of one epoch).** Built to prove the pipeline worked end-to-end (it did, after fixing three real upstream packaging bugs along the way — see "What actually broke," below). The result was directionally positive but statistically inconclusive: a +0.027 accuracy delta against a ±0.066 95% noise floor at n=304. Reported honestly as inconclusive, the same standard this project already applies to the Pillar 2 DSPy comparison.

**Run 2 — 1,300 steps (~4h15m, one full epoch).** A materially different result:

| | baseline (zero-shot) | 150 steps | **1,300 steps (1 epoch)** |
|---|---|---|---|
| accuracy | 0.766 | 0.793 | **0.855** |
| mean reward | 0.633 | 0.685 | **0.826** |
| false-consistent rate | 0.148 | 0.122 | **0.036** |
| format failures | 1/304 | 1/304 | **0/304** |

By comparison type:

| comparison_type | baseline | 150 steps | 1,300 steps |
|---|---|---|---|
| `absolute` | 0.949 | 0.932 | 0.975 |
| `absolute_change` | 0.589 | 0.714 | **0.821** |
| `bps_change` | 0.233 | 0.233 | 0.400 |
| `growth_pct` | 0.810 | 0.840 | 0.870 |

## Checking against the noise floor, not just the raw numbers

Same discipline as `eval/run_comparison.py` applies to Pillar 2 — a delta only counts if it clears the sampling noise at this test set's size (`SE(p) = sqrt(p(1-p)/n)`, combined across both proportions for a two-sample comparison, ~95% half-width `1.96 * SE`):

- **Accuracy** (n=304): delta +0.089 vs. a combined 95% half-width of ±0.062 — **clears the noise floor.** This is a real improvement, not a lucky sample.
- **False-consistent rate** (n=304): delta −0.112 vs. ±0.045 — **clears the noise floor.** The dangerous failure mode (confidently wrong in the unsafe direction) genuinely dropped, not just moved within noise.
- **`absolute_change` stratum** (n=56): delta +0.232 vs. ±0.163 — **clears the noise floor.** The largest and most defensible per-category gain.
- **`growth_pct` stratum** (n=100): delta +0.060 vs. ±0.101 — does not clear the noise floor on its own, though directionally positive.
- **`bps_change` stratum** (n=30): delta +0.167 vs. ±0.231 — does not clear the noise floor at this small n. But it's worth noting this is no longer the exact null the 150-step run produced (7/30 both before and after, identical) — the 1,300-step run moved it to 12/30. Real movement, just not provable at n=30. `bps_change` is also the smallest and hardest category in the training set (two ratios, then a basis-point difference) — the most likely place a longer run or oversampled training data would help next.

## Retraining on the fixed `bps_change` dataset

`bps_change` stayed the weakest category through the run above (0.233 → 0.400), and [`phase7/build_dataset.py`](../phase7/build_dataset.py)'s data was the traced cause, not the model's reasoning: the generator only had 2 real distinct ratios to draw from (gross margin and operating margin — 2 of its original 4 "pairs" were the same ratio under a company's alternate revenue tag, not a second ratio), making `bps_change` 8.7% of training data against 43.3% for `absolute`. Fixed by adding 5 more ratio pairs (net margin, R&D intensity, SG&A ratio, cost ratio, opex ratio) built from concepts already verified present in real company data — `bps_change`'s share rose to 13.0%, raw count 114 → 350, total dataset 1,612 → 3,403 examples.

Retrained on the same box, same architecture, same 1,300 GRPO steps, evaluated on a fresh 706-example held-out split of the larger dataset (not comparable row-for-row to the 304-example table above — different split, different size):

| | base (zero-shot) | trained (1,300 steps) | delta | 95% noise floor | resolved? |
|---|---|---|---|---|---|
| **overall accuracy** | 0.820 | 0.858 | +0.038 | ±0.038 | right at the edge |
| **false-consistent rate** | 0.108 | 0.020 | **−0.088** | ±0.025 | **yes — real, big win** |
| `absolute` | 0.969 | 0.994 | +0.025 | ±0.021 | yes, real (small, near ceiling) |
| `absolute_change` | 0.667 | 0.825 | **+0.158** | ±0.108 | **yes — real, substantial** |
| `growth_pct` | 0.858 | 0.893 | +0.035 | ±0.070 | no — inside noise floor |
| **`bps_change`** | **0.436** | **0.372** | **−0.064** | ±0.140 | **no — inside noise floor, wrong direction** |

**The honest headline: the `bps_change` dataset fix did not produce a measurable win in this run.** The number moved down slightly (0.436 → 0.372), though well inside the noise floor at this small stratum (n=94) — not proven worse either, just not proven better, despite the effort that went into tracing and fixing the underlying data bug.

The other results are real, though. The false-consistent rate — the dangerous failure mode, a model confidently endorsing a wrong claim — dropped from 10.8% to 2.0% on this independent run, a real, noise-floor-clearing improvement that echoes the first run's finding. `absolute_change` also improved substantially and really (+0.158).

**A plausible reason for the `bps_change` null, not yet confirmed:** `max_steps` stayed fixed at 1,300 while the dataset grew 2.1× (1,612 → 3,403 examples). Even though `bps_change`'s raw training-example count tripled (114 → 350), the fixed step budget means every example — including `bps_change` ones — got proportionally *less* repeated exposure than in the original run. Tripling the numerator didn't help if the denominator (training steps) didn't grow to match. If this hypothesis is right, the experiment actually worth running next is more steps on the bigger dataset, not the reproportioned data alone — flagged as follow-up, not run here.

## A worked example

Same held-out example (`achg-435`, an `absolute_change` claim about AAPL revenue), both models, greedy decoding. The gold verdict is `consistent` and both models get there — this is presented for reasoning *quality*, not a verdict flip.

**Base model:**
```
Step 1: Calculate the actual change in revenue.
Actual change = Current period value - Prior period value
             = 274,515,000,000 USD - 2260,174,000,000 USD
             = 14,341,000,000 USD
Step 2: Check if the claimed value falls within the tolerance range.
Tolerance = 0.01 * max(2260,174,000,000 USD, 274,515,000,000 USD)
         = 0.01 * 274,515,000,000 USD
         = 22,701,500,000 USD
Step 3: Compare the claimed value to the tolerance range.
Claimed value: 14,341,000,000 USD
Tolerance range: -22,701,500,000 USD to 22,701,500,000 USD
Since the claimed value of 14,341,000,000 USD falls within the tolerance range, the claim is consistent.
VERDICT: consistent
```

**Trained model (1,300 steps):**
```
Step 1: Identify the relevant values from the reported data.
- Revenue for period ending 220-09-26: 274,515,000,000 USD
- Revenue for period ending 219-09-228: 260,174,000,000 USD
Step 2: Calculate the actual change in revenue.
- Actual change = Current revenue - Prior revenue
- Actual change = 274,515,000,000 USD - 260,174,000,000 USD
- Actual change = 14,341,000,000 USD
Step 3: Compare the claimed value with the calculated actual change.
- Claimed value: 14,341,000,000 USD
- Actual change: 14,341,000,000 USD
Since the claimed value matches the actual calculated change in revenue, the claim is consistent.
VERDICT: consistent
```

Both reach the right verdict, but by different, unequally sound paths. The base model's arithmetic literally typos a number mid-calculation (`2260,174,000,000` instead of `260,174,000,000`) yet still lands on the correct difference — the final number looks recalled rather than derived. It also checks the claim against a symmetric tolerance band around zero (`is the claimed value between −22.7B and +22.7B?`), which is a different and weaker test than what it's actually supposed to check — it happens to pass here only because the claimed change is small relative to the tolerance band, not because the method is correct. The trained model computes the actual change cleanly and directly compares it to the claimed value — the same check the real `reconcile()` function performs. That's the shape of improvement the aggregate numbers above are made of.

## What actually broke getting here

Three real upstream packaging bugs, found only by running the pipeline for real, none of them about the reward design itself:

- `trl`'s `GRPOTrainer` unconditionally imports `llm_blender` (unmaintained, incompatible with a current `transformers`) and `weave` (a false-positive availability check) for judge/logging features never used here — worked around with a `sys.modules` stub (`phase7/_trl_import_shim.py`) rather than chasing dependency versions for functionality this project doesn't touch.
- `GRPOTrainer.__init__` assumes `model.warnings_issued` exists (a standard `transformers.PreTrainedModel` attribute) — the PEFT/Unsloth-wrapped model doesn't initialize it. Fixed with a one-line defensive guard.
- `generation_batch_size` (`per_device_train_batch_size * gradient_accumulation_steps`) must be evenly divisible by `num_generations` — TRL enforces this and errors otherwise; the initial config didn't satisfy it.

None of these are interesting on their own, but the honest-failure-reporting standard this project holds itself to applies to infrastructure bugs too, not just modeling results.

## Reproducing

```bash
uv sync --extra dev --extra phase7          # GPU box only
uv run python -m phase7.build_dataset       # offline, no GPU
uv run python -m phase7.evaluate --adapter none                          # baseline
uv run python -m phase7.train_grpo                                       # trains, saves phase7/outputs/lora_adapter
uv run python -m phase7.evaluate --adapter phase7/outputs/lora_adapter   # trained
```

See [`phase7/README.md`](../phase7/README.md) for the full pipeline breakdown and tunable knobs.
