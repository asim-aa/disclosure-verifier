# Phase 7 — RLVR/GRPO fine-tuning

Trains a small open model to reason through numerical reconciliation — the
arithmetic `tools/reconciler.py` already does deterministically — using the
Reconciler itself as the RLVR reward oracle. See
[`docs/phase7-reward-design.md`](../docs/phase7-reward-design.md) for the reward
design decided before any of this was built,
[`docs/phase7-results.md`](../docs/phase7-results.md) for what actually happened
when it ran (three runs: an initial one at accuracy 0.766→0.855, false-consistent
rate 0.148→0.036 at n=304; a retrain on a fixed `bps_change` dataset at
accuracy 0.820→0.858, false-consistent rate 0.108→0.020 at n=706; a third run
directly testing whether more GRPO steps would fix `bps_change`'s null result
(it didn't) that pushed accuracy to 0.865 and false-consistent to 0.024 — every
run's false-consistent-rate drop clears its own noise floor; `bps_change`
itself did not, on any of the three), and
[`docs/robustness-and-scope.md`](../docs/robustness-and-scope.md) for the
Reconciler's own pre-flight audit (0 false-"consistent" results, precision
ceiling of 1.000 at current extraction quality) that this reward depends on.

## Why this task, specifically

The model is handed **already-resolved** values — which concept, which periods,
what the reported numbers are — not asked to search for them. Concept/period
resolution (`agents/resolver.py`) is deterministic code, not a reasoning task.
This isolates the one part that's genuinely a capability question for a small
model: does it reliably reason through the *arithmetic* of reconciliation, given
clean inputs. See the README's "Pillar 4" section for why this is RL's job and
not more DSPy optimization (a capability gap, not a procedure gap).

## Environment

GPU-only — needs the `phase7` extra (`torch`, `unsloth`, `trl`, `peft`,
`bitsandbytes`, `datasets`), not installed by default:

```bash
uv sync --extra dev --extra phase7
```

Trained and run on a single RTX 4090 (24GB VRAM) — not the RTX 5090/32GB the
original planning doc assumed. Qwen2.5-7B-Instruct in 4-bit (QLoRA) via Unsloth
fits comfortably with headroom for GRPO's multi-completion rollouts.

## Pipeline

1. **Build the dataset** (offline, no GPU, no network — uses cached XBRL facts
   already in `data/cache/`):

   ```bash
   uv run python -m phase7.build_dataset
   ```

   Writes `phase7/data/{train,test}.jsonl`. Ground truth is never hand-labeled —
   every example's `gold_verdict`/`gold_reason_code` comes from literally calling
   `tools.reconciler.reconcile()` against real (or deliberately constructed —
   same philosophy as `eval/reconciler_audit.py`) `FinancialFact`s, so it's
   correct by construction. See the module docstring for why this is built
   directly from resolved `(concept, ticker, period)` triples rather than by
   running `eval/labeled_claims.jsonl` through full concept resolution (most of
   those gold claims are segment-level metrics `agents/resolver.py` can't map to
   a top-level XBRL concept by design — that would produce a dataset that's
   mostly "unverifiable" with little arithmetic-reasoning signal).

2. **Evaluate the base model first** (establishes the pre-training baseline —
   don't skip this, the whole point of an honest before/after comparison):

   ```bash
   uv run python -m phase7.evaluate --adapter none
   ```

3. **Train**:

   ```bash
   uv run python -m phase7.train_grpo
   ```

   Saves a LoRA adapter to `phase7/outputs/lora_adapter/`.

4. **Evaluate the trained model** and compare against step 2's numbers — report
   both, the same discipline `eval/run_comparison.py` applies to Pillar 2 (noise
   floor, stratified breakdown, no bare "it got better" claim without checking
   whether the delta clears the noise floor at this test set's size):

   ```bash
   uv run python -m phase7.evaluate --adapter phase7/outputs/lora_adapter
   ```

## Files

| File | Purpose |
|---|---|
| `schema.py` | `ReconciliationExample` — the shared shape for one training/eval example |
| `prompts.py` | The prompt template — shared by the dataset builder and both training/eval, so what the model trains on and what it's shown at generation time can't drift apart |
| `reward.py` | `compute_reward()` — the shaped GRPO reward (verdict match + magnitude shaping via `reason_code` − false-consistent penalty), and `parse_verdict()` |
| `build_dataset.py` | Generates the dataset from cached XBRL facts + the real `reconcile()` |
| `train_grpo.py` | Unsloth + TRL `GRPOTrainer` training loop |
| `evaluate.py` | Runs a model (base or trained) against the held-out test set, reports stratified accuracy + the false-consistent rate |
| `show_completion.py` | One-off: prints a single example's full prompt + completion for a given model — used to pull real before/after text for the results write-up |
| `_trl_import_shim.py` | Stubs two `trl` optional-integration imports (`llm_blender`, `weave`) that are broken/unused upstream dependencies unrelated to GRPO training itself — see `docs/phase7-results.md`'s "What actually broke" |

## Known knobs, if training doesn't go well

- `NUM_GENERATIONS` in `train_grpo.py` (6) — TRL's default is 8; tuned down for
  24GB VRAM headroom. Raise it if there's room; GRPO's group-relative advantage
  needs enough samples per prompt to have signal.
- The reward shaping constants in `reward.py` (`FALSE_CONSISTENT_PENALTY`,
  `_SHAPING_BY_REASON_CODE`) — watch for the GRPO failure modes named in
  `docs/phase7-reward-design.md` (zero-advantage batches, entropy collapse,
  length bias) before assuming a shaping constant needs retuning.
- Dataset class balance (`phase7/build_dataset.py`'s `PERTURBATION_KINDS`
  sampling) — printed by `build_dataset.py` on every run; a policy that
  converges to always guessing one verdict is a sign the balance drifted, not
  necessarily a reward bug.
