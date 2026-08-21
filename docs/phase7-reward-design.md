# Phase 7 reward design (planning doc — not yet built, pending GPU compute)

This records a design decision made *before* any GRPO training run, so it doesn't
get made ad hoc once compute is available and there's pressure to just start. No
code here — this is what to build when Phase 7 starts.

## Shaped, not sparse

The reward must not collapse the Reconciler's output to a bare pass/fail bit
(`1.0` if the model's verdict matches ground truth, `0.0` otherwise). A sparse
binary reward gives GRPO's group-relative advantage computation almost nothing to
discriminate between 8 sampled completions early in training, when most of them
are simply wrong — every wrong completion looks identically bad, so there's no
gradient telling the policy *which direction* of wrong to move away from.

The Reconciler already computes everything a shaped reward needs — it's just not
threaded through to a scalar yet:

- **`difference`** — the actual numeric miss (relative % for absolute/absolute_change,
  percentage points for growth_pct, bps for bps_change), already returned on every
  `ReconciliationResult`.
- **which `comparison_type` was involved** — a model that gets the right verdict
  via the wrong reasoning path (e.g. treating a `growth_pct` claim as `absolute`)
  should be distinguishable from one that reasoned correctly.
- **verdict-class confusion is not symmetric** — `consistent` mistaken for
  `inconsistent` is a false alarm (costly, but safe-ish); `inconsistent` mistaken
  for `consistent` is the dangerous one (see `eval/reconciler_audit.py`'s
  false-CONSISTENT tracking — the same asymmetry applies to a trained policy's
  verdicts, not just the Reconciler's own).

Proposed shape (subject to revision once real training data exists):

```
reward = base_verdict_match          # 1.0 if predicted verdict == ground truth, else 0.0
       + magnitude_shaping           # for wrong verdicts: small positive credit,
                                      # scaled by how close `difference` was to the
                                      # tolerance boundary — "confidently wrong" is
                                      # penalized more than "wrong by a hair"
       - false_consistent_penalty    # extra penalty specifically for predicting
                                      # "consistent" when ground truth is
                                      # "inconsistent" — the asymmetric case
```

## Audit before training, not after

Before any GRPO run spends GPU time treating the Reconciler's verdict as ground
truth, re-run `eval/reconciler_audit.py` (already built, already in CI) and
confirm `false_consistent_count == 0` still holds. If a change to the Reconciler
or a new adversarial case is added later and this regresses, that's a stop-ship
signal for starting or continuing training — a Reconciler with an exploitable
seam doesn't just add noise to the reward, it gives GRPO's optimization pressure
something specific to find and amplify. A model trained against a flawed reward
doesn't get lucky once; it internalizes the shortcut.

## Known GRPO failure modes to watch for once training starts

Named explicitly so they're monitored on purpose, not discovered as unexplained
instability mid-run:

- **Zero-advantage batches** — all 8 sampled completions in a group get the same
  reward (e.g. all wrong in the same way), giving no relative signal to learn from.
- **Entropy collapse** — the policy converges to near-deterministic outputs too
  early, losing the sampling diversity GRPO's group comparison depends on.
- **Length bias** — the model learns to pad its reasoning trace without actually
  improving verdict accuracy (the same failure shape as the verbosity-bias judge
  pathology documented for LLM-judge metrics, applied to a trained policy instead).

## Why RL, not just more DSPy optimization

Already-covered ground worth restating here for continuity: `eval/run_comparison.py`
found DSPy optimization (BootstrapFewShot few-shot demos) added *zero* measurable
improvement to claim extraction once reasoning (`ChainOfThought`) was already in
place — identical precision/recall to zero-shot. That's a "can-but-doesn't" gap
(the model already reasons correctly given the right prompt) closing, not a
capability gap. RLVR/GRPO in Phase 7 targets a different task — verdict
*classification* under the Reconciler's reward — which is plausibly a genuine
capability question for a small open model (does it reliably reason through
multi-step numerical reconciliation at all, not just "does the prompt ask nicely
enough"). That's the right kind of problem for weight-level optimization, not
prompt-level.
