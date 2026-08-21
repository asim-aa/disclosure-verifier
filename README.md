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

Built as the capstone for the SupportVectors AI Agents Bootcamp, against four required pillars: real MCP tools, a DSPy-optimized prompt with a measured baseline delta, a justified multi-agent architecture, and (pending compute) RLVR/GRPO fine-tuning.

📄 **[Read the full case study (PDF)](docs/case-study.pdf)** — results, a live example, and the real bugs found along the way.

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

**Pillar 4 pre-flight — the Reconciler's own correctness, isolated from extraction.** Before the Reconciler is ever trusted as an RLVR reward, `eval/reconciler_audit.py` runs it against a battery of known-good, known-bad, and adversarial cases (sign flips, order-of-magnitude confusion, mislabeled comparison types, exact tolerance-boundary probes) — 15/15 matched the hand-computed expected verdict, and critically, **zero false-"consistent" results**, the dangerous failure mode for a reward signal (a false "inconsistent" just costs training signal; a false "consistent" actively teaches a policy that a wrong answer was right). Enforced permanently in CI via `tests/test_reconciler_audit.py`. Reward-shaping design for Phase 7 itself — not yet built, pending GPU compute — is recorded in [`docs/phase7-reward-design.md`](docs/phase7-reward-design.md).

**A real, current end-to-end run** (NVDA's FY2026 10-K, live EDGAR + live LLM, no cached answers):

```
consistent     Revenue                             $215,900,000,000   (EDGAR: $215,938,000,000 — 0.02% off)
inconsistent   Revenue growth                       65.0%             (EDGAR implies 46.09%)
consistent     Income tax expense                  $21,400,000,000    (EDGAR: $21,383,000,000 — 0.08% off)
inconsistent   Income tax expense (FY2025 figure)  $11,100,000,000    (compared against wrong period — known limitation, see below)
unverifiable   Data Center revenue                  68.0% growth      (segment-level, no top-level XBRL tag — correctly declined, not guessed)
```

**Engineering rigor:** 125 automated tests (unit + live-network + live-LLM tiers), green on every push via GitHub Actions.

## What actually broke, and how it got caught

The interesting engineering in this project wasn't writing the happy path — it was what real SEC data did to the happy path. A few examples, each caught by testing against *live* data rather than trusting the design on paper:

- **A dangerous string-matching shortcut.** Metric resolution originally tolerated wording variance via substring matching — but `"revenue"` is a substring of `"Azure and other cloud services revenue"`, so it would have silently verified a *segment's* revenue against *total company* revenue and produced a confidently wrong "consistent" verdict. Fixed to exact-match-plus-explicit-aliases only. [`agents/resolver.py`](agents/resolver.py)

- **XBRL doesn't tag percentages as `percent`.** Rate concepts like effective tax rate are reported as `unit="pure"` — a decimal fraction (`0.19`, not `19`). A claim stating "19%" would have either never matched the fact at all, or (once that was fixed) been compared as `19.0` against `0.19` and read as wildly inconsistent even when correct. [`agents/verification_agent.py`](agents/verification_agent.py)

- **Companies retag their own XBRL concepts over time.** NVDA reported revenue under `RevenueFromContractWithCustomerExcludingAssessedTax` through FY2022, then switched to plain `Revenues` — the old tag never disappears from the company's concept list, it just stops getting new data. Picking "first candidate that exists at all" silently locked onto a tag with no data newer than 2022. Fixed to pick by data recency.

- **A later filing can corrupt an earlier one's claims.** A claim extracted from a 10-K's MD&A describes *that 10-K's* fiscal year — but if a newer 10-Q has since been filed (nearly always true), its more recent quarterly `period_end` would otherwise get picked as "current," comparing the wrong two numbers entirely. Fixed with an `as_of` cutoff anchoring resolution to the claim's own source filing.

- **A metric-matching bug caught by its own unit test before it ever ran on real data**: an early precision/recall scorer would have counted generic `"Revenue"` as matching segment-specific `"Microsoft Cloud revenue"` — a bug in the *measurement*, which is worse than a bug in the thing being measured, since it makes bad results look good. Caught and fixed before the real comparison ran.

- **Chain-of-thought reasoning silently corrupted its own numeric output.** Switching to `dspy.ChainOfThought` made the model write dollar amounts in shorthand inside its `reasoning` field — *"$215.9 billion"* — and then just echo that literal `215.9` into the structured `value` field instead of expanding it to `215900000000`, consistently, every run. Plain `Predict` never had this failure mode because it never generates that intervening shorthand text to anchor on. Fixed by instructing the signature to write the fully-expanded number in the reasoning itself, so the structured step has nothing left to get wrong. [`eval/dspy_extractor.py`](eval/dspy_extractor.py)

- **A claim could be marked wrong for being correct — at the time it was made.** SEC filings restate: a later 10-K/A amendment can revise an XBRL figure after the original filing. Without tracking *when* a fact was filed relative to the claim's own source filing, the reconciler's restatement-handling logic (correctly preferring the most recent value when duplicates disagree) could compare an old, accurate claim against a *later* restatement and call it "inconsistent" — the claim was right when it was written, and the world's record of that period simply changed afterward. Fixed by threading the claim's own filing date (`as_of`) into the reconciler's fact-matching itself, not just its period-selection layer (which Phase 5 already handled). [`tools/reconciler.py`](tools/reconciler.py)

One limitation is still open and documented rather than hidden: the verification agent can distinguish "the source filing's own period" from "the next most recent one," but doesn't yet parse a claim's *stated* period text (e.g. distinguishing a paragraph's FY2026 figure from its FY2025 comparison in the same sentence) — so a claim about an explicitly non-current period can still resolve against the wrong one. Flagged in [`agents/resolver.py`](agents/resolver.py) as follow-up, not silently left broken. A fuller accounting of what's explicitly in and out of scope — maker/checker independence, idempotency, why doom-loop detection doesn't apply here — is in [`docs/robustness-and-scope.md`](docs/robustness-and-scope.md).

## Harness: budget and checkpointing

A real 10-K MD&A can run 100+ paragraphs, each needing its own LLM call. The coordinator enforces a `Budget` (max chunks / max LLM calls / max wall-clock seconds) and checkpoints progress after every chunk — a crash or an intentional stop loses at most the one chunk in flight, not the whole run. Verified against the live pipeline: a budget-interrupted run picks up exactly where it left off on the next call, without re-processing anything already done. [`agents/checkpoint.py`](agents/checkpoint.py)

## Structure

```
tools/    MCP servers — Filing Retriever, MD&A Extractor, Numerical Reconciler (Pillar 1)
eval/     DSPy signature, hand-labeled test set, baseline-vs-optimized harness (Pillar 2)
agents/   Coordinator + retrieval/extraction/verification agents, budget, checkpointing (Pillar 3)
data/     Local cache / checkpoints (gitignored)
tests/    117 tests — unit (always run) + 8 live-network/live-LLM (opt-in via -m)
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
pytest -v                 # 117 unit tests, no network/LLM required
pytest -v -m network       # + live SEC EDGAR checks
pytest -v -m llm           # + live LLM checks (requires LLM_BASE_URL reachable)
```

## Status

Phases 0–5 complete (scaffolding, all 3 Pillar-1 MCP tools, DSPy optimization, hierarchical orchestration with mock scenario tests). Phase 6 (end-to-end integration at scale) and Phase 7 (RLVR/GRPO fine-tuning, pending GPU compute) remain.
