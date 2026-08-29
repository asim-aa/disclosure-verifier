# Robustness & scope

Explicit, checked answers to a battery of standard agent-harness robustness
questions — stated rather than silently skipped, per the principle that a
scoped-out risk is stronger evidence than an unaddressed one.

## Checked and confirmed correct by design

**Maker/checker independence.** The Verification Agent never sees the Extraction
Agent's reasoning — `agents/coordinator.py` calls `self.extraction.extract(chunk.text)`,
and `ExtractionAgent.extract()` returns only `.claims` (the structured output),
never DSPy's `.reasoning` field. A checker that inherited the maker's chain of
thought would share its blind spots along with its logic; this one only sees the
artifact, which is the correct shape for an independent check. Verified by
reading the actual call chain, not assumed.

**Idempotency.** All three MCP tools are read-only against SEC's public data or
pure functions over already-fetched data — the Filing Retriever and MD&A
Extractor issue GET requests against EDGAR (naturally idempotent, no server-side
state mutation), and the Numerical Reconciler is a pure function of its inputs.
A retried call — including one replayed by the checkpoint/resume harness after a
crash — either hits the disk cache (`tools/edgar_client.py`) or produces an
identical result from identical inputs. No idempotency-key scheme is needed
because there's no side effect to deduplicate.

**Sub-agent termination.** Each of the three sub-agents (`RetrievalAgent`,
`ExtractionAgent`, `VerificationAgent`) is a `Protocol` with a single synchronous
method that returns a concrete value or raises — there's no "silently doesn't
finish" state a synchronous Python function call can be in. Termination is
structurally guaranteed by the call convention, not something that needed adding.
In the options-formalism vocabulary (a sub-agent as ⟨I, π, β⟩ — an initiation
set, a policy, and a termination condition), β is exactly "returns or raises,"
checkable by inspection rather than by runtime monitoring.

**No growing context to reclaim.** `Coordinator.run()` processes one MD&A chunk
at a time and calls `ExtractionAgent.extract(chunk.text)` with a single
paragraph, no accumulated history — confirmed by reading the actual signature
(`agents/extraction_agent.py`) and the coordinator's loop (`agents/coordinator.py`),
not assumed. There is no session-length context window that grows across a
filing's run for compress/delegate/externalize to manage: each extraction call
is independently bounded, and the Reconciler is a pure arithmetic function with
no context at all. The Reclamation Trilemma (compress/delegate/externalize)
targets long-running, context-accumulating loops; this pipeline structurally
isn't one, so the pattern doesn't apply — checked against the actual call
signatures, not asserted from the architecture description alone.

**Checkpoint granularity makes resume idempotent by construction.** Reading
`agents/checkpoint.py` and `Coordinator.run()`'s loop together: a chunk's claims
are only added to `processed` and only checkpointed (`checkpoint_store.save`,
which serializes the *entire* accumulated `verified_claims` list) after every
claim in that chunk has been verified — never partway through. A crash mid-chunk
loses at most that one chunk's in-flight work; on resume, `load()` restores the
last fully-committed state and the loop re-processes exactly the unprocessed
chunks (`if chunk.chunk_index in processed: continue`). Since `extract()` and
`verify()` are effectively pure functions of their inputs (no side effects
beyond the return value), re-running a chunk that never got checkpointed
produces the same result, not a duplicate one. This is why no idempotency-key
scheme (a caller-minted UUID mapped to a cached result) is needed for the
resume path specifically: the chunk-level all-or-nothing commit already
prevents any double-effect, by construction rather than by an added key.

**Delegation interface completeness.** Checked whether anything a sub-agent
computes is silently dropped before it reaches the final `Report` — the risk
being that a sub-agent's report to its caller is a lossy summary, and whatever
it omits is gone for good. It isn't, currently: `VerificationOutcome` and
`VerifiedClaim` (`agents/schema.py`) both carry the *entire*
`ReconciliationResult` object (verdict, computed_value, difference, tolerance,
reason_code, all citations), not a trimmed subset, and `Coordinator.run()`
passes it straight through unchanged. There's no report-shaped bottleneck
between Verification and the final output today.

## Bitemporal correctness (implemented)

SEC filings restate: a 10-K/A amendment can revise an XBRL figure after the
original filing. A claim extracted from an *older* filing's MD&A describes that
filing's own contemporaneous numbers — comparing it against a *later* restatement
would produce a false "inconsistent" for a claim that was accurate when made.

`tools/reconciler.py`'s `_find_fact` now accepts an `as_of` cutoff (the claim's
own source filing date) and restricts fact lookups to data filed on or before it.
`agents/verification_agent.py` threads the same `as_of` used for period/concept
resolution all the way through to the reconciler's own fact-matching, closing a
gap where period selection was already bitemporally correct (Phase 5) but the
final value lookup wasn't. See `tests/test_reconciler.py`'s `as_of` test section
for the regression tests, including a live-numbers reproduction of the bug this
fixes.

The same gap existed a second time, independently, in `tools/numerical_reconciler.py`'s
MCP tool — found by auditing its docstring against the standard tool-contract
checklist (typed schema, honest error taxonomy, and a description a calling model
can act on correctly) and asking whether anything was left unsaid. `reconcile_claim`
never exposed an `as_of` parameter at all, so any caller invoking the tool directly
(bypassing the Coordinator) had no bitemporal protection even after the agent-path
fix above landed. Fixed the same way, with offline tests in
`tests/test_numerical_reconciler_tool.py`.

**Reconciler verdicts carry a typed `reason_code`.** Beyond the three verdicts
themselves, every `ReconciliationResult` now includes a machine-readable reason
(`match`, `near_miss`, `large_miss`, `missing_fact`, `ambiguous_period`,
`zero_denominator`, `missing_comparison_context`, `unsupported_comparison_type`
— see `tools/schema.py`) instead of only free-text `explanation`. This exists for
two reasons: it's usable as reward-shaping material for Phase 7 (a wrong verdict
from a hair's-width miss is a different training signal than one from a wildly
wrong number), and it enables structured error-analysis reporting today without
parsing prose.

**Precision-ceiling check.** `eval/reconciler_audit.py` also estimates the
Reconciler's own recall and false-positive rate from its case set and applies
`Pr(correct | accepted) = pr / (pr + (1-p)f)` — the formula stating that
verification precision, not extraction quality, bounds what a maker/checker loop
can deliver end-to-end. With `r = 1.000`, `f = 0.000` (0/8, ~0.375 95% upper bound
by the rule of three at this n), and the DSPy-optimized extraction precision
(`p = 0.763`), the ceiling comes out to 1.000: a zero-false-positive verifier means
every accepted claim is trustworthy regardless of upstream extraction quality, on
this test surface. Small n keeps this illustrative, not statistically tight — the
same caveat that applies to the DSPy noise-floor finding above.

## The four-exit taxonomy — only two of four apply here

The standard run-level stop-rule taxonomy names four exits: success, insolvency
(budget spent), futility (a stuck/doom loop detected), and deference (the system
concludes its own uncertainty and escalates to a human). Checked against this
project's actual outcomes rather than forcing all four to fit:

- **Success** and **insolvency** are real and already modeled — `Report.partial`
  and `Report.partial_reason` (set from `Budget.exceeded_reason`) are exactly
  this distinction today.
- **Futility** has no detector, deliberately — see "Doom-loop detection" below;
  there's no autonomous retry loop for one to fire on.
- **Deference** has no *run-level* analog — there's no human-escalation path for
  the Coordinator to invoke. But the idea shows up one level down: a per-claim
  `unverifiable` verdict *is* the system declining to force a verdict it doesn't
  have the data to support, which is deference's substance without its
  machinery. Worth naming explicitly rather than pretending a four-way taxonomy
  cleanly covers a run shape that only has two of its four exits.

## Explicitly out of scope (considered, not silently omitted)

**Sagas / compensating actions for irreversible effects.** Doesn't apply — all
three tools are read-only against SEC data with no external side effects to
compensate for. There is nothing to undo.

**Lethal-trifecta sandboxing (untrusted input + private data + external
communication).** Doesn't apply — MD&A prose is public filing text, not
adversarial injected content; the system holds no private data and makes no
outbound communication beyond read-only EDGAR requests.

**Doom-loop detection (`E[Δq] ≈ 0` over a sliding window).** Targets a failure
mode of a long-running, unsupervised multi-turn agent loop that keeps retrying
without making progress. This project doesn't have that loop shape — the
coordinator is a fixed-sequence pipeline (retrieval → extraction →
verification, each a bounded synchronous call per chunk), not an autonomous
loop that can get stuck retrying the same step. (Context-reclamation, the
sibling concern this pattern is usually paired with, is addressed separately
above with its own code-verified finding, not lumped in here by assumption.)
The `Budget`/checkpoint harness (`agents/checkpoint.py`) already covers the
actual risk this architecture has — an expensive run with no cost cap —
without needing stuck-detection machinery for a loop shape that isn't present.

**Progressive disclosure of tool descriptions.** Targets dynamic tool selection
(an agent choosing among many tools per turn, where irrelevant tool detail in
context is a real cost). The coordinator's routing is fixed-sequence, not
dynamic — there's no selection decision for progressive disclosure to improve.

**Confused-deputy / multi-tenant authorization hardening.** Single-user local
tool, no multi-tenant secrets or external callers to protect against.

**LLM-judge pathologies (position/verbosity/self-enhancement bias, disposition
drift, "the ruler bends while you measure with it").** Doesn't apply — the
Numerical Reconciler, the sole verifier in this pipeline, is a hard
programmatic verifier (arithmetic against XBRL facts), not a soft LLM-as-judge
component. There's no judge prompt in the loop for these failure modes to
attach to. Worth stating rather than leaving the reader to wonder whether it
was checked.

## Coverage: how much of a real filing is actually resolvable

`agents/resolver.py`'s `METRIC_TO_CONCEPTS` maps 39 distinct XBRL concepts (58
metric-text phrasings, counting aliases) to real, verified-present us-gaap
tags — a small, deliberately exact-match-only dictionary, not a fuzzy matcher
(see the module's own docstring for why fuzzy matching is actively dangerous
here). What that scope means in practice, measured directly rather than
estimated:

- **Against a typical company's actual reported data**: across 676 real
  companies' cached XBRL company-facts (the Phase A restatement-scan sample —
  a broad, non-cherry-picked set), the median company reports **325 distinct
  us-gaap concepts**. The resolver's 39 cover a small fraction of that by raw
  count, but they're concentrated in the concepts that actually get narrated
  in prose: of the 20 most commonly-reported concepts across all 676
  companies, the resolver covers 13 (65%) — revenue, net income, operating
  income, taxes, the core cash-flow lines, total assets/liabilities/equity.
  The 7 misses among that top 20 (`LiabilitiesAndStockholdersEquity`,
  `RetainedEarningsAccumulatedDeficit`, `AdditionalPaidInCapital`, ...) are
  almost universally balance-sheet roll-forward or footnote-table figures a
  company reports but essentially never states as an MD&A prose claim ("common
  stock par value was $X") — a gap that doesn't cost real coverage for this
  project's actual use case, not a gap being quietly ignored.
- **Against real hand-labeled claims** (the actual, non-hypothetical measure):
  of the 161 claims in `eval/labeled_claims.jsonl`, **39.8%** resolve to a
  known concept today; of the 59 claims in the genuinely fresh AMZN/AAPL
  holdout, **44.1%** do. The remainder splits into two different things, not
  one: a real data-source limit (segment/product-level claims — "Azure
  revenue," "iPhone net sales" — that SEC's company-facts API has no
  dimensional breakdown for at all, confirmed by inspecting the raw data, not
  assumed) and a genuinely closeable dictionary gap (a metric phrased in words
  the dictionary didn't have a key for, even though the underlying concept was
  already mapped — "sales" not "revenue," "provision for income taxes" not
  "income tax expense"). The 44.1% figure above already reflects closing 10
  of those gaps found by checking this exact list, up from 13.6% before —
  each new key confirmed present in real AAPL/MSFT/NVDA/AMZN data first, same
  discipline as every other entry in the dictionary.
- **A second real dictionary gap, found the same way**: [`docs/specificity-check-results.md`](specificity-check-results.md)'s tech-vertical control set produced 113 `unverifiable` claims out of 137 — a bucket that, like the one above, had never actually been inspected. [`research/unresolved_claims_audit.py`](../research/unresolved_claims_audit.py) tallied it by metric text and found 7 more real, closeable gaps (`operating profit`, `total revenue`, `total gross margin`, `diluted net income per share`, `cash provided by operations`, `common stock repurchase amount`, `interest and debt expense` — the last confirmed present in real TXN data before being added). Re-running the same 8-ticker audit after adding them: **113 → 102 unverifiable claims**, resolution rate on that control set moving from 17.5% to 25.5% — and, after also fixing two bugs the expanded coverage exposed (see [`docs/specificity-check-results.md`](specificity-check-results.md)'s corrected "Root cause 2"), the reconciler's own apparent false-positive rate on those newly-resolved claims landed statistically indistinguishable from the pre-expansion baseline (25.7% vs. 25%), not worse for the added coverage.

The honest summary: this resolver is not attempting full-taxonomy coverage,
and stating that as a number is more useful than leaving it implicit. It
covers the concepts that dominate MD&A prose in practice, and roughly 4 in 10
real claims resolve today — a number with a known, closeable half (dictionary
gaps) and a known, structural half (no segment data in the source API at any
level of effort here).

## Interpreting "inconsistent" — a known simplification

The Reconciler currently collapses every mismatch into one `inconsistent`
verdict, but a mismatch can mean at least three different things: a genuine
restatement (partially addressed by the bitemporal fix above), a definitional
mismatch (the claim's "revenue" and the resolved XBRL concept measure
subtly different things), or an actual error in the filing's prose. The current
system doesn't distinguish these in the verdict itself — only in the free-text
`explanation` field, which isn't machine-categorized. Flagged as a real
simplification for future error-analysis work, not fixed here: distinguishing
these would require classifying *why* a mismatch occurred, which is a genuinely
different (and harder) problem than detecting *that* one occurred.
