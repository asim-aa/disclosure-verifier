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

## Explicitly out of scope (considered, not silently omitted)

**Sagas / compensating actions for irreversible effects.** Doesn't apply — all
three tools are read-only against SEC data with no external side effects to
compensate for. There is nothing to undo.

**Lethal-trifecta sandboxing (untrusted input + private data + external
communication).** Doesn't apply — MD&A prose is public filing text, not
adversarial injected content; the system holds no private data and makes no
outbound communication beyond read-only EDGAR requests.

**Doom-loop detection and context-reclamation (compress/delegate/externalize).**
Both target failure modes of a long-running, unsupervised multi-turn agent loop.
This project doesn't have one — the coordinator is a fixed-sequence pipeline
(retrieval → extraction → verification, each a bounded synchronous call), not an
autonomous loop that can get stuck retrying, and each filing's run completes in
one pass rather than accumulating context across an open-ended session. The
`Budget`/checkpoint harness (`agents/checkpoint.py`) already covers the actual
risk this architecture has — an expensive run with no cost cap — without needing
stuck-detection machinery for a loop shape that isn't present.

**Progressive disclosure of tool descriptions.** Targets dynamic tool selection
(an agent choosing among many tools per turn, where irrelevant tool detail in
context is a real cost). The coordinator's routing is fixed-sequence, not
dynamic — there's no selection decision for progressive disclosure to improve.

**Confused-deputy / multi-tenant authorization hardening.** Single-user local
tool, no multi-tenant secrets or external callers to protect against.

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
