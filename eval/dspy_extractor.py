"""DSPy claim-extraction module — the optimization target for Pillar 2. Same task
and output schema as the hand-written baseline, but expressed as a dspy.Signature
so a DSPy optimizer (e.g. BootstrapFewShot) can compile few-shot demonstrations
against the hand-labeled training set instead of relying on a hand-tuned prompt.

Uses dspy.ChainOfThought rather than dspy.Predict: it adds an explicit `reasoning`
field the model fills in before committing to `claims`, so the model reasons
through which spans of the paragraph are checkable claims — and which
comparison_type each one is — before it has to produce structured output. This
is a separate layer from gpt-oss-20b's own internal reasoning trace (visible as
`reasoning_content` in the raw API response, see eval/llm_config.py) — that's
the model thinking privately; ChainOfThought's `reasoning` field is a visible
part of the prompt/output DSPy can inspect and that few-shot demos can include.
"""

import dspy

from eval.schema import ExtractedClaim


class ExtractClaims(dspy.Signature):
    """Extract discrete, checkable financial claims from a paragraph of a company's
    SEC filing. A claim is a specific assertion tying a number to a metric and a
    period. Ignore purely qualitative statements with no number attached.

    A sentence stating two absolute values side by side with no computed delta
    ("Net income was $5.00 billion compared with $4.80 billion") is TWO absolute
    claims, not one — the FIRST value is the current-period figure, the SECOND is
    the prior-period comparison figure being cited for context. This applies no
    matter which connective word joins them ("from", "compared with", "compared
    to", "versus", "up from") — never drop the first value and keep only the
    second, and never compute or report a delta the text itself doesn't state.

    When your reasoning mentions a dollar amount, always write out the fully
    expanded number there too (e.g. "$215.9 billion" -> 215900000000), not the
    abbreviated form — the structured `value` field must match a number that
    already appears fully expanded in your reasoning, not one you expand only
    at the last step.

    A sentence stating a metric "was positively/negatively impacted by X%" (or
    "benefited/hurt by X%", "contributed X percentage points") by some named
    factor (foreign currency, an acquisition, a one-time item) is NOT a
    growth_pct claim for that metric — it states only that factor's partial
    contribution to some other, separately-stated total change, not the
    metric's own total movement. ("Total revenues ... was positively impacted
    by approximately one percent in foreign currency fluctuations" does NOT
    mean revenue grew 1% — it means currency effects contributed about 1
    percentage point to revenue's real growth, a completely different, larger
    number stated elsewhere or not at all in this text.) There is no
    checkable claim to extract from a bare factor-contribution sentence like
    this — skip it, the same as any other qualitative statement with no
    directly checkable metric value.

    "Increased/decreased/expanded/contracted by X percentage points" (or "X
    bps") is a bps_change claim, NEVER growth_pct — it states the ABSOLUTE
    change in a ratio that is itself already a percentage (a margin, a rate),
    not a RELATIVE percent change in a dollar figure. ("Total gross margin
    increased by 0.2 percentage points" means the margin ratio itself moved by
    0.2 points — e.g. 62.1% to 62.3% — not that gross profit grew 0.2%; those
    are wildly different numbers and checking the wrong one against real data
    always looks like a huge, false miss.) By contrast, plain "X percent" or
    "X%" with no "percentage points"/"points"/"bps" wording ("revenue grew
    14%") is growth_pct as usual — this rule only applies when the text
    itself uses point/bps language for the change amount.

    A period stated once early in the paragraph applies to every later claim
    in it too, not just the sentence that first stated it — carry it forward
    unless a later sentence states a different one of its own.

    A sentence stating an overall company-wide figure that ALSO names a
    segment, product, subsidiary, acquisition, or geography's own
    sub-contribution ("Revenues were $7.1 billion ... which includes revenues
    from Ansys of $756.6 million", "revenue decrease 22% ... in China ...
    excluding Ansys") is describing TWO different, separately-checkable
    things, not one. The bare metric name ("revenue") is ONLY for the
    overall, unqualified company-wide figure — never drop a segment/
    geography/subsidiary qualifier and let its number stand in for the
    overall one, even when the overall figure is easy to miss in a long,
    multi-clause sentence. The qualified sub-figure gets its OWN metric text
    that keeps the qualifier ("revenue from Ansys", "China revenue") - never
    the bare metric name for a number that the sentence itself scopes to one
    piece of the business."""

    paragraph: str = dspy.InputField()
    claims: list[ExtractedClaim] = dspy.OutputField()


class ClaimExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractClaims)

    def forward(self, paragraph: str):
        return self.extract(paragraph=paragraph)


def extract_claims_dspy(paragraph: str, extractor: ClaimExtractor | None = None) -> list[ExtractedClaim]:
    extractor = extractor or ClaimExtractor()
    result = extractor(paragraph=paragraph)
    return result.claims
