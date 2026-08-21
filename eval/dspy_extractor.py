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

    When your reasoning mentions a dollar amount, always write out the fully
    expanded number there too (e.g. "$215.9 billion" -> 215900000000), not the
    abbreviated form — the structured `value` field must match a number that
    already appears fully expanded in your reasoning, not one you expand only
    at the last step."""

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
