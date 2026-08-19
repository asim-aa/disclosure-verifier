"""DSPy claim-extraction module — the optimization target for Pillar 2. Same task
and output schema as the hand-written baseline, but expressed as a dspy.Signature
so a DSPy optimizer (e.g. BootstrapFewShot) can compile few-shot demonstrations
against the hand-labeled training set instead of relying on a hand-tuned prompt.
"""

import dspy

from eval.schema import ExtractedClaim


class ExtractClaims(dspy.Signature):
    """Extract discrete, checkable financial claims from a paragraph of a company's
    SEC filing. A claim is a specific assertion tying a number to a metric and a
    period. Ignore purely qualitative statements with no number attached."""

    paragraph: str = dspy.InputField()
    claims: list[ExtractedClaim] = dspy.OutputField()


class ClaimExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(ExtractClaims)

    def forward(self, paragraph: str):
        return self.extract(paragraph=paragraph)


def extract_claims_dspy(paragraph: str, extractor: ClaimExtractor | None = None) -> list[ExtractedClaim]:
    extractor = extractor or ClaimExtractor()
    result = extractor(paragraph=paragraph)
    return result.claims
