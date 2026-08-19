"""Hand-written baseline claim extractor — a plain prompt template and a direct LLM
call, with no DSPy machinery. This is Pillar 2's baseline: what a reasonably careful
engineer would write without prompt-optimization tooling, to measure the delta DSPy
actually buys.
"""

import json

from eval.llm_config import get_lm
from eval.schema import ExtractedClaim

BASELINE_PROMPT_TEMPLATE = """You are extracting checkable financial claims from a paragraph of a company's SEC filing.

A "claim" is a specific, checkable assertion about a financial metric: a number, tied to a period, tied to a metric name. Ignore qualitative statements with no number attached (e.g. "revenue increased due to strong demand" with no percentage or dollar figure is NOT a claim).

For each claim you find, extract:
- metric: the financial metric being described, using the words from the text (e.g. "revenue", "gross margin", "Data Center revenue")
- value: the numeric value claimed, as a plain number (e.g. 27 for "27%", 214400000000 for "$214.4 billion")
- value_unit: one of "USD", "percent", "bps"
- period: the time period being described, using the words from the text (e.g. "fiscal year 2026", "the third quarter"). Use an empty string if the paragraph doesn't state one.
- comparison_type: one of
    "absolute" (a value reported at one period, e.g. "revenue was $215.9 billion" — if a sentence gives two absolute values at two periods with no stated delta, e.g. "71.1% in fiscal 2026 from 75.0% in fiscal 2025", that is TWO absolute claims, not a computed change; never compute a delta yourself)
    "growth_pct" (an explicit percent change vs. a prior period, e.g. "increased 27%")
    "absolute_change" (an explicit dollar-amount change vs. a prior period, e.g. "increased $50.1 billion" — note "Revenue increased $50.1 billion or 18%" is TWO claims, one of each type)
    "bps_change" (an explicit basis-point or percentage-point change in a ratio vs. a prior period, e.g. "margin expanded 200 bps")
- quote: the exact verbatim span of text supporting this claim

Return ONLY a JSON object of the form {{"claims": [...]}} with no other text. If there are no checkable claims, return {{"claims": []}}.

Paragraph:
\"\"\"
{paragraph}
\"\"\"
"""


def extract_claims_baseline(paragraph: str) -> list[ExtractedClaim]:
    lm = get_lm()
    prompt = BASELINE_PROMPT_TEMPLATE.format(paragraph=paragraph)
    response = lm(messages=[{"role": "user", "content": prompt}])[0]
    # gpt-oss is a reasoning model: dspy.LM returns {"text": ..., "reasoning_content":
    # ...} rather than a plain string in that case.
    text = response["text"] if isinstance(response, dict) else response

    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.removeprefix("json")
    text = text.strip()

    data = json.loads(text)
    return [ExtractedClaim(**c) for c in data.get("claims", [])]
