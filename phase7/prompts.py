"""The prompt template for Phase 7's reconciliation-reasoning task, shared by the
dataset builder (phase7/build_dataset.py) and the training script
(phase7/train_grpo.py) so what the model is trained on and what it's shown at
generation time are guaranteed to match.

Deliberately narrower than the full Verification Agent pipeline: the model is
handed already-resolved values (which concept, which periods, what the reported
numbers are), not asked to search for them. Concept/period resolution
(agents/resolver.py) is deterministic code, not a reasoning task — this isolates
the one part that's genuinely a capability question for a small model: does it
reliably reason through the *arithmetic* of reconciliation, given clean inputs.
"""

from __future__ import annotations

_UNIT_LABEL = {"USD": "USD", "percent": "%", "bps": "basis points", "pure": "(fraction)"}


def _fmt_value(value: float | None, unit: str) -> str:
    if value is None:
        return "not found in retrieved data"
    if unit == "USD":
        return f"{value:,.0f} USD"
    if unit == "pure":
        return f"{value:.6f}"
    return f"{value:g} {_UNIT_LABEL.get(unit, unit)}"


SYSTEM_PROMPT = (
    "You are a financial-disclosure verifier. You are given a quantitative claim "
    "and the actual reported data it should be checked against. Reason step by "
    "step through the arithmetic, then state your verdict."
)

_COMPARISON_EXPLANATION = {
    "absolute": "The claimed value should equal the reported value for the period, within tolerance.",
    "growth_pct": (
        "The claimed value is a percent change: "
        "(current - prior) / abs(prior) * 100, compared to the claimed percentage within tolerance."
    ),
    "absolute_change": (
        "The claimed value is a dollar change: current - prior, "
        "compared to the claimed amount within tolerance (relative to the larger of the two magnitudes)."
    ),
    "bps_change": (
        "The claimed value is a change in a ratio (metric / denominator_metric), in basis points: "
        "(current_ratio - prior_ratio) * 10000, compared to the claimed bps within tolerance."
    ),
}


# The claimed_value's own display unit is implied by comparison_type, not by
# example['unit'] — that field is the *underlying reported fact's* unit (for
# fact-lookup purposes, mirroring tools.schema.Claim.unit), which is always a
# dollar amount even for a growth_pct or bps_change claim whose claimed_value is
# a percentage or a basis-point count, not a dollar figure.
_CLAIMED_VALUE_UNIT = {
    "absolute": None,  # use example['unit'] as-is
    "absolute_change": None,
    "growth_pct": "percent",
    "bps_change": "bps",
}


def build_prompt(example: dict) -> str:
    """example: a dict shaped like phase7.schema.ReconciliationExample.to_dict()."""
    claimed_value_unit = _CLAIMED_VALUE_UNIT[example["comparison_type"]] or example["unit"]
    lines = [
        f"Claim type: {example['comparison_type']}",
        f"Metric: {example['metric']}",
        f"Claimed value: {_fmt_value(example['claimed_value'], claimed_value_unit)}",
        f"Tolerance: {example['tolerance']}",
        "",
        _COMPARISON_EXPLANATION[example["comparison_type"]],
        "",
        "Reported data:",
        (f"- {example['metric']} for period ending {example['period_end']}: "
        f"{_fmt_value(example['current_value'], example['current_value_unit'])}"),
    ]

    if example["comparison_type"] != "absolute":
        lines.append(
            f"- {example['metric']} for period ending {example['comparison_period_end']}: "
            f"{_fmt_value(example['comparison_value'], example['current_value_unit'])}"
        )

    if example["comparison_type"] == "bps_change":
        lines.append(
            f"- {example['denominator_metric']} for period ending {example['period_end']}: "
            f"{_fmt_value(example['denominator_current_value'], example['current_value_unit'])}"
        )
        lines.append(
            f"- {example['denominator_metric']} for period ending {example['comparison_period_end']}: "
            f"{_fmt_value(example['denominator_comparison_value'], example['current_value_unit'])}"
        )

    lines += [
        "",
        ("Reason through the calculation, then end your response with exactly one "
        "line in this exact form (no other text on that line):"),
        "VERDICT: <consistent|inconsistent|unverifiable>",
    ]
    return "\n".join(lines)


def to_chat_messages(example: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(example)},
    ]
