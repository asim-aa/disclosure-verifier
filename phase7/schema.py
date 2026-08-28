"""The shape of one RLVR training/eval example — everything the prompt template
needs to render, plus the ground truth (produced by literally calling
tools.reconciler.reconcile(), not hand-labeled) the reward function checks
completions against.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ReconciliationExample:
    id: str
    ticker: str
    metric: str
    comparison_type: str  # absolute | growth_pct | absolute_change | bps_change
    claimed_value: float
    unit: str
    tolerance: float
    period_end: str
    comparison_period_end: str | None
    denominator_metric: str | None

    # Values as presented to the model — None renders as "not found".
    current_value: float | None
    current_value_unit: str
    comparison_value: float | None
    denominator_current_value: float | None
    denominator_comparison_value: float | None

    # Ground truth, from the real reconcile() — see phase7/build_dataset.py.
    gold_verdict: str
    gold_reason_code: str
    gold_computed_value: float | None
    gold_difference: float | None

    source: str  # "real" | "perturbed" | "synthetic"
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
