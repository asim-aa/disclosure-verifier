"""Builds the Phase 7 RLVR dataset entirely offline, from cached XBRL facts
already on disk (data/cache/companyfacts_*.json) — no network calls.

Ground truth is never hand-labeled: every example's gold_verdict/gold_reason_code
comes from literally calling tools.reconciler.reconcile() against real (or
deliberately constructed) FinancialFacts, so it's correct by construction. This
also sidesteps a real problem with eval/labeled_claims.jsonl for this purpose:
most of its gold claims are segment-level metrics (Azure revenue, Xbox revenue)
that agents/resolver.py can't map to a top-level XBRL concept by design — running
them through full resolution would produce a dataset that's mostly "unverifiable"
with little arithmetic-reasoning signal. Building directly from resolved
(concept, ticker, period) triples with the 14 concepts resolver.py *does* handle
avoids that entirely, and there's abundant real data to draw from (see the concept
coverage printed by this script).

Run: uv run python -m phase7.build_dataset
"""

import hashlib
import itertools
import json
import random
from datetime import date
from pathlib import Path

from agents.resolver import METRIC_TO_CONCEPTS
from phase7.schema import ReconciliationExample
from tools.reconciler import reconcile
from tools.schema import Claim, FinancialFact
from tools.xbrl_parser import parse_company_facts

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "cache"
OUT_DIR = REPO_ROOT / "phase7" / "data"

TICKERS = {"0000320193": "AAPL", "0000789019": "MSFT", "0001045810": "NVDA"}
RESOLVABLE_CONCEPTS = sorted({c for cands in METRIC_TO_CONCEPTS.values() for c in cands})

# (numerator, denominator) pairs that make a real, checkable ratio — mirrors the
# margin claims that actually appear in MD&A prose ("gross margin", "operating margin").
RATIO_PAIRS = [
    ("GrossProfit", "Revenues"),
    ("GrossProfit", "RevenueFromContractWithCustomerExcludingAssessedTax"),
    ("OperatingIncomeLoss", "Revenues"),
    ("OperatingIncomeLoss", "RevenueFromContractWithCustomerExcludingAssessedTax"),
]

PERTURBATION_KINDS = ["sign_flip", "magnitude_1000x", "magnitude_0001x", "near_miss", "large_miss"]


def _load_facts_by_ticker() -> dict[str, list[FinancialFact]]:
    out = {}
    for cik, ticker in TICKERS.items():
        raw = json.loads((CACHE_DIR / f"companyfacts_{cik}.json").read_text())["data"]
        out[ticker] = parse_company_facts(raw, ticker, concepts=RESOLVABLE_CONCEPTS)
    return out


def _by_concept(facts: list[FinancialFact]) -> dict[str, list[FinancialFact]]:
    by_concept: dict[str, list[FinancialFact]] = {}
    for f in facts:
        by_concept.setdefault(f.concept, []).append(f)
    for concept_facts in by_concept.values():
        concept_facts.sort(key=lambda f: f.period_end)
    return by_concept


def _dedup_by_period(facts: list[FinancialFact]) -> list[FinancialFact]:
    """One fact per distinct (period_start, period_end) — picks the first filed,
    matching the real _find_fact tie-break, so example generation doesn't stumble
    into the "ambiguous period" or "which restatement" cases this dataset isn't
    targeting."""
    best: dict[tuple, FinancialFact] = {}
    for f in facts:
        key = (f.period_start, f.period_end)
        if key not in best or (f.filed, f.accession_number) < (best[key].filed, best[key].accession_number):
            best[key] = f
    return sorted(best.values(), key=lambda f: f.period_end)


def _period_length_days(fact: FinancialFact) -> int | None:
    if fact.period_start is None:
        return None
    return (date.fromisoformat(fact.period_end) - date.fromisoformat(fact.period_start)).days


def _same_period_length(a: FinancialFact, b: FinancialFact, tolerance_days: int = 20) -> bool:
    """A company often reports both a single quarter and a cumulative
    year-to-date figure with the *same* period_end — comparing across those
    (e.g. Q3-only vs. 9-months-cumulative) isn't a real period-over-period
    change and produces a confusing, semantically-void training example.
    Requiring comparable period lengths keeps pairs to genuine
    quarter-vs-quarter or annual-vs-annual comparisons."""
    a_len, b_len = _period_length_days(a), _period_length_days(b)
    if a_len is None or b_len is None:
        return False
    return abs(a_len - b_len) <= tolerance_days


def _dominant_period_length_facts(facts: list[FinancialFact]) -> list[FinancialFact]:
    """Restricts to one period-length class (~quarterly, <=120 days, vs.
    ~annual/YTD-cumulative, >120 days) — whichever has more facts — so a later
    period_end-keyed lookup can't silently collide two facts that happen to
    share an end date but measure different-length periods (see
    _same_period_length)."""
    quarterly = [f for f in facts if (_period_length_days(f) or 0) <= 120]
    annual = [f for f in facts if (_period_length_days(f) or 0) > 120]
    return quarterly if len(quarterly) >= len(annual) else annual


def _adjacent_comparable_pairs(concept_facts: list[FinancialFact]) -> list[tuple[FinancialFact, FinancialFact]]:
    """(prior, current) pairs with a genuine time gap and comparable period
    length — see _same_period_length."""
    pairs = []
    for i, current in enumerate(concept_facts):
        for prior in reversed(concept_facts[:i]):
            if prior.period_end == current.period_end:
                continue
            if _same_period_length(prior, current):
                pairs.append((prior, current))
                break  # nearest comparable prior period only
    return pairs


def _perturb(comparison_type: str, computed_value: float, tolerance: float, kind: str, denom: float | None = None) -> float:
    scale = denom if denom is not None else (abs(computed_value) if computed_value != 0 else 1.0)
    additive = comparison_type in ("growth_pct", "bps_change")

    if kind == "sign_flip":
        return -computed_value if computed_value != 0 else scale
    if kind == "magnitude_1000x":
        return computed_value * 1000
    if kind == "magnitude_0001x":
        return computed_value * 0.001
    if kind == "near_miss":
        offset = tolerance * 1.5 if additive else scale * tolerance * 1.5
        return computed_value + offset
    if kind == "large_miss":
        offset = tolerance * 6 if additive else scale * tolerance * 6
        return computed_value + offset
    raise ValueError(f"unknown perturbation kind: {kind}")


def _make_example(
    id_: str, ticker: str, comparison_type: str, claimed_value: float, unit: str,
    current_fact: FinancialFact | None, comparison_fact: FinancialFact | None,
    denominator_metric: str | None, den_current: FinancialFact | None, den_comparison: FinancialFact | None,
    metric: str, period_end: str, comparison_period_end: str | None, source: str, note: str = "",
) -> ReconciliationExample:
    claim = Claim(
        ticker=ticker, metric=metric, comparison_type=comparison_type, claimed_value=claimed_value,
        period_start=current_fact.period_start if current_fact else None, period_end=period_end,
        comparison_period_start=comparison_fact.period_start if comparison_fact else None,
        comparison_period_end=comparison_period_end,
        denominator_metric=denominator_metric, unit=unit,
    )
    minimal_facts = [f for f in (current_fact, comparison_fact, den_current, den_comparison) if f is not None]
    gold = reconcile(claim, minimal_facts)

    return ReconciliationExample(
        id=id_, ticker=ticker, metric=metric, comparison_type=comparison_type,
        claimed_value=claimed_value, unit=unit, tolerance=gold.tolerance,
        period_end=period_end, comparison_period_end=comparison_period_end,
        denominator_metric=denominator_metric,
        current_value=current_fact.value if current_fact else None, current_value_unit=unit,
        comparison_value=comparison_fact.value if comparison_fact else None,
        denominator_current_value=den_current.value if den_current else None,
        denominator_comparison_value=den_comparison.value if den_comparison else None,
        gold_verdict=gold.verdict, gold_reason_code=gold.reason_code,
        gold_computed_value=gold.computed_value, gold_difference=gold.difference,
        source=source, note=note,
    )


def build_examples(rng: random.Random) -> list[ReconciliationExample]:
    facts_by_ticker = _load_facts_by_ticker()
    examples: list[ReconciliationExample] = []
    counter = 0

    def next_id(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}-{counter}"

    for ticker, facts in facts_by_ticker.items():
        by_concept = {c: _dedup_by_period(fs) for c, fs in _by_concept(facts).items()}

        # ---- absolute ----
        for concept, concept_facts in by_concept.items():
            sample = rng.sample(concept_facts, min(6, len(concept_facts)))
            for f in sample:
                examples.append(_make_example(
                    next_id("abs"), ticker, "absolute", f.value, f.unit,
                    f, None, None, None, None, concept, f.period_end, None, source="real",
                ))
                # Two perturbations per real example, not all five — five
                # inconsistent claims for every one consistent claim would let a
                # degenerate "always guess inconsistent" policy score well
                # without reasoning at all (only false-*consistent* guesses are
                # penalized in the reward, not false-*inconsistent* ones).
                for kind in rng.sample(PERTURBATION_KINDS, 2):
                    perturbed = _perturb("absolute", f.value, 0.01, kind)
                    examples.append(_make_example(
                        next_id("abs-p"), ticker, "absolute", perturbed, f.unit,
                        f, None, None, None, None, concept, f.period_end, None,
                        source="perturbed", note=kind,
                    ))

        # ---- growth_pct / absolute_change (comparable-length periods, same concept) ----
        for concept, concept_facts in by_concept.items():
            pairs = _adjacent_comparable_pairs(concept_facts)
            for prior, current in rng.sample(pairs, min(3, len(pairs))):
                if prior.value == 0:
                    continue
                real_growth = (current.value - prior.value) / abs(prior.value) * 100
                real_change = current.value - prior.value
                change_denom = max(abs(current.value), abs(prior.value)) or 1.0

                examples.append(_make_example(
                    next_id("gpct"), ticker, "growth_pct", real_growth, current.unit,
                    current, prior, None, None, None, concept, current.period_end, prior.period_end, source="real",
                ))
                examples.append(_make_example(
                    next_id("achg"), ticker, "absolute_change", real_change, current.unit,
                    current, prior, None, None, None, concept, current.period_end, prior.period_end, source="real",
                ))
                for kind in rng.sample(PERTURBATION_KINDS, 2):
                    examples.append(_make_example(
                        next_id("gpct-p"), ticker, "growth_pct",
                        _perturb("growth_pct", real_growth, 1.0, kind),
                        current.unit, current, prior, None, None, None, concept,
                        current.period_end, prior.period_end, source="perturbed", note=kind,
                    ))
                for kind in rng.sample(PERTURBATION_KINDS, 2):
                    examples.append(_make_example(
                        next_id("achg-p"), ticker, "absolute_change",
                        _perturb("absolute_change", real_change, 0.01, kind, denom=change_denom),
                        current.unit, current, prior, None, None, None, concept,
                        current.period_end, prior.period_end, source="perturbed", note=kind,
                    ))

                # unverifiable: comparison period simply missing from retrieved data
                examples.append(_make_example(
                    next_id("gpct-missing"), ticker, "growth_pct", real_growth, current.unit,
                    current, None, None, None, None, concept, current.period_end, prior.period_end,
                    source="synthetic", note="comparison fact withheld",
                ))

        # ---- bps_change (ratio pairs, matching current/prior periods) ----
        for num_concept, den_concept in RATIO_PAIRS:
            if num_concept not in by_concept or den_concept not in by_concept:
                continue
            num_by_period = {f.period_end: f for f in _dominant_period_length_facts(by_concept[num_concept])}
            den_by_period = {f.period_end: f for f in _dominant_period_length_facts(by_concept[den_concept])}
            shared_periods = sorted(set(num_by_period) & set(den_by_period))
            pairs = list(itertools.pairwise(shared_periods))
            for prior_p, current_p in rng.sample(pairs, min(3, len(pairs))):
                num_cur, num_pri = num_by_period[current_p], num_by_period[prior_p]
                den_cur, den_pri = den_by_period[current_p], den_by_period[prior_p]
                if den_cur.value == 0 or den_pri.value == 0:
                    continue
                real_bps = (num_cur.value / den_cur.value - num_pri.value / den_pri.value) * 10_000

                examples.append(_make_example(
                    next_id("bps"), ticker, "bps_change", real_bps, num_cur.unit,
                    num_cur, num_pri, den_concept, den_cur, den_pri, num_concept,
                    current_p, prior_p, source="real",
                ))
                for kind in rng.sample(PERTURBATION_KINDS, 2):
                    examples.append(_make_example(
                        next_id("bps-p"), ticker, "bps_change",
                        _perturb("bps_change", real_bps, 50.0, kind),
                        num_cur.unit, num_cur, num_pri, den_concept, den_cur, den_pri, num_concept,
                        current_p, prior_p, source="perturbed", note=kind,
                    ))

                # unverifiable: zero denominator (synthetic — real filings essentially never
                # report zero revenue, so this edge case has to be constructed deliberately,
                # same as eval/reconciler_audit.py already does for its adversarial cases).
                zero_den_cur = FinancialFact(**{**den_cur.to_dict(), "value": 0.0})
                examples.append(_make_example(
                    next_id("bps-zero"), ticker, "bps_change", real_bps, num_cur.unit,
                    num_cur, num_pri, den_concept, zero_den_cur, den_pri, num_concept,
                    current_p, prior_p, source="synthetic", note="zero denominator",
                ))

    return examples


def _stable_bucket(example_id: str, n_buckets: int = 5) -> int:
    """A deterministic 0..n_buckets-1 bucket for an example id — unlike the
    builtin hash(), stable across processes and across Python runs."""
    return int(hashlib.md5(example_id.encode()).hexdigest(), 16) % n_buckets


def main() -> None:
    rng = random.Random(2026)
    examples = build_examples(rng)
    rng.shuffle(examples)

    # Held-out split by a deterministic hash of id, not index — stable across
    # regeneration even if generation order changes. Python's builtin hash()
    # is randomized per-process (PYTHONHASHSEED) for strings by default, so it
    # would silently produce a *different* split on every run — using md5
    # instead makes this reproducible.
    test = [e for e in examples if _stable_bucket(e.id) == 0]
    train = [e for e in examples if _stable_bucket(e.id) != 0]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("test", test)]:
        path = OUT_DIR / f"{name}.jsonl"
        with open(path, "w") as f:
            f.writelines(json.dumps(e.to_dict()) + "\n" for e in split)

    print(f"train: {len(train)}  test: {len(test)}  total: {len(examples)}")
    by_verdict: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for e in examples:
        by_verdict[e.gold_verdict] = by_verdict.get(e.gold_verdict, 0) + 1
        by_reason[e.gold_reason_code] = by_reason.get(e.gold_reason_code, 0) + 1
    print("by verdict:", by_verdict)
    print("by reason_code:", by_reason)


if __name__ == "__main__":
    main()
