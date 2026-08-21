"""Precision/recall scoring for claim extraction: matches predicted claims against
hand-labeled gold claims for one paragraph, then aggregates micro-averaged
precision/recall/F1 across a whole test set.

Metric/period are free text (an LLM might write "Revenue" vs "revenue", or phrase
a quote slightly differently), so matching can't require exact string equality —
only value, unit, and comparison_type must match exactly; metric text only needs
meaningful keyword overlap with the gold label.
"""

import re

from eval.schema import ExtractedClaim

VALUE_RELATIVE_TOLERANCE = 0.01  # 1%
VALUE_ABSOLUTE_TOLERANCE = 0.05  # for near-zero values (e.g. small % figures)

# Only pure grammatical stopwords — "revenue"/"expense" etc. are kept as real
# keywords, since dropping them made bare "Revenue" spuriously match ANY
# revenue-qualified metric ("Microsoft Cloud revenue", "LinkedIn revenue", ...).
_STOPWORDS = {"the", "a", "an", "of", "and", "or", "for", "to", "in", "on"}

METRIC_OVERLAP_THRESHOLD = 0.5


def _keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def _metric_matches(pred_metric: str, gold_metric: str) -> bool:
    """Jaccard-style overlap normalized by the larger keyword set, so a generic
    prediction ("Revenue") doesn't spuriously match a more specific gold label
    ("Microsoft Cloud revenue") just by sharing the word "revenue"."""
    pred_kw, gold_kw = _keywords(pred_metric), _keywords(gold_metric)
    if not pred_kw or not gold_kw:
        return pred_kw == gold_kw
    overlap = len(pred_kw & gold_kw) / max(len(pred_kw), len(gold_kw))
    return overlap >= METRIC_OVERLAP_THRESHOLD


def claims_match(pred: ExtractedClaim, gold: ExtractedClaim) -> bool:
    if pred.comparison_type != gold.comparison_type:
        return False
    if pred.value_unit != gold.value_unit:
        return False

    diff = abs(pred.value - gold.value)
    tolerance = max(VALUE_ABSOLUTE_TOLERANCE, abs(gold.value) * VALUE_RELATIVE_TOLERANCE)
    if diff > tolerance:
        return False

    return _metric_matches(pred.metric, gold.metric)


def score_example(predicted: list[ExtractedClaim], gold: list[ExtractedClaim]) -> tuple[int, int, int]:
    """Greedy one-to-one matching. Returns (true_positives, false_positives, false_negatives)."""
    unmatched_pred = list(predicted)
    tp = 0
    for g in gold:
        match = next((p for p in unmatched_pred if claims_match(p, g)), None)
        if match is not None:
            unmatched_pred.remove(match)
            tp += 1

    fp = len(unmatched_pred)
    fn = len(gold) - tp
    return tp, fp, fn


def precision_recall_f1(total_tp: int, total_fp: int, total_fn: int) -> dict:
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": total_tp, "fp": total_fp, "fn": total_fn}


def score_example_by_category(predicted: list[ExtractedClaim], gold: list[ExtractedClaim]) -> dict[str, tuple[int, int, int]]:
    """Same greedy matching as score_example, but broken out per comparison_type —
    a pooled F1 can hide a category that's doing badly inside an average that
    "smiles" while one stratum quietly fails (e.g. absolute_change might be much
    weaker than absolute while the pooled number looks fine). A false positive is
    attributed to the *predicted* claim's stated comparison_type (it has no gold
    match to categorize it by); true positives and false negatives are attributed
    to the *gold* claim's comparison_type."""
    unmatched_pred = list(predicted)
    by_category: dict[str, list[int]] = {}

    def bucket(category: str) -> list[int]:
        return by_category.setdefault(category, [0, 0, 0])

    for g in gold:
        match = next((p for p in unmatched_pred if claims_match(p, g)), None)
        if match is not None:
            unmatched_pred.remove(match)
            bucket(g.comparison_type)[0] += 1  # tp
        else:
            bucket(g.comparison_type)[2] += 1  # fn

    for p in unmatched_pred:
        bucket(p.comparison_type)[1] += 1  # fp

    return {category: tuple(counts) for category, counts in by_category.items()}


def _describe_claim(claim: ExtractedClaim) -> str:
    unit = {"USD": "", "percent": "%", "bps": " bps"}.get(claim.value_unit, f" {claim.value_unit}")
    return f"{claim.comparison_type} claim for '{claim.metric}' (~{claim.value:g}{unit})"


def feedback_for_example(predicted: list[ExtractedClaim], gold: list[ExtractedClaim]) -> str:
    """A human/model-readable diagnostic for one example — the "actionable side
    information" a reflective optimizer (e.g. dspy.GEPA) needs to improve a
    prompt, as opposed to a bare score it can't act on. Reuses score_example's
    own greedy matching so the feedback text and the scalar score it accompanies
    are always describing the same judgment, not two different ones."""
    unmatched_pred = list(predicted)
    missed: list[ExtractedClaim] = []
    for g in gold:
        match = next((p for p in unmatched_pred if claims_match(p, g)), None)
        if match is not None:
            unmatched_pred.remove(match)
        else:
            missed.append(g)
    extra = unmatched_pred  # predictions with no gold match

    tp = len(gold) - len(missed)
    parts = [f"{tp}/{len(gold)} gold claims correctly extracted."]
    if missed:
        parts.append("Missed: " + "; ".join(_describe_claim(c) for c in missed) + ".")
    if extra:
        parts.append("Extra (no matching gold claim): " + "; ".join(_describe_claim(c) for c in extra) + ".")
    if not missed and not extra:
        parts.append("Exact match — no missed or extra claims.")
    return " ".join(parts)


def merge_category_counts(
    total: dict[str, list[int]], new: dict[str, tuple[int, int, int]]
) -> None:
    """Accumulates score_example_by_category's output across a whole test set,
    in place, into `total` (a dict of category -> [tp, fp, fn])."""
    for category, (tp, fp, fn) in new.items():
        bucket = total.setdefault(category, [0, 0, 0])
        bucket[0] += tp
        bucket[1] += fp
        bucket[2] += fn
