"""Runs eval/reconciler_audit.py's adversarial probe set as a permanent regression
guard, not just a one-off report. The false-CONSISTENT count is the number that
matters most before the Reconciler is ever trusted as an RLVR reward — a false
"consistent" doesn't just cost training signal, it actively teaches a policy that
a wrong answer was right.
"""

import pytest

from eval.reconciler_audit import (
    precision_ceiling,
    reconciler_recall_and_false_positive_rate,
    run_audit,
)


def test_reconciler_audit_has_zero_false_consistent_results(capsys):
    summary = run_audit()
    capsys.readouterr()  # keep the audit's own printed report out of normal test output
    assert summary["false_consistent_count"] == 0, summary["false_consistent_cases"]


def test_reconciler_audit_matches_every_expected_verdict(capsys):
    summary = run_audit()
    capsys.readouterr()
    assert summary["correct"] == summary["total"]


def test_reconciler_recall_is_perfect_on_the_known_good_cases():
    stats = reconciler_recall_and_false_positive_rate()
    assert stats["recall"] == 1.0
    assert stats["n_should_be_consistent"] > 0


def test_reconciler_false_positive_rate_is_zero_with_a_rule_of_three_bound():
    stats = reconciler_recall_and_false_positive_rate()
    assert stats["false_positive_rate"] == 0.0
    assert stats["false_positive_rate_upper_bound_95"] == pytest.approx(3.0 / stats["n_should_not_be_consistent"])


def test_precision_ceiling_formula_matches_its_own_definition():
    # Pr(correct|accepted) = p*r / (p*r + (1-p)*f) — a hand-computable sanity check
    # independent of the Reconciler's actual audit numbers.
    assert precision_ceiling(p=0.8, r=1.0, f=0.0) == 1.0
    assert precision_ceiling(p=0.8, r=0.5, f=0.5) == pytest.approx((0.8 * 0.5) / (0.8 * 0.5 + 0.2 * 0.5))


def test_run_audit_reports_the_precision_ceiling(capsys):
    summary = run_audit()
    capsys.readouterr()
    assert summary["precision_ceiling"] == pytest.approx(1.0)
