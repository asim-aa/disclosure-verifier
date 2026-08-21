"""Runs eval/reconciler_audit.py's adversarial probe set as a permanent regression
guard, not just a one-off report. The false-CONSISTENT count is the number that
matters most before the Reconciler is ever trusted as an RLVR reward — a false
"consistent" doesn't just cost training signal, it actively teaches a policy that
a wrong answer was right.
"""

from eval.reconciler_audit import run_audit


def test_reconciler_audit_has_zero_false_consistent_results(capsys):
    summary = run_audit()
    capsys.readouterr()  # keep the audit's own printed report out of normal test output
    assert summary["false_consistent_count"] == 0, summary["false_consistent_cases"]


def test_reconciler_audit_matches_every_expected_verdict(capsys):
    summary = run_audit()
    capsys.readouterr()
    assert summary["correct"] == summary["total"]
