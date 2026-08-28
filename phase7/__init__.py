"""Phase 7: RLVR/GRPO fine-tuning. Trains a small open model to reason through
numerical reconciliation (the arithmetic Reconciler already does deterministically)
using the real Reconciler as the ground-truth reward oracle.

See docs/phase7-reward-design.md for the design decisions made before any of this
was built, and phase7/README.md for how to actually run it.
"""
