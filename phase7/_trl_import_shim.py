"""Workaround for an upstream packaging gap, not anything about our own code:
trl 0.24's GRPOTrainer unconditionally imports two optional integrations
(llm_blender for a judge class we never use, weave for W&B-style tracing we
don't use either) that aren't declared as real dependencies and, in llm_blender's
case, are broken outright against a current `transformers` (it imports
`TRANSFORMERS_CACHE`, removed years ago). Installing them for real doesn't fix
it — llm_blender is unmaintained and incompatible regardless of version, and
trl's own `is_weave_available()` check appears to return a false positive
in this environment even when `weave` isn't installed.

Since neither integration is reachable from anything GRPO training actually
does (no LLM-judge callback, no W&B/weave logging configured), stub them out
in sys.modules before trl's grpo_trainer module is ever imported, rather than
chase a dependency chain for functionality this project doesn't use.

Import this module *before* `from trl import GRPOConfig, GRPOTrainer` anywhere
in phase7/ — see train_grpo.py and evaluate.py.
"""

import importlib.machinery
import sys
import types


def _stub(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = module
    return module


def install() -> None:
    _stub("llm_blender")

    weave = _stub("weave")
    weave.EvaluationLogger = object
    weave_trace = _stub("weave.trace")
    weave_trace_context = _stub("weave.trace.context")
    weave_trace_context.weave_client_context = object
    weave.trace = weave_trace


install()
