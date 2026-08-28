"""Workaround for an upstream packaging gap, not anything about our own code:
`trl.trainer.grpo_trainer` (and modules it imports) unconditionally imports
several optional integrations this project never uses, each gated behind an
`is_X_available()` check — and on this training environment specifically,
those checks have repeatedly returned a false positive (package "found" via
`importlib.util.find_spec`, but not actually importable/usable) rather than
correctly reporting the package isn't really there. Confirmed empirically,
one at a time, as each one crashed a real training run in turn: `is_weave_available()`
(trl 0.24), then `is_vllm_available()` and `is_vllm_ascend_available()` (both
trl 0.18.0, both reachable from `trl.extras.vllm_client` AND from
`grpo_trainer.py`'s own `from vllm import LLM, SamplingParams` gate).

None of llm_blender, weave, vllm, vllm_ascend, or liger_kernel are reachable
from anything this project's training/evaluation actually does — no LLM-judge
callback, no W&B/weave logging, no `use_vllm=True` anywhere (see
train_grpo.py's `FastLanguageModel.from_pretrained` comment on why
`fast_inference` is deliberately off), no Liger fused loss configured.

Given the check function itself is what's unreliable here, patching the
`is_X_available()` functions directly (rather than faking every individual
downstream module those gates import) is the robust fix: it makes every gate
that reads them correctly skip the optional-integration branch, in trl's
`grpo_trainer.py` and in every module it imports, in one place, instead of
chasing each newly-discovered import site as it's individually hit. `weave`
and `llm_blender` are additionally stubbed directly, since they're imported
unconditionally by `trl.trainer.judges`/`trl.trainer.callbacks` at module
scope (not behind a same-module `is_X_available()` gate this shim patches).
`wandb`/`comet_ml` are stubbed too as cheap insurance against the same
false-positive pattern recurring for them, though not yet individually
confirmed to.

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

    _stub("wandb")
    _stub("comet_ml")

    # The robust fix: patch trl's own availability checks to correctly report
    # "not available" for integrations this project never uses, rather than
    # faking every module these checks gate. Import trl.import_utils directly
    # (not `import trl`) so this patches the checks before trl.trainer.* ever
    # reads them, without triggering trl's own top-level package import yet.
    import trl.import_utils as trl_import_utils

    trl_import_utils.is_vllm_available = lambda: False
    trl_import_utils.is_vllm_ascend_available = lambda: False
    trl_import_utils.is_liger_kernel_available = lambda: False


install()
