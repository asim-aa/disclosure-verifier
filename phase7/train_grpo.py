"""Phase 7: GRPO fine-tuning of a small open model on the reconciliation-reasoning
task, using the real Reconciler (via phase7/reward.py) as the RLVR reward.

Run on the GPU box only (needs the `phase7` extra — see pyproject.toml):
    uv run python -m phase7.train_grpo

Before running, build the dataset (phase7/build_dataset.py) — this script just
consumes phase7/data/{train,test}.jsonl.
"""

import json
from pathlib import Path

from datasets import Dataset

from phase7 import (
    _trl_import_shim,  # noqa: F401 - must patch sys.modules before trl is ever imported below
)
from phase7.prompts import to_chat_messages
from phase7.reward import compute_reward

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "phase7" / "data"
OUTPUT_DIR = REPO_ROOT / "phase7" / "outputs"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 2048
MAX_PROMPT_LENGTH = 768
MAX_COMPLETION_LENGTH = 768
LORA_RANK = 32

# Tuned down from TRL's default (8) for a single 24GB card — see phase7/README.md
# for the memory-vs-signal tradeoff this controls.
NUM_GENERATIONS = 6


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _to_hf_dataset(examples: list[dict], tokenizer) -> Dataset:
    prompts = [tokenizer.apply_chat_template(to_chat_messages(ex), tokenize=False, add_generation_prompt=True) for ex in examples]
    return Dataset.from_dict({
        "prompt": prompts,
        "gold_verdict": [ex["gold_verdict"] for ex in examples],
        "gold_reason_code": [ex["gold_reason_code"] for ex in examples],
    })


def reward_func(completions, gold_verdict, gold_reason_code, **kwargs) -> list[float]:
    """TRL's GRPOTrainer calls this with `completions` already expanded to
    NUM_GENERATIONS per prompt, and every non-consumed dataset column
    (gold_verdict, gold_reason_code) replicated in lockstep — so zipping them
    positionally is correct."""
    return [
        compute_reward(completion, gold, reason)
        for completion, gold, reason in zip(completions, gold_verdict, gold_reason_code)
    ]


def main() -> None:
    from trl import (
        GRPOConfig,
        GRPOTrainer,
    )
    from unsloth import FastLanguageModel  # kept out of module scope - GPU-only import

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        # fast_inference (vLLM-backed rollout generation) deliberately not used:
        # it needs `vllm` installed, which isn't in the phase7 extra — a heavy,
        # version-finicky dependency not worth the risk for a first training run.
        # Standard HF generation is slower per rollout but has no extra failure surface.
        max_lora_rank=LORA_RANK,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=2026,
    )
    # GRPOTrainer.__init__ unconditionally does model.warnings_issued[...] = True,
    # assuming the plain transformers.PreTrainedModel attribute — the PEFT/Unsloth
    # wrapping here doesn't initialize it, so it's missing rather than delegated.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    train_examples = _load_jsonl(DATA_DIR / "train.jsonl")
    train_dataset = _to_hf_dataset(train_examples, tokenizer)

    training_args = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        # Must make per_device_train_batch_size * gradient_accumulation_steps
        # (the "generation batch size" - how many prompts get their
        # completions generated before one optimizer step) evenly divisible by
        # num_generations - TRL enforces this and errors otherwise. Matching
        # it to NUM_GENERATIONS directly is the simplest way to satisfy it.
        gradient_accumulation_steps=NUM_GENERATIONS,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_train_epochs=1,
        # max_steps bounds this to a run that finishes in hours, not the many
        # hours a full epoch over 1308 examples would take at this
        # generations/completion-length setting on a single 4090 — still a
        # substantial amount of training signal (150 steps * 6 grad-accum * 6
        # generations = 3600 rollouts), just on a timescale that can actually
        # be monitored and reported on honestly. Overrides num_train_epochs
        # per HF Trainer convention (max_steps > 0 takes precedence).
        max_steps=150,
        save_steps=25,
        logging_steps=1,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    adapter_dir = OUTPUT_DIR / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved LoRA adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
