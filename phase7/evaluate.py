"""Evaluates a model (base zero-shot, or with a trained LoRA adapter) on the
Phase 7 held-out test set — the same baseline-vs-optimized rigor Pillar 2's
eval/run_comparison.py applies, now for Pillar 4: run before AND after training,
report both, and don't just assert the trained one is better.

Run on the GPU box:
    uv run python -m phase7.evaluate --adapter none                       # base model
    uv run python -m phase7.evaluate --adapter phase7/outputs/lora_adapter # trained
"""

import argparse
import json
import math
from pathlib import Path

from phase7.prompts import to_chat_messages
from phase7.reward import compute_reward, parse_verdict

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "phase7" / "data"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 512


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _load_model(adapter_path: str | None):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path or MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _generate(model, tokenizer, examples: list[dict]) -> list[str]:
    completions = []
    for ex in examples:
        prompt = tokenizer.apply_chat_template(to_chat_messages(ex), tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        completions.append(text)
    return completions


def _noise_floor_half_width(p: float, n: int) -> float:
    if n == 0:
        return float("nan")
    return 1.96 * math.sqrt(p * (1 - p) / n)


def evaluate(model, tokenizer, examples: list[dict]) -> dict:
    completions = _generate(model, tokenizer, examples)

    n_correct = 0
    n_false_consistent = 0
    n_format_failure = 0
    rewards = []
    by_comparison_type: dict[str, list[int]] = {}  # [correct, total]

    for ex, completion in zip(examples, completions):
        predicted = parse_verdict(completion)
        gold = ex["gold_verdict"]
        reward = compute_reward(completion, gold, ex["gold_reason_code"])
        rewards.append(reward)

        bucket = by_comparison_type.setdefault(ex["comparison_type"], [0, 0])
        bucket[1] += 1
        if predicted == gold:
            n_correct += 1
            bucket[0] += 1
        if predicted is None:
            n_format_failure += 1
        if predicted == "consistent" and gold != "consistent":
            n_false_consistent += 1

    n = len(examples)
    accuracy = n_correct / n if n else float("nan")
    return {
        "n": n,
        "accuracy": accuracy,
        "noise_floor_half_width_95": _noise_floor_half_width(accuracy, n),
        "mean_reward": sum(rewards) / n if n else float("nan"),
        "n_false_consistent": n_false_consistent,
        "false_consistent_rate": n_false_consistent / n if n else float("nan"),
        "n_format_failure": n_format_failure,
        "by_comparison_type": {
            ct: {"correct": c, "total": t, "accuracy": c / t if t else float("nan")}
            for ct, (c, t) in by_comparison_type.items()
        },
    }


def print_report(label: str, result: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"n={result['n']}  accuracy={result['accuracy']:.3f} "
          f"(+/-{result['noise_floor_half_width_95']:.3f} 95%)  mean_reward={result['mean_reward']:.3f}")
    print(f"false-CONSISTENT rate (dangerous case): {result['false_consistent_rate']:.3f} "
          f"({result['n_false_consistent']}/{result['n']})")
    print(f"format failures (no parseable VERDICT line): {result['n_format_failure']}/{result['n']}")
    print("by comparison_type:")
    for ct, stats in sorted(result["by_comparison_type"].items()):
        print(f"  {ct:<18} accuracy={stats['accuracy']:.3f}  ({stats['correct']}/{stats['total']})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="none", help="Path to a saved LoRA adapter, or 'none' for the base model.")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    adapter_path = None if args.adapter == "none" else args.adapter
    label = args.label or ("base (zero-shot)" if adapter_path is None else f"trained ({adapter_path})")

    model, tokenizer = _load_model(adapter_path)
    test_examples = _load_jsonl(DATA_DIR / "test.jsonl")
    result = evaluate(model, tokenizer, test_examples)
    print_report(label, result)


if __name__ == "__main__":
    main()
