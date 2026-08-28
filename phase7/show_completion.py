"""One-off: generate and print a single test example's full completion from a
given model — used to pull real before/after text for the results write-up.
Not part of the regular pipeline.

Run on the GPU box:
    uv run python -m phase7.show_completion --adapter none --id achg-435
    uv run python -m phase7.show_completion --adapter phase7/outputs/lora_adapter --id achg-435
"""

import argparse
import json
from pathlib import Path

from phase7.prompts import build_prompt, to_chat_messages

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "phase7" / "data"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="none")
    parser.add_argument("--id", required=True)
    args = parser.parse_args()

    from unsloth import FastLanguageModel

    examples = [json.loads(line) for line in (DATA_DIR / "test.jsonl").read_text().splitlines()]
    ex = next(e for e in examples if e["id"] == args.id)

    adapter_path = None if args.adapter == "none" else args.adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path or MODEL_NAME, max_seq_length=2048, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    prompt = tokenizer.apply_chat_template(to_chat_messages(ex), tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    completion = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print("=" * 70)
    print("PROMPT:")
    print(build_prompt(ex))
    print("=" * 70)
    print(f"GOLD: {ex['gold_verdict']} / {ex['gold_reason_code']}")
    print("=" * 70)
    print("COMPLETION:")
    print(completion)


if __name__ == "__main__":
    main()
