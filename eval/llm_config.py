"""Configures the DSPy LM from environment variables (see .env.example).

The backend is a self-hosted, OpenAI-compatible vLLM server, not a public API.
litellm's `openai/<model>` provider prefix strips the "openai/" segment before
sending the request — but this server's actual registered model id is the literal
string "openai/gpt-oss-20b" (confirmed via its /v1/models listing), so that prefix
would 404. `hosted_vllm/<model>` is litellm's provider for arbitrary self-hosted
OpenAI-compatible servers and passes the model id through unchanged.

gpt-oss-20b is a reasoning model: it spends tokens on an internal reasoning trace
before the final answer, so `max_tokens` needs real headroom or responses truncate
mid-thought with empty content — this is why the default here is generous.
"""

import os

import dspy
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MAX_TOKENS = 4000


def get_lm(max_tokens: int = DEFAULT_MAX_TOKENS) -> dspy.LM:
    base_url = os.environ["LLM_BASE_URL"]
    api_key = os.environ["LLM_API_KEY"]
    model = os.environ["LLM_MODEL"]
    timeout = float(os.environ.get("LLM_TIMEOUT_SECONDS", "120"))

    return dspy.LM(
        model=f"hosted_vllm/{model}",
        api_base=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def configure_dspy(max_tokens: int = DEFAULT_MAX_TOKENS) -> dspy.LM:
    lm = get_lm(max_tokens=max_tokens)
    dspy.configure(lm=lm)
    return lm
