"""Extraction agent: the one stage that genuinely needs the LLM — turns a
paragraph of prose into structured ExtractedClaims via Phase 4's DSPy module.
"""

from pathlib import Path
from typing import Protocol

from eval.dspy_extractor import ClaimExtractor
from eval.llm_config import configure_dspy
from eval.schema import ExtractedClaim

REPO_ROOT = Path(__file__).resolve().parent.parent
OPTIMIZED_EXTRACTOR_PATH = REPO_ROOT / "eval" / "optimized_extractor.json"


class ExtractionAgent(Protocol):
    def extract(self, paragraph: str) -> list[ExtractedClaim]: ...


class RealExtractionAgent:
    """Loads the DSPy-optimized extractor (eval/compile_and_save.py's output) if
    available, so it doesn't recompile few-shot demos on every run; falls back to
    the zero-shot signature if no compiled state has been saved yet."""

    def __init__(self, configure_lm: bool = True):
        if configure_lm:
            configure_dspy()
        self.extractor = ClaimExtractor()
        if OPTIMIZED_EXTRACTOR_PATH.exists():
            self.extractor.load(str(OPTIMIZED_EXTRACTOR_PATH))

    def extract(self, paragraph: str) -> list[ExtractedClaim]:
        return self.extractor(paragraph=paragraph).claims


class MockExtractionAgent:
    """Maps paragraph text to a canned list of claims, fixed at construction time.
    Unrecognized paragraphs return an empty list (not an error) — same behavior a
    real extractor would have on a paragraph with no checkable claims."""

    def __init__(self, canned: dict[str, list[ExtractedClaim]]):
        self._canned = canned

    def extract(self, paragraph: str) -> list[ExtractedClaim]:
        return list(self._canned.get(paragraph, []))
