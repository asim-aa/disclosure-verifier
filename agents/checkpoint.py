"""Durable state for an in-progress Coordinator run — so a crash partway through
a long document (a real 10-K can have 100+ MD&A chunks, each an LLM call) loses
at most the one chunk in flight, not the whole run.

Checkpoints are keyed by (ticker, form_type, limit) and stamped with the
accession numbers of the filings actually retrieved. On resume, if the current
retrieval's accession numbers don't match the checkpoint's, the checkpoint is
treated as stale (the underlying data changed — a new filing appeared, or a
different one was retrieved) and discarded rather than resumed from, since
resuming against mismatched data would silently mix two different documents'
claims into one report.
"""

import json
from pathlib import Path

from agents.schema import Report, TraceEvent, VerifiedClaim
from eval.schema import ExtractedClaim
from tools.schema import Claim, ReconciliationResult, TextChunk

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "data" / "checkpoints"


def _checkpoint_path(ticker: str, form_type: str, limit: int, checkpoint_dir: Path) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{ticker}_{form_type}_{limit}.json"


def _claim_from_dict(d: dict) -> Claim:
    return Claim(**d)


def _reconciliation_from_dict(d: dict) -> ReconciliationResult:
    claim = _claim_from_dict(d["claim"])
    rest = {k: v for k, v in d.items() if k != "claim"}
    return ReconciliationResult(claim=claim, **rest)


def _verified_claim_from_dict(d: dict) -> VerifiedClaim:
    return VerifiedClaim(
        source=TextChunk(**d["source"]),
        extracted=ExtractedClaim(**d["extracted"]),
        verdict=d["verdict"],
        explanation=d["explanation"],
        citations=d["citations"],
        reconciliation=_reconciliation_from_dict(d["reconciliation"]) if d.get("reconciliation") else None,
    )


def save(
    ticker: str,
    form_type: str,
    limit: int,
    accession_numbers: list[str],
    processed_chunk_indices: list[int],
    report: Report,
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> None:
    data = {
        "ticker": ticker,
        "form_type": form_type,
        "limit": limit,
        "accession_numbers": sorted(set(accession_numbers)),
        "processed_chunk_indices": processed_chunk_indices,
        "verified_claims": [vc.to_dict() for vc in report.verified_claims],
        "trace": [e.to_dict() for e in report.trace],
    }
    _checkpoint_path(ticker, form_type, limit, checkpoint_dir).write_text(json.dumps(data))


def load(
    ticker: str,
    form_type: str,
    limit: int,
    current_accession_numbers: list[str],
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> dict | None:
    """Returns the checkpoint dict if one exists and matches the currently
    retrieved filings, else None (no checkpoint, or a stale one that was
    discarded — either way, the caller should start fresh)."""
    path = _checkpoint_path(ticker, form_type, limit, checkpoint_dir)
    if not path.exists():
        return None

    data = json.loads(path.read_text())
    if sorted(set(data["accession_numbers"])) != sorted(set(current_accession_numbers)):
        path.unlink()  # stale — the underlying filings changed since this checkpoint was written
        return None

    return {
        "processed_chunk_indices": set(data["processed_chunk_indices"]),
        "verified_claims": [_verified_claim_from_dict(d) for d in data["verified_claims"]],
        "trace": [TraceEvent(**e) for e in data["trace"]],
    }


def delete(ticker: str, form_type: str, limit: int, checkpoint_dir: Path = CHECKPOINT_DIR) -> None:
    path = _checkpoint_path(ticker, form_type, limit, checkpoint_dir)
    if path.exists():
        path.unlink()
