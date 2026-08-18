# The Disclosure Verifier

An agentic system for verifying quantitative claims in corporate financial disclosures (10-Ks, 10-Qs, 8-Ks, earnings transcripts) against SEC EDGAR's structured XBRL data — flagging inconsistencies with citations back to the source.

Capstone project for the SupportVectors AI Agents Bootcamp.

## Structure

```
tools/    MCP servers (filing retriever, transcript retriever, numerical reconciler)
agents/   Orchestration layer (coordinator, retrieval, extraction, verification agents)
eval/     DSPy signatures, labeled test sets, evaluation harnesses
data/     Local fixtures / cached filings (gitignored beyond .gitkeep)
tests/    pytest suite
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # then fill in EDGAR_USER_AGENT
```

## Test

```bash
pytest -v
```
