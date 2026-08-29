"""Phase B of the disclosure-inconsistency backtest: for a sample of Phase A's
real restatement fingerprints, fetches the *original* filing (the one later
amended) and runs real extraction over its MD&A, keeping only claims that
plausibly state the specific figure that was later restated.

Not every restated XBRL concept gets narrated in prose - a company's MD&A
might discuss revenue and margins in words while a balance-sheet line item
like Liabilities only ever appears in a table. This step measures how many of
Phase A's findings are even reachable this way, and produces the concrete
(original filing, claim, fingerprint) triples Phase C actually backtests.

Matching a claim to a fingerprint uses two signals together, not one:
  1. The claim's metric text must resolve (via agents.resolver.METRIC_TO_CONCEPTS)
     to the SAME concept the fingerprint restated - not a fuzzy text match,
     the same exact-mapping discipline the rest of this project uses.
  2. The claim's numeric value must be within 5% of the fingerprint's
     original value - the real signal that this prose claim IS the restated
     figure, not just some other claim about the same concept. 5% (not exact)
     because prose commonly rounds ("$1.8 billion" for $1,838,000,000).

Sampled by |original_value| descending, not randomly: prioritizes real,
substantial companies over the frequent penny-stock/shell-company noise in
Item 4.02 filings (SS Innovations, Vivic Corp, etc. dominate by *count* but
are a weak advertisement for whether this matters) - capped at
MAX_COMPANIES to bound LLM cost/time, one real extraction pass per filing.

Two real reliability gaps, found the hard way on the first full run (it
stalled for over an hour with no way to tell whether it was still working,
and would have lost all 18 already-found matches if killed) and fixed here:
  1. Matches are written to OUT_PATH after every filing, not only at the
     end - killing this script mid-run now loses at most the filing in
     flight, not the whole run.
  2. Each extraction call is wrapped in a hard wall-clock timeout
     (EXTRACTION_TIMEOUT_SECONDS), independent of whatever retry/timeout
     behavior dspy's LM client does internally - that internal timeout
     (LLM_TIMEOUT_SECONDS, default 120s, x3 retries) did not visibly bound
     the stall that happened, so this is a second, script-level backstop:
     a chunk that doesn't return in time is skipped and counted, not
     waited on indefinitely.

TARGET_CIKS optionally restricts the run to specific companies (by CIK,
zero-padded 10-digit string) - used to re-run just the companies already
confirmed to produce real matches, rather than re-scanning the full
MAX_COMPANIES list. Leave as None for the full scan.

Run: python -m research.match_prose_claims
"""

import concurrent.futures
import json
from collections import defaultdict
from pathlib import Path

import httpx

from agents.extraction_agent import RealExtractionAgent
from agents.resolver import METRIC_TO_CONCEPTS
from tools.edgar_client import EdgarClient
from tools.mdna_parser import MdnaNotFoundError, chunk_mdna
from tools.schema import FilingMeta

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

REPO_ROOT = Path(__file__).resolve().parent.parent
FINGERPRINTS_PATH = REPO_ROOT / "research" / "data" / "restatement_fingerprints.json"
OUT_PATH = REPO_ROOT / "research" / "data" / "phase_b_matches.json"

MAX_COMPANIES = 60
MAX_CHUNKS_PER_FILING = 60  # bounds LLM cost per filing - a real MD&A can run 100+ paragraphs
VALUE_MATCH_TOLERANCE = 0.05  # prose commonly rounds ("$1.8 billion" for $1,838,000,000)
EXTRACTION_TIMEOUT_SECONDS = 45  # hard per-chunk backstop - see module docstring

# Discover Financial Services and Rithm Capital Corp - the two companies that
# produced all 18 real matches before the first full run was killed after
# stalling past 2 hours. None = full MAX_COMPANIES scan.
TARGET_CIKS: list[str] | None = None

# metric text (lowercased) -> concept, inverted from METRIC_TO_CONCEPTS for a
# direct "does this concept have ANY known metric-text phrasing" check.
CONCEPT_TO_METRIC_TEXTS: dict[str, list[str]] = defaultdict(list)
for _metric_text, _concepts in METRIC_TO_CONCEPTS.items():
    for _c in _concepts:
        CONCEPT_TO_METRIC_TEXTS[_c].append(_metric_text)


def claim_matches_fingerprint(claim, fingerprint: dict) -> bool:
    metric_key = claim.metric.strip().lower()
    candidates = METRIC_TO_CONCEPTS.get(metric_key)
    if not candidates or fingerprint["concept"] not in candidates:
        return False

    claimed_value = claim.value
    # a "percent" claim (e.g. a rate stated as a percentage) can't be a
    # dollar-figure restatement match - only compare like-for-like.
    if claim.value_unit == "percent" and abs(claimed_value) <= 100 and abs(fingerprint["original_value"]) > 100:
        return False

    denom = abs(fingerprint["original_value"]) or 1.0
    relative_diff = abs(claimed_value - fingerprint["original_value"]) / denom
    return relative_diff <= VALUE_MATCH_TOLERANCE


def build_cik_to_ticker(user_agent: str) -> dict[str, str]:
    response = httpx.get(TICKER_MAP_URL, headers={"User-Agent": user_agent}, timeout=20.0)
    response.raise_for_status()
    mapping = response.json()
    return {f"{entry['cik_str']:010d}": entry["ticker"] for entry in mapping.values()}


def fetch_original_filing_chunks(client: EdgarClient, cik: str, ticker: str, fingerprint: dict):
    submissions = client.get_submissions(cik)
    recent = submissions["filings"]["recent"]
    n = len(recent["accessionNumber"])
    idx = next((i for i in range(n) if recent["accessionNumber"][i] == fingerprint["original_accn"]), None)
    if idx is None:
        return None  # accession not in the "recent" window of submissions.json - older filing, skip

    filing = FilingMeta(
        ticker=ticker,
        cik=cik,
        accession_number=recent["accessionNumber"][idx],
        form=recent["form"][idx],
        filing_date=recent["filingDate"][idx],
        report_date=recent["reportDate"][idx],
        primary_document=recent["primaryDocument"][idx],
        primary_doc_description=recent.get("primaryDocDescription", [""] * n)[idx],
    )
    html = client.get_document(filing.filing_url(), cache_key=f"doc_{filing.accession_number}")
    try:
        chunks = chunk_mdna(
            html, ticker=ticker, cik=cik, accession_number=filing.accession_number,
            form=filing.form, filing_date=filing.filing_date,
        )
    except MdnaNotFoundError:
        return None
    return chunks[:MAX_CHUNKS_PER_FILING]


def extract_with_timeout(extraction: RealExtractionAgent, text: str, timeout: float):
    """extraction.extract() wrapped in a hard wall-clock budget, independent of
    whatever timeout/retry behavior dspy's LM client does internally - see the
    module docstring for why this backstop exists. Raises TimeoutError (or
    whatever extract() itself raised) rather than blocking indefinitely; the
    underlying thread may still be running afterward (Python can't force-kill
    a blocked thread), but the main loop is freed to move on and save progress."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(extraction.extract, text)
        return future.result(timeout=timeout)


def save_matches(matches: list[dict]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(matches, indent=2))


def main() -> None:
    fingerprints = json.loads(FINGERPRINTS_PATH.read_text())
    by_cik: dict[str, list[dict]] = defaultdict(list)
    for f in fingerprints:
        by_cik[f["cik"]].append(f)

    if TARGET_CIKS is not None:
        companies = [(cik, by_cik[cik]) for cik in TARGET_CIKS if cik in by_cik]
    else:
        # one entry per company, keyed by its single largest-magnitude fingerprint,
        # so the MAX_COMPANIES cap prioritizes substantial real companies
        companies = sorted(by_cik.items(), key=lambda kv: -max(abs(f["original_value"]) for f in kv[1]))
        companies = companies[:MAX_COMPANIES]

    client = EdgarClient()
    cik_to_ticker = build_cik_to_ticker(client.user_agent)
    extraction = RealExtractionAgent()

    matches = []
    n_no_native_ticker = n_no_mdna = n_extraction_error = n_timeout = 0
    n_filings_tried = 0

    for cik, company_fingerprints in companies:
        # SEC's own company_tickers.json is NOT a complete registrant list
        # (confirmed directly: it's missing Discover Financial Services, a
        # major NYSE-listed company, entirely - only ~10,400 entries). Ticker
        # is purely a label on TextChunk here, not used by any lookup logic
        # in chunk_mdna/extract_mdna_paragraphs, so fall back to the CIK
        # rather than skip a real company over a missing label.
        ticker = cik_to_ticker.get(cik)
        if not ticker:
            n_no_native_ticker += 1
            ticker = f"CIK{cik}"

        # group this company's fingerprints by original_accn - fetch/extract
        # that filing once, check it against every fingerprint from it
        by_accn: dict[str, list[dict]] = defaultdict(list)
        for f in company_fingerprints:
            by_accn[f["original_accn"]].append(f)

        for accn, accn_fingerprints in by_accn.items():
            n_filings_tried += 1
            print(f"[{n_filings_tried}] {accn_fingerprints[0]['entity_name']} ({ticker}) "
                  f"{accn} - {len(accn_fingerprints)} fingerprint(s) to check", flush=True)
            try:
                chunks = fetch_original_filing_chunks(client, cik, ticker, accn_fingerprints[0])
            except Exception as exc:  # noqa: BLE001 - one filing's fetch/parse failure shouldn't kill the whole scan
                print(f"    fetch/parse error: {exc}", flush=True)
                n_extraction_error += 1
                continue
            if not chunks:
                n_no_mdna += 1
                continue

            for chunk in chunks:
                try:
                    claims = extract_with_timeout(extraction, chunk.text, EXTRACTION_TIMEOUT_SECONDS)
                except concurrent.futures.TimeoutError:
                    print(f"    TIMEOUT on chunk {chunk.chunk_index} (>{EXTRACTION_TIMEOUT_SECONDS}s) - skipped", flush=True)
                    n_timeout += 1
                    continue
                except Exception as exc:  # noqa: BLE001 - one bad LLM response shouldn't kill the whole scan
                    print(f"    extraction error on chunk {chunk.chunk_index}: {exc}", flush=True)
                    n_extraction_error += 1
                    continue
                for claim in claims:
                    for fp in accn_fingerprints:
                        if claim_matches_fingerprint(claim, fp):
                            print(f"    MATCH: {claim.metric!r} = {claim.value}{claim.value_unit} "
                                  f"-> {fp['concept']} (fingerprint original={fp['original_value']:,})", flush=True)
                            matches.append({
                                "fingerprint": fp,
                                "claim": {
                                    "metric": claim.metric, "value": claim.value,
                                    "value_unit": claim.value_unit, "period": claim.period,
                                    "comparison_type": claim.comparison_type, "quote": claim.quote,
                                },
                                "source_chunk_text": chunk.text,
                                "source_accession": accn,
                                "source_filing_date": chunk.filing_date,
                                "ticker": ticker,
                            })

            # incremental save after every filing - killing this script mid-run
            # now loses at most the filing in flight, not the whole run (see
            # module docstring: the first full run stalled for 1+ hour with
            # 18 matches sitting only in memory, unrecoverable when killed).
            save_matches(matches)

    print(f"\nTried {n_filings_tried} filings across {len(companies)} companies "
          f"({n_no_native_ticker} used a CIK fallback label (no ticker in SEC's own list), "
          f"{n_no_mdna} had no locatable MD&A, "
          f"{n_extraction_error} hit fetch/extraction errors, {n_timeout} timed out)")
    print(f"Found {len(matches)} claims in real MD&A prose matching a real restatement fingerprint")

    save_matches(matches)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
