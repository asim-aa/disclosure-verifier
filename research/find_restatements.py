"""Phase A of the disclosure-inconsistency backtest: finds real restatements in
SEC XBRL data, offline evidence-gathering with no LLM and no GPU involved.

The question this whole backtest exists to answer: does this project's
verification actually catch something real, not just synthetic test cases?
Phase A supplies the ground truth to test that against - real companies whose
previously-reported figures were later revised.

Method: rather than parsing 8-K Item 4.02 ("Non-Reliance on Previously Issued
Financial Statements") disclosures with NLP to figure out what was restated,
this goes straight to the source SEC's own XBRL company-facts data already
contains, exploited by tools/reconciler.py's existing bitemporal design: when
a company restates a prior period, the SEC's company-facts API keeps BOTH the
original and the corrected value as separate FinancialFact-equivalent records
for the same (concept, period_start, period_end), distinguished by `filed`
date and `accn`. Finding restatements is then just: group a company's facts by
(concept, period_start, period_end), and look for groups with more than one
distinct value.

Item 4.02 filings are still used, but only to build the candidate company
list (a company that disclosed a restatement is far more likely to have a
detectable value-diff in its XBRL history than a company picked at random) -
not to determine what was restated. The XBRL diff itself is the ground truth,
confirmed by construction rather than inferred from prose.

Restricted to concepts in agents.resolver.METRIC_TO_CONCEPTS - the ~33
concepts this project's resolver can actually check - so Phase B (matching
restated figures to real MD&A prose) isn't wasted on concepts nothing in this
pipeline could ever verify anyway.

A "later distinct value for the same period" is NOT by itself required to be
an amended filing (checked directly, not assumed): a first pass counted any
later-filed value as a restatement and found 42,200 "fingerprints" across 655
companies - a number that fell apart on inspection. 82% of them came from a
later *regular* 10-K/10-Q's comparative column showing a different number
than the original filing for that period, which is routinely just a
reclassification, a discontinued-operations restatement, or a segment
realignment - not an error being corrected. Restricting to fingerprints where
the later value specifically comes from an amended filing (10-K/A or 10-Q/A -
a company doesn't refile one of those casually) drops the count to a much
more defensible 13,827 fingerprints across 541 companies. That's the actual
definition used below, not the looser one.

Median 14 fingerprints per company, not 1 - checked directly rather than
treated as a red flag, since the whole point of this exercise is not taking a
convenient-looking number at face value either way. One real restatement
event routinely corrects several line items across several quarters at once:
the busiest company found (191 fingerprints) traces back to just 7 distinct
amendment filings - each one correcting ~20-55 concept/period combinations
in a single amendment, not 191 independent errors.

Run: python -m research.find_restatements
"""

import json
import time
from pathlib import Path

import httpx

from agents.resolver import METRIC_TO_CONCEPTS
from tools.edgar_client import EdgarClient

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "research" / "data" / "restatement_fingerprints.json"

FTS_URL = "https://efts.sec.gov/LATEST/search-index"
FTS_MIN_REQUEST_INTERVAL_SECONDS = 0.3  # separate throttle - efts.sec.gov, not data.sec.gov

RESOLVABLE_CONCEPTS = sorted({c for cands in METRIC_TO_CONCEPTS.values() for c in cands})

# A same-period value that differs by less than this (relative) is more likely
# XBRL tagging noise (e.g. a units/rounding quirk between two accession
# filings) than a real restatement - skip it rather than count it as a find.
MATERIALITY_THRESHOLD = 0.01

# How many years back to search Item 4.02 8-Ks for candidate companies.
SEARCH_YEARS = 3


def search_item_402_filings(user_agent: str, start_date: str, end_date: str) -> list[dict]:
    """Paginate EDGAR full-text search for 8-K filings mentioning 'Item 4.02'
    (the SEC's own trigger for a non-reliance/restatement disclosure) in the
    given date range. Returns one dict per hit: {cik, entity_name, file_date}."""
    results = []
    last_request = 0.0
    start = 0
    page_size = 100

    while True:
        elapsed = time.monotonic() - last_request
        if elapsed < FTS_MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(FTS_MIN_REQUEST_INTERVAL_SECONDS - elapsed)
        last_request = time.monotonic()

        # EDGAR's full-text search occasionally 500s transiently under normal
        # use (reproduced directly: the exact same query succeeded on a plain
        # retry) - a real failure this deep into pagination would otherwise
        # throw away every page already fetched, so retry a few times before
        # giving up for real.
        for attempt in range(4):
            response = httpx.get(
                FTS_URL,
                params={
                    "q": '"Item 4.02"',
                    "forms": "8-K",
                    "startdt": start_date,
                    "enddt": end_date,
                    "from": start,
                },
                headers={"User-Agent": user_agent},
                timeout=20.0,
            )
            if response.status_code < 500:
                break
            time.sleep(2 * (attempt + 1))
        response.raise_for_status()
        data = response.json()
        hits = data["hits"]["hits"]
        if not hits:
            break

        for h in hits:
            src = h["_source"]
            ciks = src.get("ciks") or []
            if not ciks:
                continue
            results.append({
                "cik": ciks[0],
                "entity_name": (src.get("display_names") or [""])[0],
                "file_date": src.get("file_date"),
            })

        start += page_size
        if start >= data["hits"]["total"]["value"]:
            break

    return results


AMENDED_FORMS = {"10-K/A", "10-Q/A"}


def find_restatement_fingerprints(cik: str, entity_name: str, facts: dict) -> list[dict]:
    """Scan one company's XBRL facts for (concept, period) groups where a
    later *amended* filing (10-K/A or 10-Q/A) reports a materially different
    value than what was originally filed for that same period - a real,
    in-the-data restatement, not inferred from prose and not just any later
    filing's comparative column disagreeing with an earlier one (see the
    module docstring for why that looser definition was tried first and
    rejected: 82% of what it found wasn't actually a correction). 'original'
    is the earliest-filed value for the period from a non-amended form;
    'restated' is the latest-filed value specifically from an amended form.
    A period can restate more than once; only these two endpoints are kept,
    which is what the backtest actually needs."""
    fingerprints = []
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    for concept in RESOLVABLE_CONCEPTS:
        concept_data = us_gaap.get(concept)
        if not concept_data:
            continue
        for unit, points in concept_data.get("units", {}).items():
            by_period: dict[tuple, list[dict]] = {}
            for p in points:
                key = (p.get("start"), p.get("end"))
                by_period.setdefault(key, []).append(p)

            for (start, end), points_for_period in by_period.items():
                if end is None:
                    continue

                amended_points = [p for p in points_for_period if p["form"] in AMENDED_FORMS]
                non_amended_points = [p for p in points_for_period if p["form"] not in AMENDED_FORMS]
                if not amended_points or not non_amended_points:
                    continue

                original = min(non_amended_points, key=lambda p: p["filed"])
                restated = max(amended_points, key=lambda p: p["filed"])
                if original["val"] == restated["val"]:
                    continue

                denom = abs(original["val"]) or 1.0
                relative_diff = abs(restated["val"] - original["val"]) / denom
                if relative_diff < MATERIALITY_THRESHOLD:
                    continue

                fingerprints.append({
                    "cik": cik,
                    "entity_name": entity_name,
                    "concept": concept,
                    "unit": unit,
                    "period_start": start,
                    "period_end": end,
                    "original_value": original["val"],
                    "original_filed": original["filed"],
                    "original_accn": original["accn"],
                    "original_form": original["form"],
                    "restated_value": restated["val"],
                    "restated_filed": restated["filed"],
                    "restated_accn": restated["accn"],
                    "restated_form": restated["form"],
                    "relative_diff": relative_diff,
                })

    return fingerprints


def main() -> None:
    client = EdgarClient()
    user_agent = client.user_agent

    end_date = "2024-12-31"
    start_date = "2022-01-01"
    print(f"Searching EDGAR full-text search for 8-K Item 4.02 filings, {start_date} to {end_date}...")
    candidates = search_item_402_filings(user_agent, start_date, end_date)
    unique_ciks = {c["cik"]: c["entity_name"] for c in candidates}
    print(f"Found {len(candidates)} Item 4.02 filings across {len(unique_ciks)} unique companies")

    all_fingerprints = []
    n_no_facts = 0
    n_errors = 0
    for i, (cik, entity_name) in enumerate(unique_ciks.items()):
        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(unique_ciks)} companies scanned, "
                  f"{len(all_fingerprints)} fingerprints so far")
        try:
            facts = client.get_company_facts(cik)
        except Exception:  # noqa: BLE001 - many CIKs from FTS won't have XBRL company-facts at all (foreign filers, shells) - skip, don't crash the scan
            n_errors += 1
            continue
        if not facts.get("facts", {}).get("us-gaap"):
            n_no_facts += 1
            continue

        fingerprints = find_restatement_fingerprints(cik, entity_name, facts)
        all_fingerprints.extend(fingerprints)

    print(f"\nScanned {len(unique_ciks)} companies: {n_no_facts} had no us-gaap facts, "
          f"{n_errors} errored (foreign filers / no XBRL / etc.)")
    print(f"Found {len(all_fingerprints)} restatement fingerprints "
          f"across {len({f['cik'] for f in all_fingerprints})} companies")

    by_concept: dict[str, int] = {}
    for f in all_fingerprints:
        by_concept[f["concept"]] = by_concept.get(f["concept"], 0) + 1
    print("By concept:", dict(sorted(by_concept.items(), key=lambda kv: -kv[1])))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(all_fingerprints, indent=2))
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
