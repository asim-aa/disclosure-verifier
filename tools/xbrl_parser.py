"""Parses raw EDGAR JSON (submissions, company facts) into the internal schema."""

from typing import Optional

from tools.schema import FilingMeta, FinancialFact

DEFAULT_FORM_TYPES = ("10-K", "10-Q", "8-K")


def parse_filings(
    submissions: dict,
    ticker: str,
    form_types: tuple[str, ...] = DEFAULT_FORM_TYPES,
    limit: Optional[int] = None,
) -> list[FilingMeta]:
    """Turn the 'recent' filings block of a submissions payload into FilingMeta records,
    most recent first, optionally filtered to a set of form types."""
    cik = str(submissions["cik"]).zfill(10)
    recent = submissions["filings"]["recent"]

    n = len(recent["form"])
    filings = []
    for i in range(n):
        form = recent["form"][i]
        if form_types and form not in form_types:
            continue
        filings.append(
            FilingMeta(
                ticker=ticker.upper(),
                cik=cik,
                accession_number=recent["accessionNumber"][i],
                form=form,
                filing_date=recent["filingDate"][i],
                report_date=recent["reportDate"][i],
                primary_document=recent["primaryDocument"][i],
                primary_doc_description=recent["primaryDocDescription"][i],
            )
        )
        if limit and len(filings) >= limit:
            break

    return filings


def parse_company_facts(
    facts: dict,
    ticker: str,
    concepts: Optional[list[str]] = None,
    taxonomy: str = "us-gaap",
) -> list[FinancialFact]:
    """Flatten EDGAR's nested company-facts payload into one FinancialFact per reported
    (concept, period, unit) data point. Restrict to `concepts` (e.g. ["Revenues"]) to
    avoid materializing all ~500 reported concepts when only a few are needed."""
    cik = str(facts["cik"]).zfill(10)
    taxonomy_facts = facts.get("facts", {}).get(taxonomy, {})

    results: list[FinancialFact] = []
    for concept_name, concept_data in taxonomy_facts.items():
        if concepts and concept_name not in concepts:
            continue

        label = concept_data.get("label")
        for unit, entries in concept_data.get("units", {}).items():
            for entry in entries:
                results.append(
                    FinancialFact(
                        ticker=ticker.upper(),
                        cik=cik,
                        concept=concept_name,
                        label=label,
                        value=entry["val"],
                        unit=unit,
                        period_start=entry.get("start"),
                        period_end=entry["end"],
                        fiscal_year=entry.get("fy"),
                        fiscal_period=entry.get("fp"),
                        form=entry.get("form", ""),
                        filed=entry.get("filed", ""),
                        accession_number=entry.get("accn", ""),
                    )
                )

    return results


def list_available_concepts(facts: dict, taxonomy: str = "us-gaap") -> list[str]:
    """All XBRL concept names a company has reported under a taxonomy (default us-gaap)."""
    return sorted(facts.get("facts", {}).get(taxonomy, {}).keys())
