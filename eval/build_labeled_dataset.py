"""Builds eval/labeled_claims.jsonl — the hand-labeled ground-truth set for Phase 4.

Each entry's `claims` were written by reading the source paragraph directly (not by
running any LLM on it, to avoid circularity with what's being measured). Paragraphs
are drawn from real MD&A text already retrieved via the Phase 2 tool (MSFT and NVDA
10-K/10-Q — AAPL's current MD&A style states qualitative narrative without explicit
percentages, so it doesn't yield checkable claims; see the Phase 2/3 conversation).

Labeling rules (see eval/schema.py's module docstring for the full rationale):
  - comparison_type is only growth_pct / absolute_change / bps_change when the TEXT
    ITSELF states a computed change. Two absolute values at two periods in the same
    sentence, with no stated delta, are labeled as two separate "absolute" claims.
  - period is the literal period wording in that paragraph; empty string if the
    paragraph doesn't state one (many "highlights" bullets don't — the period is
    only established elsewhere in the filing, which paragraph-level extraction
    can't see).
  - Purely qualitative sentences, boilerplate accounting-policy text, statutory
    rates not tied to a specific reported period, and context-free table fragments
    are included as True Negatives (empty claims list) to measure precision, not
    just recall.

Run: python -m eval.build_labeled_dataset
"""

import json
from pathlib import Path

from eval.schema import ExtractedClaim

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "eval" / "labeled_claims.jsonl"

MSFT_10K = {"ticker": "MSFT", "cik": "0000789019", "form": "10-K", "accession_number": "0001193125-26-323660", "filing_date": "2026-07-29"}
MSFT_10Q = {"ticker": "MSFT", "cik": "0000789019", "form": "10-Q", "accession_number": "0001193125-26-191507", "filing_date": "2026-04-29"}
NVDA_10K = {"ticker": "NVDA", "cik": "0001045810", "form": "10-K", "accession_number": "0001045810-26-000021", "filing_date": "2026-02-25"}
NVDA_10Q = {"ticker": "NVDA", "cik": "0001045810", "form": "10-Q", "accession_number": "0001045810-26-000052", "filing_date": "2026-05-20"}


def C(metric, value, value_unit, period, comparison_type, quote):
    return {"metric": metric, "value": value, "value_unit": value_unit, "period": period, "comparison_type": comparison_type, "quote": quote}


EXAMPLES = [
    # ---------- MSFT 10-K: headline highlight bullets (no period stated) ----------
    dict(**MSFT_10K, chunk_index=5, text="Microsoft Cloud revenue increased 27% to $214.4 billion.", claims=[
        C("Microsoft Cloud revenue", 27, "percent", "", "growth_pct", "Microsoft Cloud revenue increased 27%"),
        C("Microsoft Cloud revenue", 214_400_000_000, "USD", "", "absolute", "Microsoft Cloud revenue increased 27% to $214.4 billion."),
    ]),
    dict(**MSFT_10K, chunk_index=6, text="Commercial remaining performance obligation increased 84% to $678 billion.", claims=[
        C("Commercial remaining performance obligation", 84, "percent", "", "growth_pct", "Commercial remaining performance obligation increased 84%"),
        C("Commercial remaining performance obligation", 678_000_000_000, "USD", "", "absolute", "Commercial remaining performance obligation increased 84% to $678 billion."),
    ]),
    dict(**MSFT_10K, chunk_index=7, text="Microsoft 365 Commercial cloud revenue increased 17%.", claims=[
        C("Microsoft 365 Commercial cloud revenue", 17, "percent", "", "growth_pct", "Microsoft 365 Commercial cloud revenue increased 17%."),
    ]),
    dict(**MSFT_10K, chunk_index=8, text="Microsoft 365 Consumer cloud revenue increased 28%.", claims=[
        C("Microsoft 365 Consumer cloud revenue", 28, "percent", "", "growth_pct", "Microsoft 365 Consumer cloud revenue increased 28%."),
    ]),
    dict(**MSFT_10K, chunk_index=9, text="LinkedIn revenue increased 11%.", claims=[
        C("LinkedIn revenue", 11, "percent", "", "growth_pct", "LinkedIn revenue increased 11%."),
    ]),
    dict(**MSFT_10K, chunk_index=11, text="Azure and other cloud services revenue increased 41%.", claims=[
        C("Azure and other cloud services revenue", 41, "percent", "", "growth_pct", "Azure and other cloud services revenue increased 41%."),
    ]),
    dict(**MSFT_10K, chunk_index=13, text="XBOX content and services revenue decreased 5%.", claims=[
        C("XBOX content and services revenue", -5, "percent", "", "growth_pct", "XBOX content and services revenue decreased 5%."),
    ]),
    dict(**MSFT_10K, chunk_index=14, text="Search advertising (formerly Search and news advertising) revenue excluding traffic acquisition costs increased 12%.", claims=[
        C("Search advertising revenue excluding traffic acquisition costs", 12, "percent", "", "growth_pct", "Search advertising (formerly Search and news advertising) revenue excluding traffic acquisition costs increased 12%."),
    ]),

    # ---------- MSFT 10-K: consolidated results of operations (period stated) ----------
    dict(**MSFT_10K, chunk_index=66, text="Revenue increased $50.1 billion or 18% driven by growth in Microsoft Cloud. Intelligent Cloud revenue increased driven by Azure. Productivity and Business Processes revenue increased driven by Microsoft 365 Commercial cloud. More Personal Computing revenue decreased driven by XBOX (formerly Gaming), offset in part by growth in Search advertising.", claims=[
        C("Revenue", 50_100_000_000, "USD", "", "absolute_change", "Revenue increased $50.1 billion"),
        C("Revenue", 18, "percent", "", "growth_pct", "Revenue increased $50.1 billion or 18%"),
    ]),
    dict(**MSFT_10K, chunk_index=67, text="Cost of revenue increased $18.5 billion or 21% driven by growth in Microsoft Cloud.", claims=[
        C("Cost of revenue", 18_500_000_000, "USD", "", "absolute_change", "Cost of revenue increased $18.5 billion"),
        C("Cost of revenue", 21, "percent", "", "growth_pct", "Cost of revenue increased $18.5 billion or 21%"),
    ]),
    dict(**MSFT_10K, chunk_index=68, text="Gross margin increased $31.6 billion or 16% with growth across each of our segments.", claims=[
        C("Gross margin", 31_600_000_000, "USD", "", "absolute_change", "Gross margin increased $31.6 billion"),
        C("Gross margin", 16, "percent", "", "growth_pct", "Gross margin increased $31.6 billion or 16%"),
    ]),
    dict(**MSFT_10K, chunk_index=70, text="Microsoft Cloud gross margin percentage decreased to 66% driven by continued investments in AI infrastructure and growing AI product usage, offset in part by efficiency gains in Azure and Microsoft 365 Commercial cloud.", claims=[
        C("Microsoft Cloud gross margin percentage", 66, "percent", "", "absolute", "Microsoft Cloud gross margin percentage decreased to 66%"),
    ]),
    dict(**MSFT_10K, chunk_index=71, text="Operating expenses increased $4.9 billion or 7% driven by continued investments in research and development compute capacity, AI talent, and data to support product development that benefits the entire portfolio, impairment and other related expenses in our XBOX business, investments in commercial sales, and higher Copilot advertising expenses.", claims=[
        C("Operating expenses", 4_900_000_000, "USD", "", "absolute_change", "Operating expenses increased $4.9 billion"),
        C("Operating expenses", 7, "percent", "", "growth_pct", "Operating expenses increased $4.9 billion or 7%"),
    ]),
    dict(**MSFT_10K, chunk_index=72, text="Operating income increased $26.7 billion or 21% driven by growth in Productivity and Business Processes and Intelligent Cloud.", claims=[
        C("Operating income", 26_700_000_000, "USD", "", "absolute_change", "Operating income increased $26.7 billion"),
        C("Operating income", 21, "percent", "", "growth_pct", "Operating income increased $26.7 billion or 21%"),
    ]),
    dict(**MSFT_10K, chunk_index=73, text="Revenue and operating income both included a favorable foreign currency impact of 2%.", claims=[
        C("Revenue foreign currency impact", 2, "percent", "", "absolute", "Revenue and operating income both included a favorable foreign currency impact of 2%."),
        C("Operating income foreign currency impact", 2, "percent", "", "absolute", "Revenue and operating income both included a favorable foreign currency impact of 2%."),
    ]),
    dict(**MSFT_10K, chunk_index=74, text="Current year net income and diluted EPS were positively impacted by net gains from investments in OpenAI, which resulted in an increase in net income and diluted EPS of $5.0 billion and $0.67, respectively. Prior year net income and diluted EPS were negatively impacted by net losses from investments in OpenAI, which resulted in a decrease in net income and diluted EPS of $3.6 billion and $0.49, respectively.", claims=[
        C("net income impact from OpenAI investment gains", 5_000_000_000, "USD", "current year", "absolute_change", "an increase in net income and diluted EPS of $5.0 billion and $0.67, respectively"),
        C("diluted EPS impact from OpenAI investment gains", 0.67, "USD", "current year", "absolute_change", "an increase in net income and diluted EPS of $5.0 billion and $0.67, respectively"),
        C("net income impact from OpenAI investment losses", -3_600_000_000, "USD", "prior year", "absolute_change", "a decrease in net income and diluted EPS of $3.6 billion and $0.49, respectively"),
        C("diluted EPS impact from OpenAI investment losses", -0.49, "USD", "prior year", "absolute_change", "a decrease in net income and diluted EPS of $3.6 billion and $0.49, respectively"),
    ]),

    # ---------- MSFT 10-K: segment detail (Productivity and Business Processes) ----------
    dict(**MSFT_10K, chunk_index=82, text="Microsoft 365 Commercial products and cloud services revenue increased $14.2 billion or 16%. Microsoft 365 Commercial cloud revenue grew 17% with growth in revenue per user driven by Microsoft 365 Copilot and Microsoft 365 E5. Microsoft 365 Commercial seats grew 6% driven by small and medium businesses and frontline worker offerings. Microsoft 365 Commercial products revenue grew 13% driven by growth in the Windows Commercial on-premises components of Microsoft 365 suite sales, as well as an increase in Office 2024 transactional purchasing.", claims=[
        C("Microsoft 365 Commercial products and cloud services revenue", 14_200_000_000, "USD", "", "absolute_change", "Microsoft 365 Commercial products and cloud services revenue increased $14.2 billion"),
        C("Microsoft 365 Commercial products and cloud services revenue", 16, "percent", "", "growth_pct", "Microsoft 365 Commercial products and cloud services revenue increased $14.2 billion or 16%"),
        C("Microsoft 365 Commercial cloud revenue", 17, "percent", "", "growth_pct", "Microsoft 365 Commercial cloud revenue grew 17%"),
        C("Microsoft 365 Commercial seats", 6, "percent", "", "growth_pct", "Microsoft 365 Commercial seats grew 6%"),
        C("Microsoft 365 Commercial products revenue", 13, "percent", "", "growth_pct", "Microsoft 365 Commercial products revenue grew 13%"),
    ]),
    dict(**MSFT_10K, chunk_index=84, text="LinkedIn revenue increased $2.0 billion or 11% with growth across all lines of business.", claims=[
        C("LinkedIn revenue", 2_000_000_000, "USD", "", "absolute_change", "LinkedIn revenue increased $2.0 billion"),
        C("LinkedIn revenue", 11, "percent", "", "growth_pct", "LinkedIn revenue increased $2.0 billion or 11%"),
    ]),
    dict(**MSFT_10K, chunk_index=86, text="Operating income increased $14.1 billion or 20%.", claims=[
        C("Operating income", 14_100_000_000, "USD", "", "absolute_change", "Operating income increased $14.1 billion"),
        C("Operating income", 20, "percent", "", "growth_pct", "Operating income increased $14.1 billion or 20%"),
    ]),
    dict(**MSFT_10K, chunk_index=90, text="Revenue, gross margin, and operating income included a favorable foreign currency impact of 2%, 3%, and 3%, respectively.", claims=[
        C("Revenue foreign currency impact", 2, "percent", "", "absolute", "Revenue, gross margin, and operating income included a favorable foreign currency impact of 2%, 3%, and 3%, respectively."),
        C("gross margin foreign currency impact", 3, "percent", "", "absolute", "Revenue, gross margin, and operating income included a favorable foreign currency impact of 2%, 3%, and 3%, respectively."),
        C("operating income foreign currency impact", 3, "percent", "", "absolute", "Revenue, gross margin, and operating income included a favorable foreign currency impact of 2%, 3%, and 3%, respectively."),
    ]),

    # ---------- MSFT 10-K: segment detail (Intelligent Cloud) ----------
    dict(**MSFT_10K, chunk_index=92, text="Server products and cloud services revenue increased $31.0 billion or 31% driven by Azure and other cloud services. Azure and other cloud services revenue grew 41% driven by demand for services across the platform with continued growth across all workloads. Server products revenue increased 1% primarily driven by higher purchases of licenses running in multi-cloud environments, offset in part by continued customer shift to cloud.", claims=[
        C("Server products and cloud services revenue", 31_000_000_000, "USD", "", "absolute_change", "Server products and cloud services revenue increased $31.0 billion"),
        C("Server products and cloud services revenue", 31, "percent", "", "growth_pct", "Server products and cloud services revenue increased $31.0 billion or 31%"),
        C("Azure and other cloud services revenue", 41, "percent", "", "growth_pct", "Azure and other cloud services revenue grew 41%"),
        C("Server products revenue", 1, "percent", "", "growth_pct", "Server products revenue increased 1%"),
    ]),
    dict(**MSFT_10K, chunk_index=95, text="Cost of revenue increased $17.7 billion or 44% driven by investments in AI infrastructure to support growing customer demand.", claims=[
        C("Cost of revenue", 17_700_000_000, "USD", "", "absolute_change", "Cost of revenue increased $17.7 billion"),
        C("Cost of revenue", 44, "percent", "", "growth_pct", "Cost of revenue increased $17.7 billion or 44%"),
    ]),

    # ---------- MSFT 10-K: segment detail (More Personal Computing, decreases) ----------
    dict(**MSFT_10K, chunk_index=100, text="Revenue decreased $597 million or 1%.", claims=[
        C("Revenue", -597_000_000, "USD", "", "absolute_change", "Revenue decreased $597 million"),
        C("Revenue", -1, "percent", "", "growth_pct", "Revenue decreased $597 million or 1%"),
    ]),
    dict(**MSFT_10K, chunk_index=102, text="XBOX revenue decreased $1.7 billion or 7% driven by declines in XBOX content and services and XBOX hardware. XBOX content and services revenue decreased 5% on a prior year comparable that benefited from strong first-party content performance, offset in part by growth in XBOX Game Pass. XBOX hardware revenue decreased 29% driven by lower volume of consoles sold.", claims=[
        C("XBOX revenue", -1_700_000_000, "USD", "", "absolute_change", "XBOX revenue decreased $1.7 billion"),
        C("XBOX revenue", -7, "percent", "", "growth_pct", "XBOX revenue decreased $1.7 billion or 7%"),
        C("XBOX content and services revenue", -5, "percent", "", "growth_pct", "XBOX content and services revenue decreased 5%"),
        C("XBOX hardware revenue", -29, "percent", "", "growth_pct", "XBOX hardware revenue decreased 29%"),
    ]),
    dict(**MSFT_10K, chunk_index=105, text="Cost of revenue decreased $1.8 billion or 7% driven by lower hardware sales.", claims=[
        C("Cost of revenue", -1_800_000_000, "USD", "", "absolute_change", "Cost of revenue decreased $1.8 billion"),
        C("Cost of revenue", -7, "percent", "", "growth_pct", "Cost of revenue decreased $1.8 billion or 7%"),
    ]),

    # ---------- MSFT 10-K: tax / other ----------
    dict(**MSFT_10K, chunk_index=138, text="Our effective tax rate for fiscal years 2026 and 2025 was 19% and 18%, respectively. The increase in our effective tax rate was primarily due to changes in the mix of our earnings and tax expenses between the U.S. and foreign countries.", claims=[
        C("effective tax rate", 19, "percent", "fiscal year 2026", "absolute", "Our effective tax rate for fiscal years 2026 and 2025 was 19% and 18%, respectively."),
        C("effective tax rate", 18, "percent", "fiscal year 2025", "absolute", "Our effective tax rate for fiscal years 2026 and 2025 was 19% and 18%, respectively."),
    ]),
    dict(**MSFT_10K, chunk_index=140, text="The mix of income before income taxes between the U.S. and foreign countries impacted our effective tax rate as a result of the geographic distribution of, and customer demand for, our products and services. In fiscal year 2026, our U.S. income before income taxes was $103.6 billion and our foreign income before income taxes was $62.3 billion. In fiscal year 2025, our U.S. income before income taxes was $69.2 billion and our foreign income before income taxes was $54.4 billion.", claims=[
        C("U.S. income before income taxes", 103_600_000_000, "USD", "fiscal year 2026", "absolute", "our U.S. income before income taxes was $103.6 billion"),
        C("foreign income before income taxes", 62_300_000_000, "USD", "fiscal year 2026", "absolute", "our foreign income before income taxes was $62.3 billion"),
        C("U.S. income before income taxes", 69_200_000_000, "USD", "fiscal year 2025", "absolute", "our U.S. income before income taxes was $69.2 billion"),
        C("foreign income before income taxes", 54_400_000_000, "USD", "fiscal year 2025", "absolute", "our foreign income before income taxes was $54.4 billion"),
    ]),
    dict(**MSFT_10K, chunk_index=163, text="Cash, cash equivalents, and short-term investments totaled $76.8 billion and $94.6 billion as of June 30, 2026 and 2025, respectively. Equity and other investments were $36.3 billion and $15.4 billion as of June 30, 2026 and 2025, respectively. Our short-term investments are primarily intended to facilitate liquidity and capital preservation.", claims=[
        C("cash, cash equivalents, and short-term investments", 76_800_000_000, "USD", "June 30, 2026", "absolute", "Cash, cash equivalents, and short-term investments totaled $76.8 billion and $94.6 billion as of June 30, 2026 and 2025, respectively."),
        C("cash, cash equivalents, and short-term investments", 94_600_000_000, "USD", "June 30, 2025", "absolute", "Cash, cash equivalents, and short-term investments totaled $76.8 billion and $94.6 billion as of June 30, 2026 and 2025, respectively."),
        C("equity and other investments", 36_300_000_000, "USD", "June 30, 2026", "absolute", "Equity and other investments were $36.3 billion and $15.4 billion as of June 30, 2026 and 2025, respectively."),
        C("equity and other investments", 15_400_000_000, "USD", "June 30, 2025", "absolute", "Equity and other investments were $36.3 billion and $15.4 billion as of June 30, 2026 and 2025, respectively."),
    ]),
    dict(**MSFT_10K, chunk_index=181, text="During fiscal years 2026 and 2025, we repurchased 36 million shares and 31 million shares of our common stock for $16.7 billion and $13.0 billion, respectively, through our share repurchase program. All repurchases were made using cash resources. As of June 30, 2026, $40.6 billion remained of our $60 billion share repurchase program.", claims=[
        C("common stock repurchased", 16_700_000_000, "USD", "fiscal year 2026", "absolute", "we repurchased 36 million shares and 31 million shares of our common stock for $16.7 billion and $13.0 billion, respectively"),
        C("common stock repurchased", 13_000_000_000, "USD", "fiscal year 2025", "absolute", "we repurchased 36 million shares and 31 million shares of our common stock for $16.7 billion and $13.0 billion, respectively"),
        C("remaining share repurchase authorization", 40_600_000_000, "USD", "as of June 30, 2026", "absolute", "As of June 30, 2026, $40.6 billion remained of our $60 billion share repurchase program."),
    ]),
    dict(**MSFT_10K, chunk_index=182, text="During fiscal years 2026 and 2025, our Board of Directors declared dividends totaling $27.0 billion and $24.7 billion, respectively. We intend to continue returning capital to shareholders in the form of dividends, subject to declaration by our Board of Directors.", claims=[
        C("dividends declared", 27_000_000_000, "USD", "fiscal year 2026", "absolute", "our Board of Directors declared dividends totaling $27.0 billion and $24.7 billion, respectively"),
        C("dividends declared", 24_700_000_000, "USD", "fiscal year 2025", "absolute", "our Board of Directors declared dividends totaling $27.0 billion and $24.7 billion, respectively"),
    ]),

    # ---------- MSFT 10-K: TRUE NEGATIVES (no checkable company-reported claim) ----------
    dict(**MSFT_10K, chunk_index=142, text="We remain under audit by the IRS for tax years 2014 to 2017. With respect to the audit for tax years 2004 to 2013, on September 26, 2023, we received Notices of Proposed Adjustment (“NOPAs”) from the IRS. The primary issues in the NOPAs relate to intercompany transfer pricing. In the NOPAs, the IRS is seeking an additional tax payment of $28.9 billion plus penalties and interest. As of June 30, 2026, we believe our allowances for income tax contingencies are adequate. We disagree with the proposed adjustments and will vigorously contest the NOPAs through the IRS’s administrative appeals office and, if necessary, judicial proceedings.", claims=[]),
    dict(**MSFT_10K, chunk_index=155, text="$(1,143), and $(356)", claims=[]),
    dict(**MSFT_10K, chunk_index=206, text="The objectives of accounting for income taxes are to recognize the amount of taxes payable or refundable for the current year, and deferred tax liabilities and assets for the future tax consequences of events that have been recognized in an entity’s financial statements or tax returns. We recognize the tax benefit from an uncertain tax position only if it is more likely than not that the tax position will be sustained on examination by the taxing authorities, based on the technical merits of the position. The tax benefits recognized in the financial statements from such a position are measured based on the largest benefit that has a greater than 50% likelihood of being realized upon ultimate settlement.", claims=[]),

    # ---------- MSFT 10-Q: quarterly highlights + results (period stated differently: "three/nine months") ----------
    dict(**MSFT_10Q, chunk_index=7, text="Microsoft Cloud revenue increased 29% to $54.5 billion.", claims=[
        C("Microsoft Cloud revenue", 29, "percent", "", "growth_pct", "Microsoft Cloud revenue increased 29%"),
        C("Microsoft Cloud revenue", 54_500_000_000, "USD", "", "absolute", "Microsoft Cloud revenue increased 29% to $54.5 billion."),
    ]),
    dict(**MSFT_10Q, chunk_index=8, text="Commercial remaining performance obligation increased 99% to $627 billion.", claims=[
        C("Commercial remaining performance obligation", 99, "percent", "", "growth_pct", "Commercial remaining performance obligation increased 99%"),
        C("Commercial remaining performance obligation", 627_000_000_000, "USD", "", "absolute", "Commercial remaining performance obligation increased 99% to $627 billion."),
    ]),
    dict(**MSFT_10Q, chunk_index=13, text="Azure and other cloud services revenue increased 40%.", claims=[
        C("Azure and other cloud services revenue", 40, "percent", "", "growth_pct", "Azure and other cloud services revenue increased 40%."),
    ]),
    dict(**MSFT_10Q, chunk_index=14, text="Windows OEM and Devices revenue decreased 2%.", claims=[
        C("Windows OEM and Devices revenue", -2, "percent", "", "growth_pct", "Windows OEM and Devices revenue decreased 2%."),
    ]),
    dict(**MSFT_10Q, chunk_index=68, text="Revenue increased $12.8 billion or 18% driven by growth in Microsoft Cloud. Intelligent Cloud revenue increased driven by Azure. Productivity and Business Processes revenue increased driven by Microsoft 365 Commercial cloud. More Personal Computing revenue decreased with lower hardware sales across Devices and Gaming, offset in part by growth in Search advertising.", claims=[
        C("Revenue", 12_800_000_000, "USD", "", "absolute_change", "Revenue increased $12.8 billion"),
        C("Revenue", 18, "percent", "", "growth_pct", "Revenue increased $12.8 billion or 18%"),
    ]),
    dict(**MSFT_10Q, chunk_index=75, text="Revenue, gross margin, and operating income included a favorable foreign currency impact of 3%, 3%, and 4%, respectively. Cost of revenue included an unfavorable foreign currency impact of 2%.", claims=[
        C("Revenue foreign currency impact", 3, "percent", "", "absolute", "Revenue, gross margin, and operating income included a favorable foreign currency impact of 3%, 3%, and 4%, respectively."),
        C("gross margin foreign currency impact", 3, "percent", "", "absolute", "Revenue, gross margin, and operating income included a favorable foreign currency impact of 3%, 3%, and 4%, respectively."),
        C("operating income foreign currency impact", 4, "percent", "", "absolute", "Revenue, gross margin, and operating income included a favorable foreign currency impact of 3%, 3%, and 4%, respectively."),
        C("cost of revenue foreign currency impact", -2, "percent", "", "absolute", "Cost of revenue included an unfavorable foreign currency impact of 2%."),
    ]),
    dict(**MSFT_10Q, chunk_index=76, text="Current year net income and diluted EPS were negatively impacted by net losses from investments in OpenAI, which resulted in a decrease in net income of $14 million. Prior year net income and diluted EPS were negatively impacted by net losses from investments in OpenAI, which resulted in a decrease in net income and diluted EPS of $583 million and $0.08, respectively.", claims=[
        C("net income impact from OpenAI investment losses", -14_000_000, "USD", "current year", "absolute_change", "a decrease in net income of $14 million"),
        C("net income impact from OpenAI investment losses", -583_000_000, "USD", "prior year", "absolute_change", "a decrease in net income and diluted EPS of $583 million and $0.08, respectively"),
        C("diluted EPS impact from OpenAI investment losses", -0.08, "USD", "prior year", "absolute_change", "a decrease in net income and diluted EPS of $583 million and $0.08, respectively"),
    ]),
    dict(**MSFT_10Q, chunk_index=94, text="Microsoft 365 Commercial products and cloud services revenue increased $3.7 billion or 17%. Microsoft 365 Commercial cloud revenue grew 19% with growth in revenue per user driven by Microsoft 365 E5 and Microsoft 365 Copilot. Microsoft 365 Commercial seats grew 6% driven by small and medium businesses and frontline worker offerings. Microsoft 365 Commercial products revenue grew 1%.", claims=[
        C("Microsoft 365 Commercial products and cloud services revenue", 3_700_000_000, "USD", "", "absolute_change", "Microsoft 365 Commercial products and cloud services revenue increased $3.7 billion"),
        C("Microsoft 365 Commercial products and cloud services revenue", 17, "percent", "", "growth_pct", "Microsoft 365 Commercial products and cloud services revenue increased $3.7 billion or 17%"),
        C("Microsoft 365 Commercial cloud revenue", 19, "percent", "", "growth_pct", "Microsoft 365 Commercial cloud revenue grew 19%"),
        C("Microsoft 365 Commercial seats", 6, "percent", "", "growth_pct", "Microsoft 365 Commercial seats grew 6%"),
        C("Microsoft 365 Commercial products revenue", 1, "percent", "", "growth_pct", "Microsoft 365 Commercial products revenue grew 1%"),
    ]),
    dict(**MSFT_10Q, chunk_index=104, text="Server products and cloud services revenue increased $7.8 billion or 32% driven by Azure and other cloud services. Azure and other cloud services revenue grew 40% driven by demand for services across the platform with continued growth across all workloads. Server products revenue increased slightly, primarily driven by higher purchases of licenses running in multi-cloud environments, offset in part by renewals with lower in-period revenue recognition from the mix of contracts and continued customer shift to cloud.", claims=[
        C("Server products and cloud services revenue", 7_800_000_000, "USD", "", "absolute_change", "Server products and cloud services revenue increased $7.8 billion"),
        C("Server products and cloud services revenue", 32, "percent", "", "growth_pct", "Server products and cloud services revenue increased $7.8 billion or 32%"),
        C("Azure and other cloud services revenue", 40, "percent", "", "growth_pct", "Azure and other cloud services revenue grew 40%"),
    ]),
    dict(**MSFT_10Q, chunk_index=114, text="Gaming revenue decreased $380 million or 7% driven by declines in Xbox content and services and Xbox hardware. Xbox content and services revenue decreased 5% on a prior year comparable that benefited from strong first-party content performance. Xbox hardware revenue decreased 33% driven by lower volume of consoles sold.", claims=[
        C("Gaming revenue", -380_000_000, "USD", "", "absolute_change", "Gaming revenue decreased $380 million"),
        C("Gaming revenue", -7, "percent", "", "growth_pct", "Gaming revenue decreased $380 million or 7%"),
        C("Xbox content and services revenue", -5, "percent", "", "growth_pct", "Xbox content and services revenue decreased 5%"),
        C("Xbox hardware revenue", -33, "percent", "", "growth_pct", "Xbox hardware revenue decreased 33%"),
    ]),
    dict(**MSFT_10Q, chunk_index=188, text="Our effective tax rate was 19% and 18% for the three months ended March 31, 2026 and 2025, respectively, and 20% and 18% for the nine months ended March 31, 2026 and 2025, respectively.", claims=[
        C("effective tax rate", 19, "percent", "three months ended March 31, 2026", "absolute", "Our effective tax rate was 19% and 18% for the three months ended March 31, 2026 and 2025, respectively"),
        C("effective tax rate", 18, "percent", "three months ended March 31, 2025", "absolute", "Our effective tax rate was 19% and 18% for the three months ended March 31, 2026 and 2025, respectively"),
        C("effective tax rate", 20, "percent", "nine months ended March 31, 2026", "absolute", "20% and 18% for the nine months ended March 31, 2026 and 2025, respectively"),
        C("effective tax rate", 18, "percent", "nine months ended March 31, 2025", "absolute", "20% and 18% for the nine months ended March 31, 2026 and 2025, respectively"),
    ]),
    dict(**MSFT_10Q, chunk_index=217, text="For the nine months ended March 31, 2026 and 2025, we repurchased 27 million shares and 23 million shares of our common stock for $13.3 billion and $9.8 billion, respectively, through our share repurchase program. As of March 31, 2026, $44.0 billion remained of our $60 billion share repurchase program.", claims=[
        C("common stock repurchased", 13_300_000_000, "USD", "nine months ended March 31, 2026", "absolute", "we repurchased 27 million shares and 23 million shares of our common stock for $13.3 billion and $9.8 billion, respectively"),
        C("common stock repurchased", 9_800_000_000, "USD", "nine months ended March 31, 2025", "absolute", "we repurchased 27 million shares and 23 million shares of our common stock for $13.3 billion and $9.8 billion, respectively"),
        C("remaining share repurchase authorization", 44_000_000_000, "USD", "as of March 31, 2026", "absolute", "As of March 31, 2026, $44.0 billion remained of our $60 billion share repurchase program."),
    ]),

    # ---------- MSFT 10-Q: TRUE NEGATIVES ----------
    dict(**MSFT_10Q, chunk_index=191, text="We remain under audit by the IRS for tax years 2014 to 2017. With respect to the audit for tax years 2004 to 2013, on September 26, 2023, we received Notices of Proposed Adjustment (“NOPAs”) from the IRS. In the NOPAs, the IRS is seeking an additional tax payment of $28.9 billion plus penalties and interest. As of March 31, 2026, we believe our allowances for income tax contingencies are adequate.", claims=[]),
    dict(**MSFT_10Q, chunk_index=241, text="The objectives of accounting for income taxes are to recognize the amount of taxes payable or refundable for the current year, and deferred tax liabilities and assets for the future tax consequences of events that have been recognized in an entity’s financial statements or tax returns. Judgment is required in assessing the future tax consequences of events that have been recognized in our consolidated financial statements or tax returns.", claims=[]),

    # ---------- NVDA 10-K: headline highlights (period stated: "fiscal year 2026") ----------
    dict(**NVDA_10K, chunk_index=22, text="Revenue for fiscal year 2026 was $215.9 billion, up 65% from a year ago.", claims=[
        C("Revenue", 215_900_000_000, "USD", "fiscal year 2026", "absolute", "Revenue for fiscal year 2026 was $215.9 billion"),
        C("Revenue", 65, "percent", "fiscal year 2026", "growth_pct", "up 65% from a year ago"),
    ]),
    dict(**NVDA_10K, chunk_index=23, text="Data Center revenue for fiscal year 2026 was up 68% from a year ago. The strong year-on-year growth was driven by the major platform shifts – accelerated computing and AI.", claims=[
        C("Data Center revenue", 68, "percent", "fiscal year 2026", "growth_pct", "Data Center revenue for fiscal year 2026 was up 68% from a year ago."),
    ]),
    dict(**NVDA_10K, chunk_index=24, text="Gaming revenue for fiscal year 2026 was up 41% from a year ago, driven by strong Blackwell demand. We expect supply constraints to be a headwind to Gaming in the first quarter of fiscal 2027 and beyond.", claims=[
        C("Gaming revenue", 41, "percent", "fiscal year 2026", "growth_pct", "Gaming revenue for fiscal year 2026 was up 41% from a year ago"),
    ]),
    dict(**NVDA_10K, chunk_index=25, text="Professional Visualization revenue for fiscal year 2026 was up 70% from a year ago, driven by exceptional demand for Blackwell as well as the launch of our new DGX Spark.", claims=[
        C("Professional Visualization revenue", 70, "percent", "fiscal year 2026", "growth_pct", "Professional Visualization revenue for fiscal year 2026 was up 70% from a year ago"),
    ]),
    dict(**NVDA_10K, chunk_index=26, text="Automotive revenue for fiscal year 2026 was up 39% from a year ago, driven by continued adoption of our self-driving platforms.", claims=[
        C("Automotive revenue", 39, "percent", "fiscal year 2026", "growth_pct", "Automotive revenue for fiscal year 2026 was up 39% from a year ago"),
    ]),
    dict(**NVDA_10K, chunk_index=28, text="Operating expenses for fiscal year 2026 were up 41% from a year ago, driven by higher compensation and benefits expenses due to employee growth and compute and infrastructure costs.", claims=[
        C("Operating expenses", 41, "percent", "fiscal year 2026", "growth_pct", "Operating expenses for fiscal year 2026 were up 41% from a year ago"),
    ]),
    dict(**NVDA_10K, chunk_index=59, text="– The year over year increase was driven by the major platform shifts – accelerated computing and AI. Revenue from Data Center computing grew 59% driven by demand for our Blackwell computing platform. Revenue from Data Center networking grew 142% driven by the introduction and continued ramp of NVLink compute fabric for GB200 and GB300 systems and the growth of Ethernet and InfiniBand platforms.", claims=[
        C("Data Center computing revenue", 59, "percent", "", "growth_pct", "Revenue from Data Center computing grew 59%"),
        C("Data Center networking revenue", 142, "percent", "", "growth_pct", "Revenue from Data Center networking grew 142%"),
    ]),
    dict(**NVDA_10K, chunk_index=65, text="– For fiscal year 2026, sales to one direct customer represented 22% of total revenue and sales to another direct customer represented 14% of total revenue, all of which were primarily attributable to the Compute & Networking segment.", claims=[
        C("customer concentration (largest direct customer)", 22, "percent", "fiscal year 2026", "absolute", "sales to one direct customer represented 22% of total revenue"),
        C("customer concentration (second direct customer)", 14, "percent", "fiscal year 2026", "absolute", "sales to another direct customer represented 14% of total revenue"),
    ]),
    dict(**NVDA_10K, chunk_index=69, text="Revenue by geographic region is designated based on the location of the headquarters of direct customers. The end customer and shipping location may be different from our customers' headquarters location. Revenue from sales to customers headquartered outside of the United States accounted for 31% and 41% of total revenue for fiscal years 2026 and 2025, respectively.", claims=[
        C("revenue from customers headquartered outside the U.S.", 31, "percent", "fiscal year 2026", "absolute", "Revenue from sales to customers headquartered outside of the United States accounted for 31% and 41% of total revenue for fiscal years 2026 and 2025, respectively."),
        C("revenue from customers headquartered outside the U.S.", 41, "percent", "fiscal year 2025", "absolute", "Revenue from sales to customers headquartered outside of the United States accounted for 31% and 41% of total revenue for fiscal years 2026 and 2025, respectively."),
    ]),
    dict(**NVDA_10K, chunk_index=72, text="Gross margins decreased to 71.1% in fiscal year 2026 from 75.0% in fiscal year 2025 as our business model transitioned from offering Hopper HGX systems to Blackwell full-scale datacenter solutions and a $4.5 billion charge associated with H20 excess inventory and purchase obligations in the first quarter of fiscal year 2026.", claims=[
        C("gross margin", 71.1, "percent", "fiscal year 2026", "absolute", "Gross margins decreased to 71.1% in fiscal year 2026"),
        C("gross margin", 75.0, "percent", "fiscal year 2025", "absolute", "from 75.0% in fiscal year 2025"),
        C("H20 excess inventory and purchase obligations charge", 4_500_000_000, "USD", "first quarter of fiscal year 2026", "absolute", "a $4.5 billion charge associated with H20 excess inventory and purchase obligations in the first quarter of fiscal year 2026"),
    ]),
    dict(**NVDA_10K, chunk_index=77, text="The increases in research and development expenses for fiscal year 2026 were driven by a 29% increase in compensation and benefits expense, including stock-based compensation, reflecting employee growth and compensation increases and a 79% increase in compute and infrastructure.", claims=[
        C("compensation and benefits expense", 29, "percent", "fiscal year 2026", "growth_pct", "a 29% increase in compensation and benefits expense"),
        C("compute and infrastructure expense", 79, "percent", "fiscal year 2026", "growth_pct", "a 79% increase in compute and infrastructure"),
    ]),
    dict(**NVDA_10K, chunk_index=84, text="Income tax expense was $21.4 billion and $11.1 billion for fiscal years 2026 and 2025, respectively. Income tax as a percentage of income before income tax was an expense of 15.1% and 13.3% for fiscal years 2026 and 2025, respectively.", claims=[
        C("income tax expense", 21_400_000_000, "USD", "fiscal year 2026", "absolute", "Income tax expense was $21.4 billion and $11.1 billion for fiscal years 2026 and 2025, respectively."),
        C("income tax expense", 11_100_000_000, "USD", "fiscal year 2025", "absolute", "Income tax expense was $21.4 billion and $11.1 billion for fiscal years 2026 and 2025, respectively."),
        C("effective tax rate", 15.1, "percent", "fiscal year 2026", "absolute", "Income tax as a percentage of income before income tax was an expense of 15.1% and 13.3% for fiscal years 2026 and 2025, respectively."),
        C("effective tax rate", 13.3, "percent", "fiscal year 2025", "absolute", "Income tax as a percentage of income before income tax was an expense of 15.1% and 13.3% for fiscal years 2026 and 2025, respectively."),
    ]),
    dict(**NVDA_10K, chunk_index=101, text="Our primary sources of liquidity include cash, cash equivalents, marketable securities, and cash generated by our operations. As of January 25, 2026, we had $62.6 billion in cash, cash equivalents, and marketable securities. We believe that we have sufficient liquidity to meet our operating requirements for at least the next twelve months and for the foreseeable future, including our future obligations.", claims=[
        C("cash, cash equivalents, and marketable securities", 62_600_000_000, "USD", "as of January 25, 2026", "absolute", "we had $62.6 billion in cash, cash equivalents, and marketable securities"),
    ]),
    dict(**NVDA_10K, chunk_index=105, text="On August 26, 2025, our Board of Directors approved an additional $60.0 billion in share repurchase authorization, without expiration. In fiscal year 2026, we repurchased 282 million shares of our common stock for $40.4 billion. As of January 25, 2026, we were authorized, subject to certain specifications, to repurchase up to $58.5 billion of our common stock.", claims=[
        C("share repurchase authorization approved", 60_000_000_000, "USD", "August 26, 2025", "absolute", "our Board of Directors approved an additional $60.0 billion in share repurchase authorization"),
        C("common stock repurchased", 40_400_000_000, "USD", "fiscal year 2026", "absolute", "we repurchased 282 million shares of our common stock for $40.4 billion"),
        C("remaining share repurchase authorization", 58_500_000_000, "USD", "as of January 25, 2026", "absolute", "we were authorized, subject to certain specifications, to repurchase up to $58.5 billion of our common stock"),
    ]),
    dict(**NVDA_10K, chunk_index=107, text="In fiscal year 2026, we paid cash dividends to our shareholders of $974 million. The payment of future cash dividends is subject to our Board of Directors' continuing determination that the declaration of dividends is in the best interests of our shareholders.", claims=[
        C("cash dividends paid", 974_000_000, "USD", "fiscal year 2026", "absolute", "we paid cash dividends to our shareholders of $974 million"),
    ]),

    # ---------- NVDA 10-K: TRUE NEGATIVES ----------
    dict(**NVDA_10K, chunk_index=14, text="We invested $17.5 billion in private companies and infrastructure funds, primarily to support early‑stage startups. These investments include AI model makers that purchase our products directly or through CSPs. Many of these investments are illiquid and non‑marketable. The related early-stage startups may not become profitable in the near term, or at all, and there can be no assurance that we will realize a return on our investments.", claims=[]),
    dict(**NVDA_10K, chunk_index=10, text="In February 2026, the USG granted a license that would allow us to ship small amounts of H200 products to specific China-based customers. To date, we have not generated any revenue under the H200 licensing program, and do not yet know whether any imports will be allowed into China. The license requires that the H200s go through an inspection process in the United States prior to any shipment to the customer. As a result, any H200 shipped under the new licensing program will be subject to a 25% tariff upon importation into the United States.", claims=[]),

    # ---------- NVDA 10-Q: sequential comparisons (a new period-phrasing variant) ----------
    dict(**NVDA_10Q, chunk_index=38, text="Revenue was $81.6 billion, up 85% from a year ago and up 20% sequentially.", claims=[
        C("Revenue", 81_600_000_000, "USD", "", "absolute", "Revenue was $81.6 billion"),
        C("Revenue", 85, "percent", "year ago", "growth_pct", "up 85% from a year ago"),
        C("Revenue", 20, "percent", "sequentially", "growth_pct", "up 20% sequentially"),
    ]),
    dict(**NVDA_10Q, chunk_index=39, text="Data Center revenue was $75.2 billion, up 92% from a year ago and up 21% sequentially, driven by the ramp of our Blackwell 300 products and demand for our InfiniBand, Spectrum-X Ethernet, and NVLink solutions. Hyperscaler revenue increased sequentially and remained at approximately 50% of Data Center revenue.", claims=[
        C("Data Center revenue", 75_200_000_000, "USD", "", "absolute", "Data Center revenue was $75.2 billion"),
        C("Data Center revenue", 92, "percent", "year ago", "growth_pct", "up 92% from a year ago"),
        C("Data Center revenue", 21, "percent", "sequentially", "growth_pct", "up 21% sequentially"),
        C("Hyperscaler revenue as a share of Data Center revenue", 50, "percent", "", "absolute", "Hyperscaler revenue increased sequentially and remained at approximately 50% of Data Center revenue"),
    ]),
    dict(**NVDA_10Q, chunk_index=40, text="Edge Computing revenue for the first quarter was $6.4 billion, up 29% from a year ago and up 10% sequentially. The increases were driven by robust Blackwell workstation demand, partially offset by slower consumer PC demand that was tempered by elevated memory and systems prices.", claims=[
        C("Edge Computing revenue", 6_400_000_000, "USD", "first quarter", "absolute", "Edge Computing revenue for the first quarter was $6.4 billion"),
        C("Edge Computing revenue", 29, "percent", "year ago", "growth_pct", "up 29% from a year ago"),
        C("Edge Computing revenue", 10, "percent", "sequentially", "growth_pct", "up 10% sequentially"),
    ]),
    dict(**NVDA_10Q, chunk_index=42, text="Operating expenses were up 52% from a year ago and up 12% sequentially. The increases were primarily driven by higher compensation and benefits expense due to employee growth and compensation increases, compute and infrastructure costs, and engineering development materials for new product developments.", claims=[
        C("Operating expenses", 52, "percent", "year ago", "growth_pct", "Operating expenses were up 52% from a year ago"),
        C("Operating expenses", 12, "percent", "sequentially", "growth_pct", "up 12% sequentially"),
    ]),
    dict(**NVDA_10Q, chunk_index=66, text="– For the first quarter of fiscal year 2027, three direct customers represented 21%, 17%, and 16% of total revenue, all of which was primarily attributable to the Compute & Networking segment.", claims=[
        C("customer concentration (largest direct customer)", 21, "percent", "first quarter of fiscal year 2027", "absolute", "three direct customers represented 21%, 17%, and 16% of total revenue"),
        C("customer concentration (second direct customer)", 17, "percent", "first quarter of fiscal year 2027", "absolute", "three direct customers represented 21%, 17%, and 16% of total revenue"),
        C("customer concentration (third direct customer)", 16, "percent", "first quarter of fiscal year 2027", "absolute", "three direct customers represented 21%, 17%, and 16% of total revenue"),
    ]),
    dict(**NVDA_10Q, chunk_index=70, text="Revenue by geographic region is designated based on the location of the headquarters of direct customers. The end customer and shipping location may be different from our customers' headquarters location. Revenue from sales to customers headquartered outside of the United States accounted for 22% of total revenue for the first quarter of fiscal year 2027 and 42% of total revenue for the first quarter of fiscal year 2026.", claims=[
        C("revenue from customers headquartered outside the U.S.", 22, "percent", "first quarter of fiscal year 2027", "absolute", "Revenue from sales to customers headquartered outside of the United States accounted for 22% of total revenue for the first quarter of fiscal year 2027"),
        C("revenue from customers headquartered outside the U.S.", 42, "percent", "first quarter of fiscal year 2026", "absolute", "42% of total revenue for the first quarter of fiscal year 2026"),
    ]),
    dict(**NVDA_10Q, chunk_index=73, text="Gross margin increased to 74.9% for the first quarter of fiscal year 2027 compared to 60.5% for the first quarter of fiscal year 2026, primarily due to the prior year's $4.5 billion charge associated with H20 excess inventory and purchase obligations.", claims=[
        C("gross margin", 74.9, "percent", "first quarter of fiscal year 2027", "absolute", "Gross margin increased to 74.9% for the first quarter of fiscal year 2027"),
        C("gross margin", 60.5, "percent", "first quarter of fiscal year 2026", "absolute", "compared to 60.5% for the first quarter of fiscal year 2026"),
    ]),
    dict(**NVDA_10Q, chunk_index=78, text="The increase in research and development expenses for the first quarter of fiscal year 2027 was primarily driven by a 112% increase in compute and infrastructure, a 31% increase in compensation and benefits, including stock-based compensation, reflecting employee growth and compensation increases, and a 204% increase in engineering development materials for new product introductions.", claims=[
        C("compute and infrastructure expense", 112, "percent", "first quarter of fiscal year 2027", "growth_pct", "a 112% increase in compute and infrastructure"),
        C("compensation and benefits expense", 31, "percent", "first quarter of fiscal year 2027", "growth_pct", "a 31% increase in compensation and benefits"),
        C("engineering development materials expense", 204, "percent", "first quarter of fiscal year 2027", "growth_pct", "a 204% increase in engineering development materials for new product introductions"),
    ]),
    dict(**NVDA_10Q, chunk_index=84, text="Income tax expense was $11.6 billion and $3.1 billion for the first quarter of fiscal years 2027 and 2026, respectively. Income tax as a percentage of income before income tax was 16.6% and 14.3% for the first quarter of fiscal years 2027 and 2026, respectively.", claims=[
        C("income tax expense", 11_600_000_000, "USD", "first quarter of fiscal year 2027", "absolute", "Income tax expense was $11.6 billion and $3.1 billion for the first quarter of fiscal years 2027 and 2026, respectively."),
        C("income tax expense", 3_100_000_000, "USD", "first quarter of fiscal year 2026", "absolute", "Income tax expense was $11.6 billion and $3.1 billion for the first quarter of fiscal years 2027 and 2026, respectively."),
        C("effective tax rate", 16.6, "percent", "first quarter of fiscal year 2027", "absolute", "Income tax as a percentage of income before income tax was 16.6% and 14.3% for the first quarter of fiscal years 2027 and 2026, respectively."),
        C("effective tax rate", 14.3, "percent", "first quarter of fiscal year 2026", "absolute", "Income tax as a percentage of income before income tax was 16.6% and 14.3% for the first quarter of fiscal years 2027 and 2026, respectively."),
    ]),
    dict(**NVDA_10Q, chunk_index=103, text="In the first quarter of fiscal year 2027, we repurchased 108 million shares of our common stock for $20.2 billion. As of April 26, 2026, we were authorized, subject to certain specifications, to repurchase up to $38.5 billion of our common stock.", claims=[
        C("common stock repurchased", 20_200_000_000, "USD", "first quarter of fiscal year 2027", "absolute", "we repurchased 108 million shares of our common stock for $20.2 billion"),
        C("remaining share repurchase authorization", 38_500_000_000, "USD", "as of April 26, 2026", "absolute", "we were authorized, subject to certain specifications, to repurchase up to $38.5 billion of our common stock"),
    ]),
    dict(**NVDA_10Q, chunk_index=104, text="On May 18, 2026, our Board of Directors approved an additional $80.0 billion in share repurchase authorization, without expiration.", claims=[
        C("share repurchase authorization approved", 80_000_000_000, "USD", "May 18, 2026", "absolute", "our Board of Directors approved an additional $80.0 billion in share repurchase authorization"),
    ]),
    dict(**NVDA_10Q, chunk_index=106, text="We paid cash dividends to our shareholders of $243 million during the first quarter of fiscal year 2027. On May 18, 2026, we increased our quarterly cash dividend from $0.01 per share to $0.25 per share to all shareholders of record on June 4, 2026.", claims=[
        C("cash dividends paid", 243_000_000, "USD", "first quarter of fiscal year 2027", "absolute", "We paid cash dividends to our shareholders of $243 million during the first quarter of fiscal year 2027."),
        C("quarterly cash dividend per share", 0.01, "USD", "before May 18, 2026", "absolute", "we increased our quarterly cash dividend from $0.01 per share to $0.25 per share"),
        C("quarterly cash dividend per share", 0.25, "USD", "after May 18, 2026", "absolute", "we increased our quarterly cash dividend from $0.01 per share to $0.25 per share"),
    ]),

    # ---------- NVDA 10-Q: TRUE NEGATIVES ----------
    dict(**NVDA_10Q, chunk_index=14, text="Beginning in February 2026, the U.S. government, or USG, granted licenses that allow us to ship small amounts of H200 products to specific China-based customers. To date, we have not generated any revenue under the H200 licensing program, and do not yet know whether any imports will be allowed into China. As a result, any H200 shipped under the new licensing program will be subject to a 25% tariff upon importation into the United States.", claims=[]),
    dict(**NVDA_10Q, chunk_index=99, text="Our primary sources of liquidity include cash, cash equivalents, marketable debt and equity securities, and cash generated by our operations. We believe that we have sufficient liquidity to meet our operating requirements for at least the next twelve months and for the foreseeable future, including our future obligations. We continuously evaluate our liquidity and capital resources, including our access to external capital, to ensure we can finance future capital requirements and commitments.", claims=[]),
]


def main():
    records = []
    for i, ex in enumerate(EXAMPLES):
        claims = [ExtractedClaim(**c).model_dump() for c in ex["claims"]]  # validates schema
        records.append({**ex, "id": i, "claims": claims})

    with open(OUTPUT_PATH, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)

    n_claims = sum(len(r["claims"]) for r in records)
    n_negative = sum(1 for r in records if not r["claims"])
    print(f"Wrote {len(records)} examples ({n_claims} total claims, {n_negative} true negatives) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
