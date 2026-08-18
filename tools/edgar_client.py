"""Thin client over SEC EDGAR's public data API.

Handles the two things every caller needs and shouldn't reimplement:
  - a compliant User-Agent header (EDGAR blocks requests without one)
  - ticker -> CIK resolution, and on-disk caching so repeated runs/tests don't
    hammer EDGAR (their fair-access rules cap request rates)
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

EDGAR_BASE_URL = os.environ.get("EDGAR_BASE_URL", "https://data.sec.gov")
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "cache"

# SEC asks for no more than ~10 requests/sec; stay well under that.
_MIN_REQUEST_INTERVAL_SECONDS = 0.15


class EdgarClientError(RuntimeError):
    """Raised for EDGAR request failures or unresolvable tickers."""


class EdgarClient:
    def __init__(
        self,
        user_agent: Optional[str] = None,
        cache_dir: Path = CACHE_DIR,
        use_cache: bool = True,
    ):
        self.user_agent = user_agent or os.environ.get("EDGAR_USER_AGENT")
        if not self.user_agent or "@" not in self.user_agent:
            raise EdgarClientError(
                "EDGAR_USER_AGENT must be set to 'Your Name your-email@example.com' "
                "(see .env.example) — SEC rejects requests without a compliant User-Agent."
            )
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self._last_request_time = 0.0
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(_MIN_REQUEST_INTERVAL_SECONDS - elapsed)
        self._last_request_time = time.monotonic()

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _get_json(self, url: str, cache_key: Optional[str] = None) -> dict:
        if self.use_cache and cache_key:
            cache_path = self._cache_path(cache_key)
            if cache_path.exists():
                return json.loads(cache_path.read_text())

        self._throttle()
        try:
            response = httpx.get(
                url, headers={"User-Agent": self.user_agent}, timeout=20.0
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EdgarClientError(f"EDGAR request failed for {url}: {exc}") from exc

        data = response.json()

        if self.use_cache and cache_key:
            self._cache_path(cache_key).write_text(json.dumps(data))

        return data

    def resolve_cik(self, ticker: str) -> str:
        """Return the zero-padded 10-digit CIK for a ticker, e.g. 'AAPL' -> '0000320193'."""
        ticker = ticker.upper().strip()
        mapping = self._get_json(TICKER_MAP_URL, cache_key="company_tickers")
        for entry in mapping.values():
            if entry["ticker"] == ticker:
                return f"{entry['cik_str']:010d}"
        raise EdgarClientError(f"Ticker '{ticker}' not found in SEC's company_tickers.json")

    def get_submissions(self, cik: str) -> dict:
        """Raw filing history (10-K/10-Q/8-K/... metadata) for a CIK."""
        url = f"{EDGAR_BASE_URL}/submissions/CIK{cik}.json"
        return self._get_json(url, cache_key=f"submissions_{cik}")

    def get_company_facts(self, cik: str) -> dict:
        """Raw XBRL company facts (every reported us-gaap concept, all periods) for a CIK."""
        url = f"{EDGAR_BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
        return self._get_json(url, cache_key=f"companyfacts_{cik}")
