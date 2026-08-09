"""
WebSearchProvider — site:-scoped general web search.

Design rules:
  - All discovery goes through this class.  Source adapters never search.
  - The actual HTTP call to a search engine is isolated in _execute_search().
  - The class ships with a DuckDuckGo HTML fallback for real usage; swap
    out _execute_search() to plug in any other engine (Google CSE, Bing, etc.)
  - Rate-limit responses raise RateLimitedError so the orchestrator can back off.

Legal / reliability note:
  - LinkedIn: site:-scoped search finds indexed listing pages, but automated
    *retrieval* of those pages may violate LinkedIn's ToS and robots.txt.
    The system records candidates from search results only; it does NOT
    log in, bypass authentication, or scrape behind gating.  Always review
    the ToS of any source before enabling automated retrieval.
  - The system does NOT bypass CAPTCHA, anti-bot controls, rate limits, or
    access controls, and does NOT use proxy rotation for evasion.
"""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urlparse

from job_discovery.search.providers.base import (
    ProviderSearchResult,
    RateLimitedError,
    SearchProvider,
    SearchProviderError,
)
from job_discovery.search.query_builder import SOURCE_DOMAINS

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Source detection helpers
# ─────────────────────────────────────────────────────────────────────────────

# Reverse map: domain fragment → source name
_DOMAIN_TO_SOURCE: Dict[str, str] = {v: k for k, v in SOURCE_DOMAINS.items()}
# Also handle bare domains
_BARE_DOMAIN_TO_SOURCE: Dict[str, str] = {
    "linkedin.com":  "linkedin",
    "indeed.com":    "indeed",
    "naukri.com":    "naukri",
    "cutshort.io":   "cutshort",
    "instahyre.com": "instahyre",
    "hirist.tech":   "hirist",
    "wellfound.com": "wellfound",
    "hirect.in":     "hirect",
}


def detect_source_from_url(url: str) -> str:
    """
    Detect which job source a URL belongs to.
    Returns 'generic' when no known source matches.
    """
    try:
        host = urlparse(url).netloc.lower()
        # Strip leading 'www.' prefix only (not arbitrary chars)
        if host.startswith("www."):
            host = host[4:]
    except Exception:
        return "generic"

    for domain, source in _BARE_DOMAIN_TO_SOURCE.items():
        if host == domain or host.endswith("." + domain):
            return source
    return "generic"


# ─────────────────────────────────────────────────────────────────────────────
# WebSearchProvider
# ─────────────────────────────────────────────────────────────────────────────

class WebSearchProvider(SearchProvider):
    """
    Executes site:-scoped queries through a general web search engine and
    routes results to the correct source tag.

    By default, queries are broken out per source so each search is
    site:-scoped.  This maximises precision and avoids one source
    dominating results.

    Args:
        http_client:   An object with a `.get(url, timeout)` method that
                       returns a response-like object with `.text` and
                       `.status_code`.  Inject a mock for tests.
        search_engine: One of "ddg" (DuckDuckGo HTML, default) or "mock".
        request_delay: Seconds to sleep between requests (rate-limit safety).
    """

    def __init__(
        self,
        http_client: Optional[Any] = None,
        *,
        search_engine: str = "ddg",
        request_delay: float = 1.0,
    ) -> None:
        self._http = http_client
        self._engine = search_engine
        self._delay = request_delay

    @property
    def provider_name(self) -> str:
        return f"web_search:{self._engine}"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        source_filter: Optional[List[str]] = None,
    ) -> List[ProviderSearchResult]:
        """
        Run site:-scoped searches across all (or filtered) sources.

        For each source, a scoped query is constructed:
            '<query> site:linkedin.com/jobs'
        Results are tagged with the detected source name.
        """
        sources = source_filter or list(SOURCE_DOMAINS.keys())
        all_results: List[ProviderSearchResult] = []
        per_source = max(1, max_results // len(sources))

        for source_name in sources:
            domain = SOURCE_DOMAINS.get(source_name)
            if not domain:
                continue
            scoped_query = f"{query} site:{domain}"
            try:
                raw = self._execute_search(scoped_query, max_results=per_source)
                for rank, item in enumerate(raw, start=1):
                    url = item.get("url", "")
                    detected = detect_source_from_url(url) or source_name
                    result = ProviderSearchResult(
                        title=item.get("title"),
                        url=url,
                        snippet=item.get("snippet"),
                        source_name=detected,
                        query=query,
                        query_hash=ProviderSearchResult.make_query_hash(query),
                        rank=rank,
                    )
                    all_results.append(result)
            except RateLimitedError:
                logger.warning(
                    "Rate limited on source=%s, query=%r — recording and skipping",
                    source_name,
                    query,
                )
            except SearchProviderError as exc:
                logger.warning(
                    "Search failed for source=%s: %s", source_name, exc
                )
            if self._delay > 0:
                time.sleep(self._delay)

        return all_results[:max_results]

    def build_scoped_url(self, query: str, source_name: str) -> Optional[str]:
        """
        Return the search URL that would be used for a given query+source.
        Useful for debugging / audit logging.
        Always uses DDG format regardless of the active engine.
        """
        domain = SOURCE_DOMAINS.get(source_name)
        if not domain:
            return None
        scoped = f"{query} site:{domain}"
        return f"https://html.duckduckgo.com/html/?q={quote_plus(scoped)}"

    # ------------------------------------------------------------------ #
    # Internal: search engine integration
    # ------------------------------------------------------------------ #

    def _execute_search(
        self, scoped_query: str, *, max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Delegate to the configured search engine.
        Swap this method to change engines without touching the public API.

        Returns list of dicts with keys: title, url, snippet.
        Raises RateLimitedError or SearchProviderError on failure.
        """
        if self._engine == "mock" or self._http is None:
            return []          # no-op for tests without a live engine

        if self._engine == "ddg":
            return self._ddg_search(scoped_query, max_results=max_results)

        raise SearchProviderError(f"Unknown search engine: {self._engine!r}")

    def _ddg_search(
        self, query: str, *, max_results: int
    ) -> List[Dict[str, Any]]:
        """
        DuckDuckGo HTML scraper (no API key required).
        NOTE: DuckDuckGo does not require authentication or CAPTCHA bypass.
        This uses the public HTML endpoint and respects rate limits.
        """
        try:
            from bs4 import BeautifulSoup  # optional dependency
        except ImportError:
            logger.warning(
                "beautifulsoup4 not installed — DuckDuckGo parsing unavailable"
            )
            return []

        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        try:
            resp = self._http.get(url, timeout=15)
        except Exception as exc:
            raise SearchProviderError(f"HTTP error: {exc}") from exc

        if resp.status_code == 429:
            raise RateLimitedError("DuckDuckGo rate limit")
        if resp.status_code != 200:
            raise SearchProviderError(f"HTTP {resp.status_code}")

        soup = BeautifulSoup(resp.text, "html.parser")
        results: List[Dict[str, Any]] = []

        for item in soup.select(".result"):
            title_el = item.select_one(".result__title a")
            snippet_el = item.select_one(".result__snippet")
            if not title_el:
                continue
            href = title_el.get("href", "")
            # DDG redirect URLs — extract actual URL
            if href.startswith("//duckduckgo.com/l/?"):
                from urllib.parse import parse_qs, urlparse as _up
                qs = parse_qs(_up(href).query)
                href = qs.get("uddg", [href])[0]
            results.append(
                {
                    "title": title_el.get_text(strip=True),
                    "url": href,
                    "snippet": snippet_el.get_text(strip=True) if snippet_el else None,
                }
            )
            if len(results) >= max_results:
                break

        return results
