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
import re
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
        self.failure_log: List[Dict[str, Any]] = []

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
        failures = 0
        if not sources:
            logger.info("SEARCH_SUCCESS_WITH_ZERO_RESULTS query=%r source_filter=%r reason=no-source-targets", query, source_filter)
            return []

        per_source = max(1, max_results // len(sources))
        logger.info(
            "SEARCH_DISCOVERY_START query=%r source_filter=%s max_results=%s per_source=%s",
            query,
            sources,
            max_results,
            per_source,
        )

        for source_name in sources:
            domain = SOURCE_DOMAINS.get(source_name)
            if not domain:
                logger.warning(
                    "SEARCH_PARSE_ERROR source=%s status=SEARCH_PARSE_ERROR query=%r reason=unknown-source-domain",
                    source_name,
                    query,
                )
                continue

            scoped_query = f"{query} site:{domain}"
            engine_url = self.build_scoped_url(query, source_name)
            logger.info(
                "DISCOVERY_BUILD source=%s generated_query=%r final_site_scoped_query=%r search_engine_url=%s",
                source_name,
                query,
                scoped_query,
                engine_url,
            )
            try:
                raw = self._execute_search(scoped_query, max_results=per_source)
                if raw:
                    logger.info(
                        "DISCOVERY_SEARCH_RESULT source=%s query=%r parsed_results=%d success_status=SEARCH_SUCCESS",
                        source_name,
                        query,
                        len(raw),
                    )
                else:
                    logger.info(
                        "DISCOVERY_SEARCH_RESULT source=%s query=%r parsed_results=%d success_status=SEARCH_SUCCESS_WITH_ZERO_RESULTS",
                        source_name,
                        query,
                        len(raw),
                    )
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
            except RateLimitedError as exc:
                failures += 1
                self._record_failure(
                    source_name,
                    query,
                    scoped_query,
                    engine_url,
                    exc,
                )
                logger.error(
                    "SEARCH_RATE_LIMITED source=%s query=%r final_site_scoped_query=%r search_engine_url=%s reason=%s",
                    source_name,
                    query,
                    scoped_query,
                    engine_url,
                    exc,
                )
            except SearchProviderError as exc:
                failures += 1
                self._record_failure(
                    source_name,
                    query,
                    scoped_query,
                    engine_url,
                    exc,
                )
                classification = getattr(exc, "classification", "SEARCH_HTTP_ERROR")
                logger.error(
                    "SEARCH_FAILURE source=%s query=%r final_site_scoped_query=%r search_engine_url=%s classification=%s http_status=%s reason=%s",
                    source_name,
                    query,
                    scoped_query,
                    engine_url,
                    classification,
                    exc.http_status if hasattr(exc, "http_status") else None,
                    exc,
                )
            except Exception as exc:
                failures += 1
                self._record_failure(
                    source_name,
                    query,
                    scoped_query,
                    engine_url,
                    SearchProviderError(
                        str(exc),
                        classification="SEARCH_NETWORK_ERROR",
                        provider=self.provider_name,
                        source=source_name,
                        query=query,
                        reason=str(exc),
                    ),
                )
                logger.error(
                    "SEARCH_NETWORK_ERROR source=%s query=%r final_site_scoped_query=%r search_engine_url=%s reason=%s",
                    source_name,
                    query,
                    scoped_query,
                    engine_url,
                    exc,
                )
            if self._delay > 0:
                time.sleep(self._delay)

        if not all_results and failures:
            logger.error(
                "DISCOVERY_FAILURE_SUMMARY query=%r provider=%s failures=%s parsed_results=%s",
                query,
                self.provider_name,
                failures,
                len(all_results),
            )

        return all_results[:max_results]

    def _record_failure(
        self,
        source_name: str,
        query: str,
        site_scoped_query: str,
        search_engine_url: str,
        exc: Exception,
    ) -> None:
        detail = {
            "provider": self.provider_name,
            "source": source_name,
            "query": query,
            "site_scoped_query": site_scoped_query,
            "search_engine_url": search_engine_url,
            "http_status": getattr(exc, "http_status", None),
            "classification": getattr(exc, "classification", "SEARCH_NETWORK_ERROR"),
            "reason": getattr(exc, "reason", str(exc)) or str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.failure_log.append(detail)

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
            raise SearchProviderError("SEARCH_PARSE_ERROR: beautifulsoup4 unavailable")

        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Upgrade-Insecure-Requests": "1",
        }
        try:
            try:
                resp = self._http.get(url, timeout=15, headers=headers)
            except TypeError:
                resp = self._http.get(url, timeout=15)
        except Exception as exc:
            logger.error("SEARCH_NETWORK_ERROR query=%r engine_url=%s reason=%s", query, url, exc)
            raise SearchProviderError(
                f"SEARCH_NETWORK_ERROR: HTTP error: {exc}",
                classification="SEARCH_NETWORK_ERROR",
                provider=self.provider_name,
                query=query,
                reason=str(exc),
            ) from exc

        response_text = resp.text or ""
        logger.info(
            "SEARCH_ENGINE_TRACE query=%r engine_url=%s source=%s status=%s content_length=%s",
            query,
            url,
            "ddg",
            resp.status_code,
            len(response_text),
        )

        if resp.status_code == 200 and self._looks_malformed_html(response_text):
            logger.error(
                "SEARCH_PARSE_ERROR query=%r engine_url=%s status=%s content_length=%s reason=malformed-html-shape",
                query,
                url,
                resp.status_code,
                len(response_text),
            )
            raise SearchProviderError(
                "SEARCH_PARSE_ERROR: malformed HTML response from search engine",
                classification="SEARCH_PARSE_ERROR",
                provider=self.provider_name,
                query=query,
                http_status=resp.status_code,
                reason="Malformed HTML response from search engine",
            )

        if resp.status_code == 429:
            logger.error(
                "SEARCH_RATE_LIMITED query=%r engine_url=%s status=%s content_length=%s",
                query,
                url,
                resp.status_code,
                len(response_text),
            )
            raise RateLimitedError(
                "DuckDuckGo rate limit",
                provider=self.provider_name,
                query=query,
                http_status=429,
                reason="DuckDuckGo rate limit",
            )
        if resp.status_code == 403:
            logger.error(
                "SEARCH_HTTP_ERROR query=%r engine_url=%s status=%s content_length=%s reason=http_403_forbidden",
                query,
                url,
                resp.status_code,
                len(response_text),
            )
            raise SearchProviderError(
                "SEARCH_HTTP_ERROR: HTTP 403 forbidden",
                classification="SEARCH_HTTP_ERROR",
                provider=self.provider_name,
                query=query,
                http_status=403,
                reason="HTTP 403 forbidden",
            )
        if resp.status_code == 202:
            logger.error(
                "SEARCH_CHALLENGED query=%r engine_url=%s status=%s content_length=%s reason=duckduckgo-html-challenge-page-202",
                query,
                url,
                resp.status_code,
                len(response_text),
            )
            raise SearchProviderError(
                "SEARCH_CHALLENGED: DuckDuckGo returned 202 challenge page",
                classification="SEARCH_CHALLENGED",
                provider=self.provider_name,
                query=query,
                http_status=202,
                reason="DuckDuckGo HTML returned 202 challenge page",
            )
        if resp.status_code != 200:
            logger.error(
                "SEARCH_HTTP_ERROR query=%r engine_url=%s status=%s content_length=%s",
                query,
                url,
                resp.status_code,
                len(response_text),
            )
            raise SearchProviderError(
                f"SEARCH_HTTP_ERROR: HTTP {resp.status_code}",
                classification="SEARCH_HTTP_ERROR",
                provider=self.provider_name,
                query=query,
                http_status=resp.status_code,
                reason=f"HTTP {resp.status_code}",
            )

        soup = BeautifulSoup(response_text, "html.parser")
        challenge_form = soup.select_one("#challenge-form") or soup.select_one("form#challenge-form")
        anomaly_form = soup.select_one("#anomaly-form") or soup.select_one("form#img-form")
        body_text = soup.get_text(" ", strip=True).lower()
        challenge_markers = (
            "unfortunately, bots use duckduckgo too"
            in body_text
            or "select all squares containing a duck" in body_text
            or "challenge" in body_text
            or "anomaly" in body_text
            or challenge_form is not None
            or anomaly_form is not None
        )
        if challenge_markers:
            logger.error(
                "SEARCH_CHALLENGED query=%r engine_url=%s status=%s content_length=%s reason=duckduckgo-bot-challenge-page",
                query,
                url,
                resp.status_code,
                len(response_text),
            )
            raise SearchProviderError(
                "SEARCH_CHALLENGED: DuckDuckGo HTML returned bot-challenge page",
                classification="SEARCH_CHALLENGED",
                provider=self.provider_name,
                query=query,
                http_status=resp.status_code,
                reason="DuckDuckGo HTML returned bot-challenge page",
            )

        results: List[Dict[str, Any]] = []
        items = soup.select(".result")
        logger.info("SEARCH_HTML_PARSE query=%r result_nodes=%d", query, len(items))
        for item in items:
            title_el = item.select_one(".result__title a") or item.select_one(".result__a") or item.select_one("a.result__a")
            snippet_el = item.select_one(".result__snippet")
            if not title_el:
                logger.warning("SEARCH_PARSE_ERROR query=%r status=SEARCH_PARSE_ERROR reason=result-node-without-title", query)
                continue
            href = title_el.get("href", "")
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

        if resp.status_code == 200 and len(results) == 0:
            logger.info(
                "SEARCH_SUCCESS_WITH_ZERO_RESULTS query=%r engine_url=%s status=%s content_length=%s parsed_results=0",
                query,
                url,
                resp.status_code,
                len(response_text),
            )
        return results

    def _looks_malformed_html(self, response_text: str) -> bool:
        """Heuristic for malformed DDG HTML that parser recovery could incorrectly accept."""
        if not response_text:
            return False
        text = response_text.lower()
        if 'class="result"' not in text and 'class="result' not in text:
            return False
        if '<a ' in text and '</a>' not in text:
            return True
        if '<h2 class="result__title"' in text and '</h2>' not in text:
            return True
        if '<div class="result__title"' in text and '</div>' not in text:
            return True
        if '<div class="result__snippet"' in text and '</div>' not in text:
            return True
        if text.count('<a ') > text.count('</a>'):
            return True
        if text.count('<div class="result"') > text.count('</div>'):
            return True
        return False
