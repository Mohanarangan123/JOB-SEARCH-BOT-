"""
PageFetcher — async HTTP page retrieval with retry, circuit-breaker integration,
robots.txt compliance, and raw storage.

Design rules:
  - Uses httpx (async) for all HTTP.
  - Uses Tenacity for retry (no hand-rolled loops).
  - Checks FetchPolicy before every request.
  - Records every outcome in a FetchResult (never silently discards failures).
  - Stores raw HTML + metadata via RawStore.
  - No CAPTCHA bypass, no proxy rotation, no stealth browser.
  - Identifies itself with FETCH_USER_AGENT.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from tenacity import RetryError

from job_discovery.deduplication.identity import IdentityResolver, NormalizedUrl
from job_discovery.fetch.circuit_breaker import CircuitBreakerRegistry
from job_discovery.fetch.fetch_policy import (
    FETCH_USER_AGENT,
    FetchDecision,
    FetchPolicy,
    RobotsCache,
)
from job_discovery.fetch.retry import (
    PermanentFetchError,
    RetryConfig,
    TransientFetchError,
    build_retry_decorator,
    is_permanent_failure,
    is_retryable_status,
)
from job_discovery.storage.raw_store import RawStore, compute_content_hash

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FetchStatus / FetchResult
# ─────────────────────────────────────────────────────────────────────────────

class FetchStatus(str, Enum):
    SUCCESS          = "success"
    SKIPPED_CIRCUIT  = "skipped_circuit_open"
    SKIPPED_ROBOTS   = "skipped_robots"
    SKIPPED_BUDGET   = "skipped_budget"
    FAILED_PERMANENT = "failed_permanent"   # 403, 404, 410, etc.
    FAILED_TRANSIENT = "failed_transient"   # exhausted retries
    FAILED_TIMEOUT   = "failed_timeout"
    FAILED_ERROR     = "failed_error"       # unexpected exception


@dataclass
class FetchResult:
    """
    Structured result for one fetch attempt.  Never silently discarded.
    """
    url: str
    canonical_url: str
    source: str
    status: FetchStatus
    http_status: Optional[int] = None
    content_type: Optional[str] = None
    retrieved_at: Optional[datetime] = None
    content_hash: Optional[str] = None
    raw_content_path: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# PageFetcher
# ─────────────────────────────────────────────────────────────────────────────

class PageFetcher:
    """
    Async page fetcher with policy enforcement and raw storage.

    Usage (async context):
        fetcher = PageFetcher(raw_store=store, circuit_registry=registry)
        result = await fetcher.fetch(normalized_url)

    Args:
        raw_store:        RawStore instance for persisting content.
        circuit_registry: CircuitBreakerRegistry shared across sources.
        robots_cache:     RobotsCache (robots.txt compliance).
        retry_config:     Tenacity retry parameters.
        timeout:          HTTP timeout in seconds.
        max_fetches:      Per-run budget (0 = unlimited).
        http_client:      Inject a mock httpx.AsyncClient for tests.
    """

    def __init__(
        self,
        raw_store: RawStore,
        circuit_registry: Optional[CircuitBreakerRegistry] = None,
        robots_cache: Optional[RobotsCache] = None,
        retry_config: Optional[RetryConfig] = None,
        *,
        timeout: float = 30.0,
        max_fetches: int = 0,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._store = raw_store
        self._circuits = circuit_registry or CircuitBreakerRegistry()
        self._robots = robots_cache or RobotsCache()
        self._retry_cfg = retry_config or RetryConfig()
        self._timeout = timeout
        self._http = http_client   # None → created fresh per request
        self._policy = FetchPolicy(
            self._circuits,
            self._robots,
            max_fetches=max_fetches,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def fetch(self, norm: NormalizedUrl) -> FetchResult:
        """
        Fetch a single job page.

        1. Check FetchPolicy (circuit, robots, budget).
        2. Execute HTTP GET with Tenacity retry.
        3. Persist raw content to RawStore.
        4. Return FetchResult.
        """
        url = norm.canonical_url
        source = norm.source_name

        # ── Policy check ────────────────────────────────────────────────
        policy_result = self._policy.check(url, source)
        if policy_result.decision != FetchDecision.ALLOW:
            status_map = {
                FetchDecision.BLOCK_CIRCUIT_OPEN: FetchStatus.SKIPPED_CIRCUIT,
                FetchDecision.BLOCK_ROBOTS:       FetchStatus.SKIPPED_ROBOTS,
                FetchDecision.BLOCK_BUDGET:       FetchStatus.SKIPPED_BUDGET,
            }
            return FetchResult(
                url=norm.original_url,
                canonical_url=url,
                source=source,
                status=status_map.get(policy_result.decision, FetchStatus.FAILED_ERROR),
                error=policy_result.reason,
            )

        # ── Execute with retry ───────────────────────────────────────────
        retry_decorator = build_retry_decorator(self._retry_cfg)
        attempt_count = 0

        @retry_decorator
        async def _do_fetch() -> httpx.Response:
            nonlocal attempt_count
            attempt_count += 1
            client = self._http
            if client is None:
                raise RuntimeError("No HTTP client — use fetch_with_client()")
            try:
                resp = await client.get(
                    url,
                    headers={"User-Agent": FETCH_USER_AGENT},
                    timeout=self._timeout,
                    follow_redirects=True,
                )
            except httpx.TimeoutException as exc:
                raise TransientFetchError(f"Timeout fetching {url!r}: {exc}") from exc
            except httpx.RequestError as exc:
                raise TransientFetchError(f"Request error fetching {url!r}: {exc}") from exc

            if is_retryable_status(resp.status_code):
                raise TransientFetchError(
                    f"HTTP {resp.status_code} for {url!r}", resp.status_code
                )
            if is_permanent_failure(resp.status_code):
                raise PermanentFetchError(
                    f"HTTP {resp.status_code} for {url!r}", resp.status_code
                )
            return resp

        start_ts = datetime.now(timezone.utc)
        try:
            resp = await _do_fetch()
        except PermanentFetchError as exc:
            self._circuits.record_failure(source, str(exc))
            return FetchResult(
                url=norm.original_url,
                canonical_url=url,
                source=source,
                status=FetchStatus.FAILED_PERMANENT,
                http_status=exc.status_code,
                retrieved_at=start_ts,
                error=str(exc),
                retry_count=attempt_count - 1,
            )
        except TransientFetchError as exc:
            self._circuits.record_failure(source, str(exc))
            return FetchResult(
                url=norm.original_url,
                canonical_url=url,
                source=source,
                status=FetchStatus.FAILED_TRANSIENT,
                http_status=exc.status_code,
                retrieved_at=start_ts,
                error=str(exc),
                retry_count=attempt_count - 1,
            )
        except RetryError as exc:
            self._circuits.record_failure(source, str(exc))
            return FetchResult(
                url=norm.original_url,
                canonical_url=url,
                source=source,
                status=FetchStatus.FAILED_TRANSIENT,
                retrieved_at=start_ts,
                error=str(exc),
                retry_count=attempt_count - 1,
            )
        except httpx.TimeoutException as exc:
            self._circuits.record_failure(source, str(exc))
            return FetchResult(
                url=norm.original_url,
                canonical_url=url,
                source=source,
                status=FetchStatus.FAILED_TIMEOUT,
                retrieved_at=start_ts,
                error=str(exc),
                retry_count=attempt_count - 1,
            )
        except Exception as exc:
            self._circuits.record_failure(source, str(exc))
            logger.exception("Unexpected error fetching %r", url)
            return FetchResult(
                url=norm.original_url,
                canonical_url=url,
                source=source,
                status=FetchStatus.FAILED_ERROR,
                retrieved_at=start_ts,
                error=str(exc),
                retry_count=attempt_count - 1,
            )

        # ── Success path ────────────────────────────────────────────────
        self._circuits.record_success(source)
        self._policy.consume()

        raw_bytes = resp.content
        content_hash = compute_content_hash(raw_bytes)
        content_type = resp.headers.get("content-type", "")

        # Derive a stable job_key from content_hash (short prefix)
        job_key = content_hash[:16]

        slot = self._store.save(
            source=source,
            job_key=job_key,
            raw_html=raw_bytes,
            raw_text="",           # text extraction is Prompt 4
            metadata={
                "url": url,
                "canonical_url": url,
                "source": source,
                "http_status": resp.status_code,
                "content_type": content_type,
                "content_hash": content_hash,
                "retrieved_at": start_ts.isoformat(),
            },
            retrieval={
                "original_url": norm.original_url,
                "canonical_url": url,
                "source": source,
                "retry_count": attempt_count - 1,
                "retrieved_at": start_ts.isoformat(),
            },
        )

        return FetchResult(
            url=norm.original_url,
            canonical_url=url,
            source=source,
            status=FetchStatus.SUCCESS,
            http_status=resp.status_code,
            content_type=content_type,
            retrieved_at=start_ts,
            content_hash=content_hash,
            raw_content_path=str(slot),
            retry_count=attempt_count - 1,
        )

    async def fetch_many(self, urls: List[NormalizedUrl]) -> List[FetchResult]:
        """Fetch multiple URLs sequentially (rate-limit safe)."""
        results = []
        for norm in urls:
            result = await self.fetch(norm)
            results.append(result)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: create a PageFetcher with a fresh async client
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_url(
    url: str,
    *,
    raw_store: RawStore,
    circuit_registry: Optional[CircuitBreakerRegistry] = None,
    retry_config: Optional[RetryConfig] = None,
    timeout: float = 30.0,
) -> FetchResult:
    """
    One-shot async helper: resolve, fetch, and store a single URL.
    Creates its own httpx.AsyncClient.
    """
    resolver = IdentityResolver()
    norm = resolver.resolve(url)
    if norm is None:
        return FetchResult(
            url=url,
            canonical_url=url,
            source="generic",
            status=FetchStatus.FAILED_ERROR,
            error="URL failed validation",
        )

    async with httpx.AsyncClient() as client:
        fetcher = PageFetcher(
            raw_store=raw_store,
            circuit_registry=circuit_registry,
            retry_config=retry_config,
            timeout=timeout,
            http_client=client,
        )
        return await fetcher.fetch(norm)
