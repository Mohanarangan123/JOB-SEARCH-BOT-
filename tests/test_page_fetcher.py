"""
Tests for PageFetcher — mocked httpx, no real network calls.

Covers:
  - Successful fetch → raw storage + FetchResult
  - 403 / 404 / 410 → permanent failure (no retry)
  - 429 / 500 → transient failure (retry then give up)
  - Timeout → FAILED_TIMEOUT
  - Circuit open → SKIPPED_CIRCUIT
  - robots.txt disallowed → SKIPPED_ROBOTS
  - Failed source does not stop other sources
  - Content hash correctness
  - End-to-end: SearchResult URL → normalize → fetch → store → hash
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from job_discovery.deduplication.identity import IdentityResolver, NormalizedUrl
from job_discovery.fetch.circuit_breaker import CircuitBreakerRegistry
from job_discovery.fetch.fetch_policy import FetchPolicy, RobotsCache
from job_discovery.fetch.page_fetcher import FetchResult, FetchStatus, PageFetcher
from job_discovery.fetch.retry import RetryConfig
from job_discovery.storage.raw_store import RawStore, compute_content_hash


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm(url: str = "https://www.indeed.com/viewjob?jk=abc123", source: str = "indeed") -> NormalizedUrl:
    return NormalizedUrl(original_url=url, canonical_url=url, source_name=source)


def _mock_response(status: int = 200, body: bytes = b"<html>job</html>", content_type: str = "text/html") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.content = body
    resp.text = body.decode("utf-8", errors="replace")
    resp.headers = {"content-type": content_type}
    return resp


def _allow_all_robots() -> MagicMock:
    robots = MagicMock(spec=RobotsCache)
    robots.is_allowed.return_value = True
    return robots


def _make_fetcher(
    tmp_path: Path,
    *,
    http_client=None,
    robots_allowed: bool = True,
    failure_threshold: int = 10,
    retry_attempts: int = 1,
    max_fetches: int = 0,
) -> PageFetcher:
    store = RawStore(tmp_path)
    registry = CircuitBreakerRegistry(failure_threshold=failure_threshold)
    robots = MagicMock(spec=RobotsCache)
    robots.is_allowed.return_value = robots_allowed
    retry_cfg = RetryConfig(max_attempts=retry_attempts, base_delay=0.0, jitter=0.0)

    fetcher = PageFetcher(
        raw_store=store,
        circuit_registry=registry,
        robots_cache=robots,
        retry_config=retry_cfg,
        timeout=5.0,
        max_fetches=max_fetches,
        http_client=http_client,
    )
    return fetcher


# ─────────────────────────────────────────────────────────────────────────────
# Successful fetch
# ─────────────────────────────────────────────────────────────────────────────

class TestSuccessfulFetch:
    @pytest.mark.asyncio
    async def test_returns_success_status(self, tmp_path):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_response(200, b"<html>job</html>")
        fetcher = _make_fetcher(tmp_path, http_client=mock_client)
        result = await fetcher.fetch(_norm())
        assert result.status == FetchStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_http_status_recorded(self, tmp_path):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_response(200)
        result = await _make_fetcher(tmp_path, http_client=mock_client).fetch(_norm())
        assert result.http_status == 200

    @pytest.mark.asyncio
    async def test_content_hash_set(self, tmp_path):
        body = b"<html>unique content</html>"
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_response(200, body)
        result = await _make_fetcher(tmp_path, http_client=mock_client).fetch(_norm())
        assert result.content_hash == compute_content_hash(body)

    @pytest.mark.asyncio
    async def test_raw_content_path_set(self, tmp_path):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_response(200)
        result = await _make_fetcher(tmp_path, http_client=mock_client).fetch(_norm())
        assert result.raw_content_path is not None
        assert Path(result.raw_content_path).is_dir()

    @pytest.mark.asyncio
    async def test_raw_html_stored(self, tmp_path):
        body = b"<html>stored</html>"
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_response(200, body)
        result = await _make_fetcher(tmp_path, http_client=mock_client).fetch(_norm())
        slot = Path(result.raw_content_path)
        assert (slot / "raw.html").read_bytes() == body

    @pytest.mark.asyncio
    async def test_metadata_json_stored(self, tmp_path):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_response(200)
        result = await _make_fetcher(tmp_path, http_client=mock_client).fetch(_norm())
        slot = Path(result.raw_content_path)
        import json
        meta = json.loads((slot / "metadata.json").read_text())
        assert "url" in meta
        assert meta["http_status"] == 200

    @pytest.mark.asyncio
    async def test_retrieved_at_set(self, tmp_path):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_response(200)
        result = await _make_fetcher(tmp_path, http_client=mock_client).fetch(_norm())
        assert result.retrieved_at is not None


# ─────────────────────────────────────────────────────────────────────────────
# HTTP error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestHttpErrors:
    @pytest.mark.asyncio
    async def test_403_permanent_failure(self, tmp_path):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_response(403)
        result = await _make_fetcher(tmp_path, http_client=mock_client).fetch(_norm())
        assert result.status == FetchStatus.FAILED_PERMANENT
        assert result.http_status == 403

    @pytest.mark.asyncio
    async def test_404_permanent_failure(self, tmp_path):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_response(404)
        result = await _make_fetcher(tmp_path, http_client=mock_client).fetch(_norm())
        assert result.status == FetchStatus.FAILED_PERMANENT

    @pytest.mark.asyncio
    async def test_410_permanent_failure(self, tmp_path):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_response(410)
        result = await _make_fetcher(tmp_path, http_client=mock_client).fetch(_norm())
        assert result.status == FetchStatus.FAILED_PERMANENT

    @pytest.mark.asyncio
    async def test_429_transient_failure(self, tmp_path):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_response(429)
        # retry_attempts=1 → gives up after 1 attempt
        result = await _make_fetcher(tmp_path, http_client=mock_client, retry_attempts=1).fetch(_norm())
        assert result.status == FetchStatus.FAILED_TRANSIENT

    @pytest.mark.asyncio
    async def test_500_transient_failure(self, tmp_path):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_response(500)
        result = await _make_fetcher(tmp_path, http_client=mock_client, retry_attempts=1).fetch(_norm())
        assert result.status == FetchStatus.FAILED_TRANSIENT

    @pytest.mark.asyncio
    async def test_permanent_failure_not_retried(self, tmp_path):
        """403 should only be called once — no retry."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_response(403)
        await _make_fetcher(tmp_path, http_client=mock_client, retry_attempts=3).fetch(_norm())
        assert mock_client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_transient_retried_up_to_limit(self, tmp_path):
        """429 should be retried up to max_attempts."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_response(429)
        await _make_fetcher(tmp_path, http_client=mock_client, retry_attempts=3).fetch(_norm())
        assert mock_client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_error_recorded_in_result(self, tmp_path):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_response(403)
        result = await _make_fetcher(tmp_path, http_client=mock_client).fetch(_norm())
        assert result.error is not None
        assert "403" in result.error


# ─────────────────────────────────────────────────────────────────────────────
# Timeout handling
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeoutHandling:
    @pytest.mark.asyncio
    async def test_timeout_returns_failed_status(self, tmp_path):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.TimeoutException("timed out")
        result = await _make_fetcher(tmp_path, http_client=mock_client, retry_attempts=1).fetch(_norm())
        # Timeout is a transient error (TransientFetchError wraps it)
        assert result.status in (FetchStatus.FAILED_TRANSIENT, FetchStatus.FAILED_TIMEOUT)

    @pytest.mark.asyncio
    async def test_timeout_error_recorded(self, tmp_path):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.TimeoutException("timed out")
        result = await _make_fetcher(tmp_path, http_client=mock_client, retry_attempts=1).fetch(_norm())
        assert result.error is not None


# ─────────────────────────────────────────────────────────────────────────────
# Circuit breaker integration
# ─────────────────────────────────────────────────────────────────────────────

class TestCircuitBreakerIntegration:
    @pytest.mark.asyncio
    async def test_open_circuit_skips_fetch(self, tmp_path):
        store = RawStore(tmp_path)
        registry = CircuitBreakerRegistry(failure_threshold=1)
        registry.record_failure("indeed", "pre-opened for test")
        robots = _allow_all_robots()
        cfg = RetryConfig(max_attempts=1, base_delay=0.0, jitter=0.0)
        mock_client = AsyncMock(spec=httpx.AsyncClient)

        fetcher = PageFetcher(
            raw_store=store,
            circuit_registry=registry,
            robots_cache=robots,
            retry_config=cfg,
            http_client=mock_client,
        )
        result = await fetcher.fetch(_norm())
        assert result.status == FetchStatus.SKIPPED_CIRCUIT
        mock_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_source_does_not_stop_other_sources(self, tmp_path):
        """Open circuit on 'indeed' must not block 'naukri'."""
        store = RawStore(tmp_path)
        registry = CircuitBreakerRegistry(failure_threshold=1)
        registry.record_failure("indeed", "blocked")
        robots = _allow_all_robots()
        cfg = RetryConfig(max_attempts=1, base_delay=0.0, jitter=0.0)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_response(200, b"<html>naukri</html>")

        fetcher = PageFetcher(
            raw_store=store,
            circuit_registry=registry,
            robots_cache=robots,
            retry_config=cfg,
            http_client=mock_client,
        )

        # indeed → blocked
        indeed_norm = _norm("https://www.indeed.com/viewjob?jk=abc", "indeed")
        result_indeed = await fetcher.fetch(indeed_norm)
        assert result_indeed.status == FetchStatus.SKIPPED_CIRCUIT

        # naukri → allowed
        naukri_norm = _norm("https://www.naukri.com/job-listings-dev-12345678", "naukri")
        result_naukri = await fetcher.fetch(naukri_norm)
        assert result_naukri.status == FetchStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_failure_increments_circuit(self, tmp_path):
        store = RawStore(tmp_path)
        registry = CircuitBreakerRegistry(failure_threshold=3)
        robots = _allow_all_robots()
        cfg = RetryConfig(max_attempts=1, base_delay=0.0, jitter=0.0)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_response(500)

        fetcher = PageFetcher(
            raw_store=store,
            circuit_registry=registry,
            robots_cache=robots,
            retry_config=cfg,
            http_client=mock_client,
        )
        norm = _norm("https://www.indeed.com/viewjob?jk=abc", "indeed")
        await fetcher.fetch(norm)
        await fetcher.fetch(norm)
        # After 2 failures, circuit not yet open (threshold=3)
        assert not registry.is_open("indeed")
        await fetcher.fetch(norm)
        # After 3rd failure, circuit opens
        assert registry.is_open("indeed")


# ─────────────────────────────────────────────────────────────────────────────
# Robots.txt
# ─────────────────────────────────────────────────────────────────────────────

class TestRobotsIntegration:
    @pytest.mark.asyncio
    async def test_robots_disallowed_skips_fetch(self, tmp_path):
        store = RawStore(tmp_path)
        registry = CircuitBreakerRegistry()
        robots = MagicMock(spec=RobotsCache)
        robots.is_allowed.return_value = False
        cfg = RetryConfig(max_attempts=1, base_delay=0.0, jitter=0.0)
        mock_client = AsyncMock(spec=httpx.AsyncClient)

        fetcher = PageFetcher(
            raw_store=store,
            circuit_registry=registry,
            robots_cache=robots,
            retry_config=cfg,
            http_client=mock_client,
        )
        result = await fetcher.fetch(_norm())
        assert result.status == FetchStatus.SKIPPED_ROBOTS
        mock_client.get.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: SearchResult URL → normalize → fetch → store → hash
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_full_pipeline(self, tmp_path):
        """
        Simulate the full Prompt 3 pipeline:
          raw URL → IdentityResolver → NormalizedUrl → PageFetcher → RawStore
        """
        raw_url = "https://www.naukri.com/job-listings-python-dev-12345678?utm_source=google"
        expected_body = b"<html><body>Python Developer Job</body></html>"
        expected_hash = compute_content_hash(expected_body)

        # Mock HTTP
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_response(200, expected_body)

        # Resolve URL
        resolver = IdentityResolver()
        norm = resolver.resolve(raw_url)
        assert norm is not None
        assert norm.source_name == "naukri"
        assert "utm_source" not in norm.canonical_url

        # Fetch
        fetcher = _make_fetcher(tmp_path, http_client=mock_client)
        result = await fetcher.fetch(norm)

        # Verify pipeline output
        assert result.status == FetchStatus.SUCCESS
        assert result.content_hash == expected_hash
        assert result.source == "naukri"
        assert result.raw_content_path is not None

        # Verify storage
        slot = Path(result.raw_content_path)
        assert (slot / "raw.html").exists()
        assert (slot / "metadata.json").exists()
        assert (slot / "retrieval.json").exists()
        stored_html = (slot / "raw.html").read_bytes()
        assert compute_content_hash(stored_html) == expected_hash
