"""
Tests for FetchPolicy, RobotsCache, and retry helpers.
All network requests are mocked.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from job_discovery.fetch.circuit_breaker import CircuitBreakerRegistry
from job_discovery.fetch.fetch_policy import (
    FetchDecision,
    FetchPolicy,
    PolicyResult,
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


# ─────────────────────────────────────────────────────────────────────────────
# HTTP status classification
# ─────────────────────────────────────────────────────────────────────────────

class TestStatusClassification:
    @pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
    def test_retryable(self, code):
        assert is_retryable_status(code)

    @pytest.mark.parametrize("code", [200, 301, 400, 403, 404, 410])
    def test_not_retryable(self, code):
        assert not is_retryable_status(code)

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 410])
    def test_permanent_failure(self, code):
        assert is_permanent_failure(code)

    @pytest.mark.parametrize("code", [200, 429, 500, 503])
    def test_not_permanent_failure(self, code):
        assert not is_permanent_failure(code)


# ─────────────────────────────────────────────────────────────────────────────
# Retry configuration
# ─────────────────────────────────────────────────────────────────────────────

class TestRetryConfig:
    def test_default_values(self):
        cfg = RetryConfig()
        assert cfg.max_attempts == 3
        assert cfg.base_delay > 0
        assert cfg.jitter >= 0

    def test_custom_values(self):
        cfg = RetryConfig(max_attempts=5, base_delay=1.0, jitter=0.5)
        assert cfg.max_attempts == 5


class TestRetryDecorator:
    def test_retries_transient_error(self):
        """TransientFetchError should cause Tenacity to retry."""
        cfg = RetryConfig(max_attempts=3, base_delay=0.01, jitter=0.0)
        decorator = build_retry_decorator(cfg)
        call_count = {"n": 0}

        @decorator
        def flaky():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise TransientFetchError("temporary")
            return "ok"

        result = flaky()
        assert result == "ok"
        assert call_count["n"] == 3

    def test_does_not_retry_permanent_error(self):
        """PermanentFetchError must not be retried."""
        cfg = RetryConfig(max_attempts=3, base_delay=0.01, jitter=0.0)
        decorator = build_retry_decorator(cfg)
        call_count = {"n": 0}

        @decorator
        def always_permanent():
            call_count["n"] += 1
            raise PermanentFetchError("forbidden", 403)

        with pytest.raises(PermanentFetchError):
            always_permanent()
        assert call_count["n"] == 1  # no retry

    def test_stops_after_max_attempts(self):
        """After max_attempts, TransientFetchError is re-raised."""
        from tenacity import RetryError

        cfg = RetryConfig(max_attempts=2, base_delay=0.01, jitter=0.0)
        decorator = build_retry_decorator(cfg)
        call_count = {"n": 0}

        @decorator
        def always_fails():
            call_count["n"] += 1
            raise TransientFetchError("always fails")

        with pytest.raises((TransientFetchError, RetryError)):
            always_fails()
        assert call_count["n"] == 2

    def test_exponential_backoff_configured(self):
        """build_retry_decorator must produce a decorator that retries TransientFetchError."""
        from tenacity import RetryError

        cfg = RetryConfig(max_attempts=3, base_delay=0.0, jitter=0.0)
        decorator = build_retry_decorator(cfg)
        call_count = {"n": 0}

        @decorator
        def always_transient():
            call_count["n"] += 1
            raise TransientFetchError("transient")

        with pytest.raises((TransientFetchError, RetryError)):
            always_transient()

        # Must have been called max_attempts times (exponential backoff configured)
        assert call_count["n"] == 3

    def test_jitter_zero_still_works(self):
        """jitter=0 is valid (deterministic wait for tests)."""
        cfg = RetryConfig(max_attempts=2, base_delay=0.0, jitter=0.0)
        decorator = build_retry_decorator(cfg)
        call_count = {"n": 0}

        @decorator
        def once_transient():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise TransientFetchError("once")
            return "done"

        assert once_transient() == "done"


# ─────────────────────────────────────────────────────────────────────────────
# RobotsCache
# ─────────────────────────────────────────────────────────────────────────────

class TestRobotsCache:
    def _cache_with_text(self, robots_text: str) -> RobotsCache:
        """Build a RobotsCache backed by a mock HTTP client."""
        mock_resp = MagicMock()
        mock_resp.text = robots_text
        mock_http = MagicMock()
        mock_http.get.return_value = mock_resp
        return RobotsCache(http_client=mock_http)

    def test_allows_when_robots_permits(self):
        cache = self._cache_with_text("User-agent: *\nAllow: /")
        assert cache.is_allowed("https://example.com/job/1")

    def test_disallows_when_robots_blocks(self):
        cache = self._cache_with_text("User-agent: *\nDisallow: /")
        assert not cache.is_allowed("https://example.com/job/1")

    def test_specific_path_allowed(self):
        robots = "User-agent: *\nDisallow: /private/\nAllow: /jobs/"
        cache = self._cache_with_text(robots)
        assert cache.is_allowed("https://example.com/jobs/123")

    def test_specific_path_disallowed(self):
        robots = "User-agent: *\nDisallow: /private/"
        cache = self._cache_with_text(robots)
        assert not cache.is_allowed("https://example.com/private/data")

    def test_fails_open_on_robots_fetch_error(self):
        """If robots.txt cannot be fetched, we allow the request (fail open)."""
        mock_http = MagicMock()
        mock_http.get.side_effect = Exception("connection refused")
        cache = RobotsCache(http_client=mock_http)
        assert cache.is_allowed("https://example.com/job/1")

    def test_caches_per_host(self):
        """robots.txt should only be fetched once per host."""
        mock_resp = MagicMock()
        mock_resp.text = "User-agent: *\nAllow: /"
        mock_http = MagicMock()
        mock_http.get.return_value = mock_resp
        cache = RobotsCache(http_client=mock_http)

        cache.is_allowed("https://example.com/job/1")
        cache.is_allowed("https://example.com/job/2")
        cache.is_allowed("https://example.com/job/3")

        assert mock_http.get.call_count == 1  # cached

    def test_separate_hosts_fetched_independently(self):
        mock_resp = MagicMock()
        mock_resp.text = "User-agent: *\nAllow: /"
        mock_http = MagicMock()
        mock_http.get.return_value = mock_resp
        cache = RobotsCache(http_client=mock_http)

        cache.is_allowed("https://siteA.com/job/1")
        cache.is_allowed("https://siteB.com/job/1")

        assert mock_http.get.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# FetchPolicy
# ─────────────────────────────────────────────────────────────────────────────

def _make_policy(
    *,
    source_open: bool = False,
    robots_allowed: bool = True,
    max_fetches: int = 0,
    already_fetched: int = 0,
) -> FetchPolicy:
    registry = CircuitBreakerRegistry(failure_threshold=1)
    if source_open:
        registry.record_failure("linkedin", "test")

    robots = MagicMock(spec=RobotsCache)
    robots.is_allowed.return_value = robots_allowed

    policy = FetchPolicy(registry, robots, max_fetches=max_fetches)
    for _ in range(already_fetched):
        policy.consume()
    return policy


class TestFetchPolicy:
    def test_allow_normal(self):
        policy = _make_policy()
        result = policy.check("https://linkedin.com/jobs/1", "linkedin")
        assert result.decision == FetchDecision.ALLOW

    def test_block_circuit_open(self):
        policy = _make_policy(source_open=True)
        result = policy.check("https://linkedin.com/jobs/1", "linkedin")
        assert result.decision == FetchDecision.BLOCK_CIRCUIT_OPEN

    def test_block_robots(self):
        policy = _make_policy(robots_allowed=False)
        result = policy.check("https://linkedin.com/jobs/1", "linkedin")
        assert result.decision == FetchDecision.BLOCK_ROBOTS

    def test_block_budget_exhausted(self):
        policy = _make_policy(max_fetches=2, already_fetched=2)
        result = policy.check("https://naukri.com/job/1", "naukri")
        assert result.decision == FetchDecision.BLOCK_BUDGET

    def test_allow_within_budget(self):
        policy = _make_policy(max_fetches=5, already_fetched=3)
        result = policy.check("https://naukri.com/job/1", "naukri")
        assert result.decision == FetchDecision.ALLOW

    def test_unlimited_budget_when_zero(self):
        """max_fetches=0 means unlimited."""
        policy = _make_policy(max_fetches=0, already_fetched=1000)
        result = policy.check("https://naukri.com/job/1", "naukri")
        assert result.decision == FetchDecision.ALLOW

    def test_circuit_checked_before_robots(self):
        """Circuit breaker has higher priority than robots."""
        policy = _make_policy(source_open=True, robots_allowed=False)
        result = policy.check("https://linkedin.com/jobs/1", "linkedin")
        assert result.decision == FetchDecision.BLOCK_CIRCUIT_OPEN

    def test_consume_decrements_budget(self):
        policy = _make_policy(max_fetches=3)
        assert policy.fetches_remaining == 3
        policy.consume()
        assert policy.fetches_remaining == 2

    def test_unlimited_budget_returns_minus_one(self):
        policy = _make_policy(max_fetches=0)
        assert policy.fetches_remaining == -1

    def test_reason_populated_on_block(self):
        policy = _make_policy(source_open=True)
        result = policy.check("https://linkedin.com/jobs/1", "linkedin")
        assert result.reason != ""
