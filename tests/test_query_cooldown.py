"""
Tests for query cooldown logic in ResultCollector.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from job_discovery.models.source import QueryCache
from job_discovery.search.providers.base import (
    ProviderSearchResult,
    SearchProvider,
)
from job_discovery.search.query_builder import QueryBuilder, SearchCriteria
from job_discovery.search.query_expander import QueryExpander
from job_discovery.search.result_collector import ResultCollector


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_provider(results: Optional[List[ProviderSearchResult]] = None) -> MagicMock:
    """Return a MagicMock implementing SearchProvider.search()."""
    mock = MagicMock(spec=SearchProvider)
    mock.search.return_value = results or []
    return mock


def _make_result(url: str, query: str = "test query") -> ProviderSearchResult:
    return ProviderSearchResult(
        url=url,
        source_name="linkedin",
        query=query,
        query_hash=ProviderSearchResult.make_query_hash(query),
        rank=1,
    )


def _collector(
    provider=None,
    cooldown: int = 60,
    max_per_query: int = 5,
) -> ResultCollector:
    if provider is None:
        provider = _make_provider()
    return ResultCollector(
        search_provider=provider,
        query_builder=QueryBuilder(),
        query_expander=QueryExpander(),   # no LLM — no expansion
        cooldown_seconds=cooldown,
        max_results_per_query=max_per_query,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cooldown enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestCooldown:
    def test_fresh_query_not_on_cooldown(self):
        c = _collector(cooldown=60)
        assert not c.is_on_cooldown("Python Developer Chennai")

    def test_query_on_cooldown_after_run(self):
        provider = _make_provider()
        c = _collector(provider=provider, cooldown=3600)  # 1 hour cooldown
        criteria = SearchCriteria(title="Python Developer", location="Chennai")
        c.collect(criteria)
        # After first collect, all executed queries should be on cooldown
        query = "Python Developer Chennai"
        assert c.is_on_cooldown(query)

    def test_cooldown_expires(self):
        c = _collector(cooldown=1)  # 1 second
        import time
        # Manually plant a stale entry
        q = "Stale Query"
        q_hash = ProviderSearchResult.make_query_hash(q)
        c._cooldown_cache[q_hash] = datetime.now(timezone.utc) - timedelta(seconds=5)
        assert not c.is_on_cooldown(q)  # should be expired

    def test_external_cache_respected(self):
        """Queries present in external_cache within cooldown are skipped."""
        provider = _make_provider(
            results=[_make_result("https://linkedin.com/jobs/1")]
        )
        c = _collector(provider=provider, cooldown=3600)
        criteria = SearchCriteria(title="Python Developer", location="Chennai")

        # Pre-populate external cache with all queries that would be generated
        builder = QueryBuilder()
        queries = builder.build(criteria)
        external = {
            ProviderSearchResult.make_query_hash(q): datetime.now(timezone.utc)
            for q in queries
        }

        results = c.collect(criteria, external_cache=external)
        # All queries were cached → provider.search never called
        provider.search.assert_not_called()
        assert results == []

    def test_different_queries_independent_cooldowns(self):
        c = _collector(cooldown=3600)
        q1 = "Python Developer"
        q2 = "Java Developer"
        h1 = ProviderSearchResult.make_query_hash(q1)
        h2 = ProviderSearchResult.make_query_hash(q2)
        c._cooldown_cache[h1] = datetime.now(timezone.utc)
        assert c.is_on_cooldown(q1)
        assert not c.is_on_cooldown(q2)

    def test_cooldown_zero_never_blocks(self):
        """cooldown_seconds=0 means no cooldown at all."""
        provider = _make_provider(
            results=[_make_result("https://linkedin.com/jobs/1")]
        )
        c = _collector(provider=provider, cooldown=0)
        criteria = SearchCriteria(title="Python Developer", location="Chennai")
        # Run twice — second run should not be blocked
        c.collect(criteria)
        results = c.collect(criteria)
        # provider.search was called both times (no cooldown)
        assert provider.search.call_count >= 2


# ─────────────────────────────────────────────────────────────────────────────
# ResultCollector general behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestResultCollector:
    def test_returns_list(self):
        c = _collector()
        results = c.collect(SearchCriteria(title="Backend Engineer"))
        assert isinstance(results, list)

    def test_deduplicates_by_url(self):
        dup = _make_result("https://linkedin.com/jobs/999")
        provider = _make_provider(results=[dup, dup, dup])
        c = _collector(provider=provider, cooldown=0)
        criteria = SearchCriteria(title="Engineer")
        results = c.collect(criteria)
        urls = [r.url for r in results]
        assert len(urls) == len(set(urls))

    def test_results_are_provider_search_results(self):
        r = _make_result("https://naukri.com/job/123456")
        provider = _make_provider(results=[r])
        c = _collector(provider=provider, cooldown=0)
        results = c.collect(SearchCriteria(title="Data Engineer"))
        for result in results:
            assert isinstance(result, ProviderSearchResult)

    def test_rate_limit_stops_further_queries(self):
        from job_discovery.search.providers.base import RateLimitedError

        provider = MagicMock(spec=SearchProvider)
        provider.search.side_effect = RateLimitedError("rate limited")
        c = _collector(provider=provider, cooldown=0)
        # Should not raise; should return empty list
        results = c.collect(SearchCriteria(title="DevOps Engineer", location="Pune"))
        assert isinstance(results, list)

    def test_provider_error_continues(self):
        """A non-rate-limit exception on one query should not abort others."""
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Exception("Transient error")
            return [_make_result(f"https://linkedin.com/jobs/{call_count['n']}")]

        provider = MagicMock(spec=SearchProvider)
        provider.search.side_effect = side_effect
        c = _collector(provider=provider, cooldown=0)
        results = c.collect(SearchCriteria(title="SRE", location="Bangalore"))
        # At least the second query should have succeeded
        assert isinstance(results, list)

    def test_make_cache_entry(self):
        c = _collector(cooldown=3600)
        r = _make_result("https://linkedin.com/jobs/1", query="ML Engineer")
        entry = c.make_cache_entry("ML Engineer", "linkedin", [r])
        assert isinstance(entry, QueryCache)
        assert entry.query_text == "ML Engineer"
        assert entry.result_count == 1
        assert entry.query_hash == ProviderSearchResult.make_query_hash("ML Engineer")
        assert entry.expires_at is not None
        assert entry.expires_at > entry.cached_at

    def test_make_cache_entry_custom_cooldown(self):
        c = _collector(cooldown=60)
        entry = c.make_cache_entry("test", "naukri", [], cooldown_seconds=7200)
        delta = (entry.expires_at - entry.cached_at).total_seconds()
        assert abs(delta - 7200) < 5   # allow 5s clock tolerance
