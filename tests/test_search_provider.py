"""
Tests for SearchProvider interface, WebSearchProvider, and source detection.
All tests use mocked providers — no live search engines.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from job_discovery.search.providers.base import (
    ProviderSearchResult,
    RateLimitedError,
    SearchProvider,
    SearchProviderError,
)
from job_discovery.search.providers.web_search import (
    WebSearchProvider,
    detect_source_from_url,
)
from job_discovery.search.query_builder import SOURCE_DOMAINS


# ─────────────────────────────────────────────────────────────────────────────
# Mock provider for interface testing
# ─────────────────────────────────────────────────────────────────────────────

class MockSearchProvider(SearchProvider):
    """Concrete mock implementing the SearchProvider interface."""

    def __init__(self, canned_results: Optional[List[ProviderSearchResult]] = None):
        self._results = canned_results or []
        self.search_calls: list = []

    @property
    def provider_name(self) -> str:
        return "mock"

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        source_filter: Optional[List[str]] = None,
    ) -> List[ProviderSearchResult]:
        self.search_calls.append({"query": query, "max_results": max_results})
        return self._results[:max_results]


def _make_result(url: str, source: str = "linkedin", query: str = "test") -> ProviderSearchResult:
    return ProviderSearchResult(
        title="Software Engineer",
        url=url,
        snippet="Great opportunity",
        source_name=source,
        query=query,
        query_hash=ProviderSearchResult.make_query_hash(query),
        rank=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SearchProvider interface
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchProviderInterface:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            SearchProvider()  # type: ignore

    def test_mock_implements_interface(self):
        provider = MockSearchProvider()
        assert isinstance(provider, SearchProvider)

    def test_provider_name(self):
        provider = MockSearchProvider()
        assert provider.provider_name == "mock"

    def test_search_returns_list(self):
        r = _make_result("https://linkedin.com/jobs/1")
        provider = MockSearchProvider(canned_results=[r])
        results = provider.search("Python Developer")
        assert isinstance(results, list)

    def test_search_respects_max_results(self):
        results_list = [_make_result(f"https://linkedin.com/jobs/{i}") for i in range(20)]
        provider = MockSearchProvider(canned_results=results_list)
        results = provider.search("Python Developer", max_results=5)
        assert len(results) == 5

    def test_search_returns_provider_search_results(self):
        r = _make_result("https://linkedin.com/jobs/1")
        provider = MockSearchProvider(canned_results=[r])
        results = provider.search("test")
        for result in results:
            assert isinstance(result, ProviderSearchResult)

    def test_search_empty_when_no_results(self):
        provider = MockSearchProvider(canned_results=[])
        results = provider.search("obscure query no results")
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# ProviderSearchResult validation
# ─────────────────────────────────────────────────────────────────────────────

class TestProviderSearchResult:
    def test_minimal_construction(self):
        r = ProviderSearchResult(
            url="https://linkedin.com/jobs/123",
            source_name="linkedin",
            query="Python Developer",
            query_hash="abc123",
        )
        assert r.url == "https://linkedin.com/jobs/123"
        assert r.title is None
        assert r.snippet is None
        assert r.rank is None

    def test_full_construction(self):
        r = ProviderSearchResult(
            title="Python Engineer",
            url="https://naukri.com/job/1234567",
            snippet="We are looking for...",
            source_name="naukri",
            query="Python Developer Chennai",
            query_hash=ProviderSearchResult.make_query_hash("Python Developer Chennai"),
            rank=2,
        )
        assert r.title == "Python Engineer"
        assert r.rank == 2

    def test_discovered_at_auto_set(self):
        r = _make_result("https://indeed.com/viewjob?jk=abc")
        assert r.discovered_at is not None
        assert isinstance(r.discovered_at, datetime)

    def test_query_hash_on_result(self):
        query = "Data Scientist Bangalore"
        r = ProviderSearchResult(
            url="https://linkedin.com/jobs/1",
            source_name="linkedin",
            query=query,
            query_hash=ProviderSearchResult.make_query_hash(query),
        )
        expected = ProviderSearchResult.make_query_hash(query)
        assert r.query_hash == expected

    def test_model_dump(self):
        r = _make_result("https://linkedin.com/jobs/9")
        d = r.model_dump()
        assert "url" in d
        assert "source_name" in d
        assert "query_hash" in d


# ─────────────────────────────────────────────────────────────────────────────
# Source detection from URL
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectSourceFromUrl:
    @pytest.mark.parametrize("url,expected", [
        ("https://www.linkedin.com/jobs/view/123456789/", "linkedin"),
        ("https://linkedin.com/jobs/view/987654/", "linkedin"),
        ("https://www.indeed.com/viewjob?jk=abc123", "indeed"),
        ("https://indeed.com/rc/clk?jk=xyz", "indeed"),
        ("https://www.naukri.com/job-listings-python-dev-123456", "naukri"),
        ("https://cutshort.io/job/python-dev-abcd1234", "cutshort"),
        ("https://www.instahyre.com/job-12345/", "instahyre"),
        ("https://www.hirist.tech/j/python-developer-12345", "hirist"),
        ("https://wellfound.com/jobs/123456", "wellfound"),
        ("https://hirect.in/job/abc123", "hirect"),
        ("https://unknown-site.com/job/1", "generic"),
        ("https://glassdoor.com/job/1234", "generic"),
        ("not-a-url", "generic"),
    ])
    def test_detection(self, url: str, expected: str):
        assert detect_source_from_url(url) == expected

    def test_www_stripped(self):
        assert detect_source_from_url("https://www.linkedin.com/jobs/view/1") == "linkedin"

    def test_subdomain_handled(self):
        # e.g. in.linkedin.com
        result = detect_source_from_url("https://in.linkedin.com/jobs/view/1")
        assert result == "linkedin"


# ─────────────────────────────────────────────────────────────────────────────
# WebSearchProvider (mock engine — no live HTTP)
# ─────────────────────────────────────────────────────────────────────────────

class TestWebSearchProviderMock:
    """WebSearchProvider with engine='mock' returns empty lists (no HTTP)."""

    def setup_method(self):
        self.provider = WebSearchProvider(search_engine="mock", request_delay=0)

    def test_provider_name(self):
        assert "web_search" in self.provider.provider_name

    def test_search_returns_list(self):
        results = self.provider.search("Python Developer Chennai")
        assert isinstance(results, list)

    def test_search_returns_empty_without_http_client(self):
        # No HTTP client injected → _execute_search returns []
        results = self.provider.search("anything")
        assert results == []

    def test_build_scoped_url_linkedin(self):
        url = self.provider.build_scoped_url("Python Developer", "linkedin")
        assert url is not None
        assert "linkedin" in url
        assert "Python" in url or "python" in url.lower()

    def test_build_scoped_url_unknown_source(self):
        url = self.provider.build_scoped_url("query", "unknown_source")
        assert url is None

    def test_source_filter_respected(self):
        """With mock engine, we just verify no exception and correct return type."""
        results = self.provider.search(
            "Python Developer",
            max_results=5,
            source_filter=["linkedin", "naukri"],
        )
        assert isinstance(results, list)

    def test_rate_limited_source_skipped(self):
        """
        Simulate rate limiting on one source — provider should skip and continue.
        """
        provider = WebSearchProvider(search_engine="mock", request_delay=0)

        original_execute = provider._execute_search

        call_count = {"n": 0}

        def patched_execute(scoped_query, *, max_results):
            call_count["n"] += 1
            if "linkedin" in scoped_query:
                raise RateLimitedError("Simulated rate limit")
            return []

        provider._execute_search = patched_execute
        results = provider.search("Python Developer", source_filter=["linkedin", "naukri"])
        assert isinstance(results, list)
        # linkedin was rate-limited; naukri was attempted
        assert call_count["n"] >= 1

    def test_provider_error_source_skipped(self):
        """A SearchProviderError on one source should not crash the whole search."""
        provider = WebSearchProvider(search_engine="mock", request_delay=0)

        def patched_execute(scoped_query, *, max_results):
            raise SearchProviderError("Simulated error")

        provider._execute_search = patched_execute
        results = provider.search("Python Developer", source_filter=["naukri"])
        assert isinstance(results, list)

    def test_site_scoped_query_constructed(self):
        """Verify site:-scoped query is built correctly for each source."""
        captured = []

        provider = WebSearchProvider(search_engine="mock", request_delay=0)

        def capture_execute(scoped_query, *, max_results):
            captured.append(scoped_query)
            return []

        provider._execute_search = capture_execute
        provider.search("Python Developer", source_filter=["linkedin", "naukri"])

        assert any("site:linkedin.com/jobs" in q for q in captured)
        assert any("site:naukri.com" in q for q in captured)

    def test_adapters_do_not_search(self):
        """Verify adapters don't have a search() method (design constraint)."""
        from job_discovery.sources.linkedin import LinkedInAdapter
        from job_discovery.sources.naukri import NaukriAdapter
        adapter_li = LinkedInAdapter()
        adapter_nk = NaukriAdapter()
        assert not hasattr(adapter_li, "search"), "LinkedIn adapter must not have search()"
        assert not hasattr(adapter_nk, "search"), "Naukri adapter must not have search()"
