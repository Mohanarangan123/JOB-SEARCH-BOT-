"""
Tests for all source adapters:
  - URL recognition
  - Canonical URL normalisation
  - External job ID extraction
  - Tier 2 unavailable / graceful degradation behaviour
  - Source registry routing
  - Design constraint: adapters do not construct search queries
"""
from __future__ import annotations

import pytest

from job_discovery.sources.base import (
    JobSourceAdapter,
    SourceRegistry,
    SourceTier,
    UnavailableError,
)
from job_discovery.sources.cutshort import CutshortAdapter
from job_discovery.sources.generic import GenericAdapter
from job_discovery.sources.hirect import HirectAdapter
from job_discovery.sources.hirist import HiristAdapter
from job_discovery.sources.indeed import IndeedAdapter
from job_discovery.sources.instahyre import InstahyreAdapter
from job_discovery.sources.linkedin import LinkedInAdapter
from job_discovery.sources.naukri import NaukriAdapter
from job_discovery.sources.wellfound import WellfoundAdapter
from job_discovery.sources import build_default_registry


# ─────────────────────────────────────────────────────────────────────────────
# LinkedIn — Tier 1
# ─────────────────────────────────────────────────────────────────────────────

class TestLinkedInAdapter:
    def setup_method(self):
        self.adapter = LinkedInAdapter()

    def test_meta_name(self):
        assert self.adapter.meta.name == "linkedin"

    def test_tier1(self):
        assert self.adapter.tier == SourceTier.TIER1

    # URL recognition
    @pytest.mark.parametrize("url", [
        "https://www.linkedin.com/jobs/view/3456789012/",
        "https://linkedin.com/jobs/view/1234567890",
        "https://www.linkedin.com/jobs/collections/recommended/?currentJobId=3456789012",
    ])
    def test_recognises_valid_urls(self, url):
        assert self.adapter.recognises_url(url)

    @pytest.mark.parametrize("url", [
        "https://www.naukri.com/job-listings-python-dev",
        "https://indeed.com/viewjob?jk=abc",
        "https://google.com",
    ])
    def test_rejects_other_urls(self, url):
        assert not self.adapter.recognises_url(url)

    # External ID extraction
    @pytest.mark.parametrize("url,expected_id", [
        ("https://www.linkedin.com/jobs/view/3456789012/", "3456789012"),
        ("https://linkedin.com/jobs/view/1234567890", "1234567890"),
        ("https://www.linkedin.com/jobs/collections/recommended/?currentJobId=9876543210", "9876543210"),
    ])
    def test_extract_external_id(self, url, expected_id):
        assert self.adapter.extract_external_id(url) == expected_id

    def test_extract_external_id_returns_none_for_non_job_url(self):
        assert self.adapter.extract_external_id("https://linkedin.com/in/profile/") is None

    # Canonical URL
    def test_canonical_url_from_job_id(self):
        url = "https://www.linkedin.com/jobs/view/3456789012/?trk=sometracking"
        canonical = self.adapter.canonical_url(url)
        assert "3456789012" in canonical
        assert "trk" not in canonical

    def test_canonical_url_strips_tracking(self):
        url = "https://www.linkedin.com/jobs/view/111/?refId=abc&trackingId=xyz"
        canonical = self.adapter.canonical_url(url)
        assert "refId" not in canonical
        assert "trackingId" not in canonical

    def test_canonical_url_idempotent(self):
        url = "https://www.linkedin.com/jobs/view/3456789012/"
        assert self.adapter.canonical_url(url) == self.adapter.canonical_url(
            self.adapter.canonical_url(url)
        )

    # Stubs
    def test_fetch_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.adapter.fetch_job("https://www.linkedin.com/jobs/view/1/")

    def test_access_notes_documented(self):
        assert len(self.adapter.meta.access_notes) > 0
        assert "ToS" in self.adapter.meta.access_notes or "terms" in self.adapter.meta.access_notes.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Indeed — Tier 1
# ─────────────────────────────────────────────────────────────────────────────

class TestIndeedAdapter:
    def setup_method(self):
        self.adapter = IndeedAdapter()

    def test_meta_name(self):
        assert self.adapter.meta.name == "indeed"

    def test_tier1(self):
        assert self.adapter.tier == SourceTier.TIER1

    @pytest.mark.parametrize("url", [
        "https://www.indeed.com/viewjob?jk=abc123def456",
        "https://indeed.com/rc/clk?jk=abc123def456",
        "https://in.indeed.com/viewjob?jk=abc123",
    ])
    def test_recognises_valid_urls(self, url):
        assert self.adapter.recognises_url(url)

    def test_rejects_non_indeed(self):
        assert not self.adapter.recognises_url("https://linkedin.com/jobs/1")

    @pytest.mark.parametrize("url,expected_jk", [
        ("https://www.indeed.com/viewjob?jk=abc123def456", "abc123def456"),
        ("https://indeed.com/rc/clk?jk=xyz789abc012", "xyz789abc012"),
    ])
    def test_extract_external_id(self, url, expected_jk):
        assert self.adapter.extract_external_id(url) == expected_jk

    def test_extract_external_id_none_when_missing(self):
        assert self.adapter.extract_external_id("https://indeed.com/jobs") is None

    def test_canonical_url_with_jk(self):
        url = "https://www.indeed.com/viewjob?jk=abc123&from=serp"
        canonical = self.adapter.canonical_url(url)
        assert "jk=abc123" in canonical
        assert "from=" not in canonical

    def test_canonical_url_idempotent(self):
        url = "https://www.indeed.com/viewjob?jk=abc123"
        assert self.adapter.canonical_url(url) == self.adapter.canonical_url(
            self.adapter.canonical_url(url)
        )

    def test_fetch_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.adapter.fetch_job("https://indeed.com/viewjob?jk=abc")


# ─────────────────────────────────────────────────────────────────────────────
# Naukri — Tier 1
# ─────────────────────────────────────────────────────────────────────────────

class TestNaukriAdapter:
    def setup_method(self):
        self.adapter = NaukriAdapter()

    def test_meta_name(self):
        assert self.adapter.meta.name == "naukri"

    def test_tier1(self):
        assert self.adapter.tier == SourceTier.TIER1

    @pytest.mark.parametrize("url", [
        "https://www.naukri.com/job-listings-python-developer-bangalore-12345678",
        "https://naukri.com/job-listings-backend-engineer-99887766",
    ])
    def test_recognises_valid_urls(self, url):
        assert self.adapter.recognises_url(url)

    def test_rejects_non_naukri(self):
        assert not self.adapter.recognises_url("https://indeed.com/viewjob?jk=abc")

    @pytest.mark.parametrize("url,expected_id", [
        ("https://www.naukri.com/job-listings-python-developer-12345678", "12345678"),
        ("https://www.naukri.com/job-listings-backend-99887766?jid=99887766", "99887766"),
    ])
    def test_extract_external_id(self, url, expected_id):
        result = self.adapter.extract_external_id(url)
        assert result == expected_id

    def test_canonical_url_strips_query(self):
        url = "https://www.naukri.com/job-listings-python-dev-12345678?sid=xyz"
        canonical = self.adapter.canonical_url(url)
        assert "?" not in canonical
        assert "12345678" in canonical

    def test_fetch_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.adapter.fetch_job("https://naukri.com/job-listings-x-12345678")


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 adapters — graceful unavailable handling
# ─────────────────────────────────────────────────────────────────────────────

class TestTier2Adapters:
    """All Tier 2 adapters must raise UnavailableError on fetch_job()."""

    @pytest.mark.parametrize("adapter_cls,url", [
        (CutshortAdapter,  "https://cutshort.io/job/python-dev-abcd1234"),
        (InstahyreAdapter, "https://www.instahyre.com/job-98765/"),
        (HiristAdapter,    "https://www.hirist.tech/j/python-developer-12345"),
        (WellfoundAdapter, "https://wellfound.com/jobs/123456"),
        (HirectAdapter,    "https://hirect.in/job/abc123"),
        (GenericAdapter,   "https://somesite.com/jobs/1"),
    ])
    def test_fetch_raises_unavailable(self, adapter_cls, url):
        adapter = adapter_cls()
        with pytest.raises(UnavailableError):
            adapter.fetch_job(url)

    @pytest.mark.parametrize("adapter_cls", [
        CutshortAdapter, InstahyreAdapter, HiristAdapter,
        WellfoundAdapter, HirectAdapter,
    ])
    def test_tier2_flag(self, adapter_cls):
        adapter = adapter_cls()
        assert adapter.tier == SourceTier.TIER2

    def test_tier2_access_notes_present(self):
        for adapter_cls in [CutshortAdapter, InstahyreAdapter, HiristAdapter,
                            WellfoundAdapter, HirectAdapter]:
            adapter = adapter_cls()
            assert len(adapter.meta.access_notes) > 0, (
                f"{adapter_cls.__name__} has no access notes"
            )


class TestCutshortAdapter:
    def setup_method(self):
        self.adapter = CutshortAdapter()

    def test_recognises_url(self):
        assert self.adapter.recognises_url("https://cutshort.io/job/python-dev-abcd1234")

    def test_rejects_other(self):
        assert not self.adapter.recognises_url("https://linkedin.com/jobs/1")

    def test_canonical_url(self):
        url = "https://cutshort.io/job/python-dev-abcd1234?ref=xyz"
        canonical = self.adapter.canonical_url(url)
        assert "ref=" not in canonical

    def test_extract_external_id(self):
        url = "https://cutshort.io/job/python-dev-abcd1234"
        result = self.adapter.extract_external_id(url)
        assert result is not None


class TestInstahyreAdapter:
    def setup_method(self):
        self.adapter = InstahyreAdapter()

    def test_recognises_url(self):
        assert self.adapter.recognises_url("https://www.instahyre.com/job-12345/")

    def test_extract_external_id(self):
        url = "https://www.instahyre.com/job-12345/"
        assert self.adapter.extract_external_id(url) == "12345"

    def test_canonical_url(self):
        url = "https://www.instahyre.com/job-12345/?ref=abc"
        canonical = self.adapter.canonical_url(url)
        assert "ref=" not in canonical


class TestHiristAdapter:
    def setup_method(self):
        self.adapter = HiristAdapter()

    def test_recognises_url(self):
        assert self.adapter.recognises_url("https://www.hirist.tech/j/python-dev-12345")

    def test_extract_external_id(self):
        url = "https://www.hirist.tech/j/python-developer-12345"
        assert self.adapter.extract_external_id(url) == "12345"

    def test_canonical_url(self):
        url = "https://www.hirist.tech/j/python-dev-12345?q=test"
        canonical = self.adapter.canonical_url(url)
        assert "?" not in canonical


class TestWellfoundAdapter:
    def setup_method(self):
        self.adapter = WellfoundAdapter()

    @pytest.mark.parametrize("url", [
        "https://wellfound.com/jobs/123456",
        "https://angel.co/l/some-startup/123456",
    ])
    def test_recognises_url(self, url):
        assert self.adapter.recognises_url(url)

    def test_canonical_url(self):
        url = "https://wellfound.com/jobs/123456?ref=foo"
        canonical = self.adapter.canonical_url(url)
        assert "ref=" not in canonical


class TestHirectAdapter:
    def setup_method(self):
        self.adapter = HirectAdapter()

    def test_recognises_url(self):
        assert self.adapter.recognises_url("https://hirect.in/job/abc123")

    def test_canonical_url(self):
        url = "https://hirect.in/job/abc123?campaign=google"
        canonical = self.adapter.canonical_url(url)
        assert "campaign" not in canonical


class TestGenericAdapter:
    def setup_method(self):
        self.adapter = GenericAdapter()

    def test_recognises_any_url(self):
        assert self.adapter.recognises_url("https://randomjobsite.com/job/1234")

    def test_does_not_recognise_empty_string(self):
        assert not self.adapter.recognises_url("")

    def test_canonical_url_strips_query(self):
        url = "https://somesite.com/job/1234?ref=abc"
        canonical = self.adapter.canonical_url(url)
        assert "ref=" not in canonical

    def test_extract_external_id_returns_path(self):
        url = "https://somesite.com/jobs/position/1234"
        result = self.adapter.extract_external_id(url)
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# Source Registry routing
# ─────────────────────────────────────────────────────────────────────────────

class TestSourceRegistry:
    def setup_method(self):
        self.registry = build_default_registry()

    def test_registry_has_all_adapters(self):
        names = [a.source_name for a in self.registry.all_adapters()]
        expected = ["linkedin", "indeed", "naukri", "cutshort",
                    "instahyre", "hirist", "wellfound", "hirect", "generic"]
        for name in expected:
            assert name in names

    def test_routes_linkedin_url(self):
        adapter = self.registry.route("https://www.linkedin.com/jobs/view/123456789/")
        assert adapter is not None
        assert adapter.source_name == "linkedin"

    def test_routes_indeed_url(self):
        adapter = self.registry.route("https://www.indeed.com/viewjob?jk=abc123")
        assert adapter is not None
        assert adapter.source_name == "indeed"

    def test_routes_naukri_url(self):
        adapter = self.registry.route("https://www.naukri.com/job-listings-python-12345678")
        assert adapter is not None
        assert adapter.source_name == "naukri"

    def test_routes_cutshort_url(self):
        adapter = self.registry.route("https://cutshort.io/job/python-dev-abcd1234")
        assert adapter is not None
        assert adapter.source_name == "cutshort"

    def test_routes_unknown_url_to_generic(self):
        adapter = self.registry.route("https://randomjobsite.com/job/1234")
        assert adapter is not None
        assert adapter.source_name == "generic"

    def test_get_by_name_linkedin(self):
        adapter = self.registry.get_by_name("linkedin")
        assert adapter is not None
        assert isinstance(adapter, LinkedInAdapter)

    def test_get_by_name_unknown_returns_none(self):
        adapter = self.registry.get_by_name("notasite")
        assert adapter is None

    def test_adapters_do_not_have_search_method(self):
        """Core design constraint: adapters must not own search logic."""
        for adapter in self.registry.all_adapters():
            assert not hasattr(adapter, "search"), (
                f"{adapter.__class__.__name__} must not have a search() method"
            )

    def test_routing_priority_specific_before_generic(self):
        """LinkedIn URL must be routed to LinkedIn adapter, not generic."""
        url = "https://www.linkedin.com/jobs/view/987654321/"
        adapter = self.registry.route(url)
        assert adapter.source_name == "linkedin"


# ─────────────────────────────────────────────────────────────────────────────
# Canonical URL idempotency for all Tier 1 adapters
# ─────────────────────────────────────────────────────────────────────────────

class TestCanonicalUrlIdempotency:
    @pytest.mark.parametrize("adapter_cls,url", [
        (LinkedInAdapter, "https://www.linkedin.com/jobs/view/3456789012/"),
        (IndeedAdapter,   "https://www.indeed.com/viewjob?jk=abc123def456"),
        (NaukriAdapter,   "https://www.naukri.com/job-listings-python-12345678"),
        (CutshortAdapter, "https://cutshort.io/job/python-dev-abcd1234"),
        (InstahyreAdapter,"https://www.instahyre.com/job-12345/"),
        (HiristAdapter,   "https://www.hirist.tech/j/python-developer-12345"),
        (WellfoundAdapter,"https://wellfound.com/jobs/123456"),
        (HirectAdapter,   "https://hirect.in/job/abc123"),
        (GenericAdapter,  "https://genericsite.com/job/5678"),
    ])
    def test_idempotent(self, adapter_cls, url):
        adapter = adapter_cls()
        once = adapter.canonical_url(url)
        twice = adapter.canonical_url(once)
        assert once == twice, f"{adapter_cls.__name__}: canonical not idempotent"
