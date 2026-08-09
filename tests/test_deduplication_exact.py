"""
Tests for exact-match deduplication (V1 only):
  - canonical URL dedup
  - external platform job ID dedup
  - content hash dedup
  - no semantic / embedding dedup
  - job identity using only exact-match signals
"""
from __future__ import annotations

import pytest

from job_discovery.deduplication.exact_match import (
    ExactMatchDeduplicator,
    UrlValidationError,
    normalize_url,
)
from job_discovery.deduplication.identity import IdentityResolver, NormalizedUrl


# ─────────────────────────────────────────────────────────────────────────────
# Exact URL deduplication
# ─────────────────────────────────────────────────────────────────────────────

class TestCanonicalUrlDedup:
    def setup_method(self):
        self.dedup = ExactMatchDeduplicator()

    def test_first_url_not_duplicate(self):
        _, is_dup = self.dedup.check("https://naukri.com/job-listings-python-12345678")
        assert not is_dup

    def test_same_url_duplicate(self):
        url = "https://naukri.com/job-listings-python-12345678"
        self.dedup.check(url)
        _, is_dup = self.dedup.check(url)
        assert is_dup

    def test_tracking_params_stripped_before_dedup(self):
        url1 = "https://linkedin.com/jobs/view/111?trk=abc"
        url2 = "https://linkedin.com/jobs/view/111?refId=xyz"
        self.dedup.check(url1)
        _, is_dup = self.dedup.check(url2)
        assert is_dup, "Same job after param-strip must be detected as duplicate"

    def test_different_job_ids_not_duplicate(self):
        self.dedup.check("https://linkedin.com/jobs/view/111")
        _, is_dup = self.dedup.check("https://linkedin.com/jobs/view/222")
        assert not is_dup

    def test_cross_source_not_deduplicated(self):
        """Same job title on LinkedIn and Naukri = different URLs = NOT dedup."""
        self.dedup.check("https://www.linkedin.com/jobs/view/111")
        _, is_dup = self.dedup.check("https://www.naukri.com/job-listings-senior-python-dev-111")
        assert not is_dup

    def test_canonical_url_returned(self):
        canonical, _ = self.dedup.check("https://indeed.com/viewjob?jk=abc&from=serp")
        assert "from=" not in canonical


# ─────────────────────────────────────────────────────────────────────────────
# External job ID deduplication
# ─────────────────────────────────────────────────────────────────────────────

class TestExternalIdDedup:
    """
    External ID dedup is handled at the repository layer (MongoDB unique index
    on external_job_id + source_name). These tests verify the data model supports
    it and that the identity resolver extracts IDs correctly.
    """

    def test_linkedin_id_extracted(self):
        resolver = IdentityResolver()
        r = resolver.resolve("https://www.linkedin.com/jobs/view/3456789012/")
        assert r is not None
        # canonical URL is stable and contains the ID
        assert "3456789012" in r.canonical_url

    def test_indeed_jk_extracted(self):
        resolver = IdentityResolver()
        r = resolver.resolve("https://www.indeed.com/viewjob?jk=abc123def456")
        assert r is not None
        assert "abc123def456" in r.canonical_url

    def test_naukri_id_in_canonical(self):
        resolver = IdentityResolver()
        r = resolver.resolve("https://www.naukri.com/job-listings-python-dev-12345678")
        assert r is not None
        assert "12345678" in r.canonical_url

    def test_same_external_id_different_sources_not_deduped(self):
        """Cross-source duplicates allowed in V1."""
        resolver = IdentityResolver()
        r1 = resolver.resolve("https://www.linkedin.com/jobs/view/12345")
        r2 = resolver.resolve("https://www.naukri.com/job-listings-x-12345678")
        assert r1 is not None
        assert r2 is not None
        assert not r2.is_duplicate  # different canonical URLs → not a dup


# ─────────────────────────────────────────────────────────────────────────────
# Content hash deduplication
# ─────────────────────────────────────────────────────────────────────────────

class TestContentHashDedup:
    """
    Content hash dedup: identical HTML content → same hash → no new version.
    This is tested at the VersionTracker level.
    """

    def test_identical_html_produces_same_hash(self):
        from job_discovery.storage.raw_store import compute_content_hash
        html = b"<html><body>Python Developer at TechCorp, Chennai</body></html>"
        h1 = compute_content_hash(html)
        h2 = compute_content_hash(html)
        assert h1 == h2

    def test_different_html_produces_different_hash(self):
        from job_discovery.storage.raw_store import compute_content_hash
        h1 = compute_content_hash(b"<html>Job A</html>")
        h2 = compute_content_hash(b"<html>Job B</html>")
        assert h1 != h2

    def test_whitespace_normalised_in_hash(self):
        from job_discovery.storage.raw_store import compute_content_hash
        h1 = compute_content_hash("  <html>same</html>  ")
        h2 = compute_content_hash("<html>same</html>")
        assert h1 == h2


# ─────────────────────────────────────────────────────────────────────────────
# No semantic deduplication
# ─────────────────────────────────────────────────────────────────────────────

class TestNoSemanticDedup:
    def test_resolver_has_no_embedding_method(self):
        resolver = IdentityResolver()
        assert not hasattr(resolver, "embed")
        assert not hasattr(resolver, "similarity")
        assert not hasattr(resolver, "vector_search")

    def test_deduplicator_has_no_embedding_method(self):
        dedup = ExactMatchDeduplicator()
        assert not hasattr(dedup, "embed")
        assert not hasattr(dedup, "similarity_threshold")

    def test_near_duplicate_different_urls_both_accepted(self):
        """Two listings with similar text but different URLs must BOTH be kept."""
        dedup = ExactMatchDeduplicator()
        dedup.check("https://linkedin.com/jobs/view/111")
        # Very similar URL with different job ID
        _, is_dup = dedup.check("https://linkedin.com/jobs/view/112")
        assert not is_dup, "Different job IDs must not be deduplicated"


# ─────────────────────────────────────────────────────────────────────────────
# Job Identity — exact-match signals only
# ─────────────────────────────────────────────────────────────────────────────

class TestJobIdentity:
    def test_identity_based_on_canonical_url(self):
        resolver = IdentityResolver()
        r = resolver.resolve("https://www.naukri.com/job-listings-python-dev-12345678?sid=x")
        assert r is not None
        # Canonical URL is the identity key
        assert "sid=" not in r.canonical_url
        assert "12345678" in r.canonical_url

    def test_source_detected_in_identity(self):
        resolver = IdentityResolver()
        r = resolver.resolve("https://www.linkedin.com/jobs/view/999/")
        assert r.source_name == "linkedin"

    def test_duplicate_flag_is_identity_signal(self):
        resolver = IdentityResolver()
        r1 = resolver.resolve("https://cutshort.io/job/dev-abcd1234")
        r2 = resolver.resolve("https://cutshort.io/job/dev-abcd1234?ref=google")
        assert not r1.is_duplicate
        assert r2.is_duplicate  # same canonical → same identity

    def test_no_cross_source_identity_resolution(self):
        """V1: no semantic identity across sources."""
        resolver = IdentityResolver()
        r1 = resolver.resolve("https://www.linkedin.com/jobs/view/123")
        r2 = resolver.resolve("https://www.indeed.com/viewjob?jk=abc123")
        assert r1.source_name == "linkedin"
        assert r2.source_name == "indeed"
        # Both records exist independently — not merged
        assert not r2.is_duplicate
