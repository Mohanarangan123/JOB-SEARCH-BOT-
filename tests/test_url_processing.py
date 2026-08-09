"""
Tests for URL validation, normalization, and exact-match deduplication.
"""
from __future__ import annotations

import pytest

from job_discovery.deduplication.exact_match import (
    ExactMatchDeduplicator,
    UrlValidationError,
    normalize_url,
    validate_url,
)
from job_discovery.deduplication.identity import IdentityResolver, NormalizedUrl


# ─────────────────────────────────────────────────────────────────────────────
# URL Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateUrl:
    def test_valid_https(self):
        validate_url("https://www.linkedin.com/jobs/view/123456789/")  # no exception

    def test_valid_http(self):
        validate_url("http://example.com/job/1")

    def test_empty_raises(self):
        with pytest.raises(UrlValidationError, match="empty"):
            validate_url("")

    def test_whitespace_raises(self):
        with pytest.raises(UrlValidationError):
            validate_url("   ")

    def test_no_scheme_raises(self):
        with pytest.raises(UrlValidationError, match="scheme"):
            validate_url("linkedin.com/jobs/view/123")

    def test_ftp_scheme_raises(self):
        with pytest.raises(UrlValidationError, match="scheme"):
            validate_url("ftp://example.com/file")

    def test_no_host_raises(self):
        with pytest.raises(UrlValidationError):
            validate_url("https:///just-a-path")

    def test_javascript_scheme_raises(self):
        with pytest.raises(UrlValidationError):
            validate_url("javascript:alert(1)")


# ─────────────────────────────────────────────────────────────────────────────
# URL Normalization
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeUrl:
    def test_lowercase_scheme(self):
        n = normalize_url("HTTPS://linkedin.com/jobs/view/1")
        assert n.startswith("https://")

    def test_lowercase_host(self):
        n = normalize_url("https://WWW.LinkedIn.COM/jobs/view/1")
        assert "linkedin.com" in n
        assert "LinkedIn" not in n

    def test_strips_fragment(self):
        n = normalize_url("https://indeed.com/viewjob?jk=abc#apply")
        assert "#" not in n

    def test_strips_utm_params(self):
        n = normalize_url("https://naukri.com/job?utm_source=google&utm_medium=cpc")
        assert "utm_" not in n

    def test_strips_tracking_params(self):
        n = normalize_url("https://www.linkedin.com/jobs/view/1?refId=abc&trk=xyz")
        assert "refId" not in n
        assert "trk" not in n

    def test_keeps_job_id_param(self):
        n = normalize_url("https://indeed.com/viewjob?jk=abc123&from=serp")
        assert "jk=abc123" in n
        assert "from=" not in n

    def test_sorts_query_params(self):
        n1 = normalize_url("https://example.com/job?z=1&a=2")
        n2 = normalize_url("https://example.com/job?a=2&z=1")
        assert n1 == n2

    def test_removes_default_http_port(self):
        n = normalize_url("http://example.com:80/job/1")
        assert ":80" not in n

    def test_removes_default_https_port(self):
        n = normalize_url("https://example.com:443/job/1")
        assert ":443" not in n

    def test_keeps_non_default_port(self):
        n = normalize_url("https://example.com:8443/job/1")
        assert "8443" in n

    def test_strips_trailing_slash_on_path(self):
        n1 = normalize_url("https://example.com/job/123/")
        n2 = normalize_url("https://example.com/job/123")
        assert n1 == n2

    def test_keeps_root_slash(self):
        n = normalize_url("https://example.com/")
        assert n.endswith("/") or n == "https://example.com"

    def test_idempotent(self):
        url = "https://linkedin.com/jobs/view/3456789012?trk=abc"
        assert normalize_url(normalize_url(url)) == normalize_url(url)

    def test_strips_ref_param(self):
        n = normalize_url("https://cutshort.io/job/dev-abc?ref=linkedin")
        assert "ref=" not in n

    @pytest.mark.parametrize("url,expected_contains", [
        ("https://www.indeed.com/viewjob?jk=abc123", "jk=abc123"),
        ("https://naukri.com/job-listing-12345678", "12345678"),
    ])
    def test_keeps_identifier_params(self, url, expected_contains):
        assert expected_contains in normalize_url(url)


# ─────────────────────────────────────────────────────────────────────────────
# ExactMatchDeduplicator
# ─────────────────────────────────────────────────────────────────────────────

class TestExactMatchDeduplicator:
    def test_first_url_not_duplicate(self):
        d = ExactMatchDeduplicator()
        _, is_dup = d.check("https://linkedin.com/jobs/view/111")
        assert not is_dup

    def test_same_url_is_duplicate(self):
        d = ExactMatchDeduplicator()
        d.check("https://linkedin.com/jobs/view/111")
        _, is_dup = d.check("https://linkedin.com/jobs/view/111")
        assert is_dup

    def test_normalized_duplicates_detected(self):
        d = ExactMatchDeduplicator()
        d.check("https://linkedin.com/jobs/view/111?trk=abc")
        _, is_dup = d.check("https://linkedin.com/jobs/view/111?refId=xyz")
        assert is_dup

    def test_different_urls_not_duplicate(self):
        d = ExactMatchDeduplicator()
        d.check("https://linkedin.com/jobs/view/111")
        _, is_dup = d.check("https://linkedin.com/jobs/view/222")
        assert not is_dup

    def test_returns_canonical_url(self):
        d = ExactMatchDeduplicator()
        canonical, _ = d.check("https://linkedin.com/jobs/view/111?trk=abc")
        assert "trk" not in canonical

    def test_invalid_url_raises(self):
        d = ExactMatchDeduplicator()
        with pytest.raises(UrlValidationError):
            d.check("not-a-url")

    def test_seen_count(self):
        d = ExactMatchDeduplicator()
        d.check("https://a.com/1")
        d.check("https://b.com/2")
        d.check("https://a.com/1")  # duplicate
        assert d.seen_count() == 2

    def test_reset_clears_state(self):
        d = ExactMatchDeduplicator()
        d.check("https://a.com/1")
        d.reset()
        assert d.seen_count() == 0
        _, is_dup = d.check("https://a.com/1")
        assert not is_dup

    def test_filter_new_returns_only_new(self):
        d = ExactMatchDeduplicator()
        urls = [
            "https://a.com/1",
            "https://b.com/2",
            "https://a.com/1?utm_source=x",  # dup of first after normalisation
        ]
        result = d.filter_new(urls)
        assert len(result) == 2
        canonical_urls = [r[1] for r in result]
        assert all("utm_" not in c for c in canonical_urls)

    def test_filter_new_skips_invalid(self):
        d = ExactMatchDeduplicator()
        result = d.filter_new(["https://valid.com/1", "not-a-url", "ftp://bad.com"])
        assert len(result) == 1

    def test_is_seen_without_registering(self):
        d = ExactMatchDeduplicator()
        d.check("https://a.com/1")
        assert d.is_seen("https://a.com/1")
        assert not d.is_seen("https://b.com/2")


# ─────────────────────────────────────────────────────────────────────────────
# IdentityResolver
# ─────────────────────────────────────────────────────────────────────────────

class TestIdentityResolver:
    def test_resolves_valid_url(self):
        r = IdentityResolver()
        result = r.resolve("https://www.linkedin.com/jobs/view/123456/")
        assert result is not None
        assert isinstance(result, NormalizedUrl)

    def test_source_detected(self):
        r = IdentityResolver()
        result = r.resolve("https://www.linkedin.com/jobs/view/123456/")
        assert result.source_name == "linkedin"

    def test_naukri_source_detected(self):
        r = IdentityResolver()
        result = r.resolve("https://www.naukri.com/job-listings-python-12345678")
        assert result.source_name == "naukri"

    def test_generic_source_for_unknown(self):
        r = IdentityResolver()
        result = r.resolve("https://randomjobsite.com/job/1234")
        assert result.source_name == "generic"

    def test_invalid_url_returns_none(self):
        r = IdentityResolver()
        assert r.resolve("not-a-url") is None
        assert r.resolve("") is None

    def test_duplicate_flagged(self):
        r = IdentityResolver()
        r1 = r.resolve("https://www.linkedin.com/jobs/view/111/")
        r2 = r.resolve("https://www.linkedin.com/jobs/view/111/")
        assert r1 is not None and not r1.is_duplicate
        assert r2 is not None and r2.is_duplicate

    def test_resolve_many(self):
        r = IdentityResolver()
        urls = [
            "https://www.linkedin.com/jobs/view/1/",
            "https://www.indeed.com/viewjob?jk=abc",
            "not-a-url",
            "https://www.naukri.com/job-listings-dev-12345678",
        ]
        results = r.resolve_many(urls)
        assert len(results) == 3  # invalid URL filtered out

    def test_reset_clears_dedup_state(self):
        r = IdentityResolver()
        r.resolve("https://www.linkedin.com/jobs/view/111/")
        r.reset()
        result = r.resolve("https://www.linkedin.com/jobs/view/111/")
        assert result is not None and not result.is_duplicate

    def test_original_url_preserved(self):
        r = IdentityResolver()
        original = "https://www.linkedin.com/jobs/view/111/?trk=abc"
        result = r.resolve(original)
        assert result.original_url == original
        assert "trk" not in result.canonical_url
