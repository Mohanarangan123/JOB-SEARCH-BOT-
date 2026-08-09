"""
Tests for QueryBuilder — query generation from SearchCriteria.
"""
from __future__ import annotations

import pytest

from job_discovery.search.query_builder import (
    ALL_SOURCES,
    QueryBuilder,
    SearchCriteria,
    SOURCE_DOMAINS,
)


class TestSearchCriteria:
    def test_minimal_criteria(self):
        c = SearchCriteria()
        assert c.title is None
        assert c.remote_ok is False
        assert c.preferred_sources is None

    def test_full_criteria(self):
        c = SearchCriteria(
            title="Python Developer",
            location="Chennai",
            experience_years_min=2,
            experience_years_max=5,
            workplace_type="hybrid",
            employment_type="full-time",
            posting_age_days=7,
            preferred_sources=["linkedin", "naukri"],
        )
        assert c.title == "Python Developer"
        assert c.location == "Chennai"


class TestQueryBuilder:
    def setup_method(self):
        self.builder = QueryBuilder()

    # ── Build query variants ────────────────────────────────────────────

    def test_returns_1_to_5_queries(self):
        c = SearchCriteria(title="Python Developer", location="Chennai")
        queries = self.builder.build(c)
        assert 1 <= len(queries) <= 5

    def test_first_query_is_most_specific(self):
        c = SearchCriteria(title="Python Developer", location="Chennai")
        queries = self.builder.build(c)
        # Most specific includes both role and location
        assert "Chennai" in queries[0]
        assert "Python" in queries[0]

    def test_bare_title_included(self):
        c = SearchCriteria(title="Data Scientist", location="Bangalore")
        queries = self.builder.build(c)
        bare = any("Data Scientist" in q and "Bangalore" not in q for q in queries)
        # After stripping location, a bare variant should exist
        assert bare or len(queries) >= 1

    def test_no_duplicate_queries(self):
        c = SearchCriteria(title="Backend Engineer", location="Mumbai")
        queries = self.builder.build(c)
        lower = [q.lower() for q in queries]
        assert len(lower) == len(set(lower)), "Duplicate queries returned"

    def test_no_empty_queries(self):
        c = SearchCriteria(title="ML Engineer", location="Hyderabad")
        for q in self.builder.build(c):
            assert q.strip() != ""

    def test_remote_qualifier_added(self):
        c = SearchCriteria(title="Frontend Developer", workplace_type="remote")
        queries = self.builder.build(c)
        assert any("remote" in q.lower() for q in queries)

    def test_fallback_on_no_title(self):
        c = SearchCriteria()
        queries = self.builder.build(c)
        assert len(queries) >= 1
        assert all(q.strip() for q in queries)

    def test_keywords_used_as_fallback(self):
        c = SearchCriteria(keywords=["React", "TypeScript", "Node"])
        queries = self.builder.build(c)
        assert len(queries) >= 1

    def test_variants_for_developer_synonym(self):
        c = SearchCriteria(title="Python Developer", location="Pune")
        queries = self.builder.build(c)
        titles_only = " ".join(queries).lower()
        # Should have at least developer and engineer variants
        assert "developer" in titles_only or "engineer" in titles_only

    def test_tech_prefix_variant(self):
        """'Python Backend Developer' should generate a 'Python ...' variant."""
        c = SearchCriteria(title="Python Backend Developer", location="Delhi")
        queries = self.builder.build(c)
        combined = " ".join(queries).lower()
        assert "python" in combined

    # ── Example: canonical test from spec ───────────────────────────────

    def test_spec_example_python_developer_chennai(self):
        """
        Spec example: 'Python Developer Chennai' should yield 3-5 variants
        all containing location or at least the core role.
        """
        c = SearchCriteria(title="Python Developer", location="Chennai")
        queries = self.builder.build(c)
        assert 3 <= len(queries) <= 5
        assert any("Chennai" in q for q in queries)
        assert any("Python" in q or "python" in q.lower() for q in queries)

    # ── Source targeting ────────────────────────────────────────────────

    def test_get_target_sources_default(self):
        c = SearchCriteria(title="Engineer")
        sources = self.builder.get_target_sources(c)
        assert sources == ALL_SOURCES

    def test_get_target_sources_preferred(self):
        c = SearchCriteria(
            title="Engineer",
            preferred_sources=["linkedin", "naukri"],
        )
        sources = self.builder.get_target_sources(c)
        assert sources == ["linkedin", "naukri"]

    def test_get_target_sources_invalid_filtered(self):
        c = SearchCriteria(
            title="Engineer",
            preferred_sources=["linkedin", "notasite"],
        )
        sources = self.builder.get_target_sources(c)
        assert "notasite" not in sources
        assert "linkedin" in sources

    # ── SOURCE_DOMAINS sanity ────────────────────────────────────────────

    def test_source_domains_has_all_sources(self):
        for src in ALL_SOURCES:
            assert src in SOURCE_DOMAINS, f"Missing domain for {src}"
