"""
Tests for QueryExpander — LLM-assisted query variant generation.
"""
from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest

from job_discovery.search.query_expander import LLMClient, QueryExpander


class TestQueryExpanderNoLLM:
    """Without an LLM, expander returns base queries unchanged."""

    def test_returns_base_queries_when_no_llm(self):
        expander = QueryExpander()
        bases = ["Python Developer Chennai", "Python Engineer Chennai"]
        result = expander.expand(bases, max_total=5)
        assert result[:2] == bases

    def test_empty_input(self):
        expander = QueryExpander()
        assert expander.expand([]) == []

    def test_single_base_query(self):
        expander = QueryExpander()
        bases = ["Data Scientist Bangalore"]
        result = expander.expand(bases)
        assert "Data Scientist Bangalore" in result

    def test_respects_max_total(self):
        expander = QueryExpander()
        bases = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
        result = expander.expand(bases, max_total=4)
        assert len(result) == 4

    def test_base_queries_first(self):
        expander = QueryExpander()
        bases = ["A", "B", "C"]
        result = expander.expand(bases, max_total=5)
        assert result[:3] == bases


class TestQueryExpanderWithMockLLM:
    """With a mock LLM, expander appends valid variants."""

    def _make_llm(self, response: str) -> MagicMock:
        mock = MagicMock(spec=LLMClient)
        mock.complete.return_value = response
        return mock

    def test_appends_llm_variants(self):
        llm = self._make_llm(
            '["Python Engineer Bangalore", "Backend Developer Python Bangalore"]'
        )
        expander = QueryExpander(llm_client=llm)
        bases = ["Python Developer Bangalore"]
        result = expander.expand(bases, max_total=5)
        assert "Python Developer Bangalore" in result
        assert "Python Engineer Bangalore" in result
        assert len(result) <= 5

    def test_no_duplicates(self):
        llm = self._make_llm('["Python Developer Bangalore"]')
        expander = QueryExpander(llm_client=llm)
        bases = ["Python Developer Bangalore"]
        result = expander.expand(bases, max_total=5)
        lower = [r.lower() for r in result]
        assert len(lower) == len(set(lower))

    def test_llm_called_with_first_query(self):
        llm = self._make_llm('["Variant A"]')
        expander = QueryExpander(llm_client=llm)
        bases = ["Base Query Chennai"]
        expander.expand(bases, max_total=5)
        call_args = llm.complete.call_args[0][0]
        assert "Base Query Chennai" in call_args

    def test_llm_failure_falls_back_gracefully(self):
        llm = MagicMock(spec=LLMClient)
        llm.complete.side_effect = RuntimeError("LLM unavailable")
        expander = QueryExpander(llm_client=llm)
        bases = ["Python Developer Mumbai"]
        result = expander.expand(bases, max_total=5)
        # Must still return base queries
        assert "Python Developer Mumbai" in result

    def test_llm_invalid_json_falls_back(self):
        llm = self._make_llm("not json at all")
        expander = QueryExpander(llm_client=llm)
        result = expander.expand(["Base Query"], max_total=5)
        assert "Base Query" in result

    def test_llm_non_array_response_falls_back(self):
        llm = self._make_llm('{"key": "value"}')
        expander = QueryExpander(llm_client=llm)
        result = expander.expand(["Base Query"], max_total=5)
        assert "Base Query" in result

    def test_max_llm_variants_capped(self):
        llm = self._make_llm('["V1","V2","V3","V4","V5","V6","V7"]')
        expander = QueryExpander(llm_client=llm, max_llm_variants=3)
        result = expander.expand(["Base"], max_total=10)
        # Base + up to 3 LLM variants = 4
        assert len(result) <= 4

    def test_llm_skipped_when_max_already_reached(self):
        llm = MagicMock(spec=LLMClient)
        expander = QueryExpander(llm_client=llm)
        bases = ["A", "B", "C", "D", "E"]
        result = expander.expand(bases, max_total=5)
        # Already at max — LLM should not be called
        llm.complete.assert_not_called()
        assert result == bases


class TestQueryHashing:
    def test_hash_deterministic(self):
        from job_discovery.search.providers.base import ProviderSearchResult
        h1 = ProviderSearchResult.make_query_hash("Python Developer Chennai")
        h2 = ProviderSearchResult.make_query_hash("Python Developer Chennai")
        assert h1 == h2

    def test_hash_case_insensitive(self):
        from job_discovery.search.providers.base import ProviderSearchResult
        h1 = ProviderSearchResult.make_query_hash("Python Developer Chennai")
        h2 = ProviderSearchResult.make_query_hash("python developer chennai")
        assert h1 == h2

    def test_hash_whitespace_normalised(self):
        from job_discovery.search.providers.base import ProviderSearchResult
        h1 = ProviderSearchResult.make_query_hash("Python  Developer   Chennai")
        h2 = ProviderSearchResult.make_query_hash("Python Developer Chennai")
        assert h1 == h2

    def test_different_queries_produce_different_hashes(self):
        from job_discovery.search.providers.base import ProviderSearchResult
        h1 = ProviderSearchResult.make_query_hash("Python Developer Chennai")
        h2 = ProviderSearchResult.make_query_hash("Java Developer Bangalore")
        assert h1 != h2

    def test_hash_is_64_hex_chars(self):
        from job_discovery.search.providers.base import ProviderSearchResult
        h = ProviderSearchResult.make_query_hash("test query")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
