"""Tests for source Pydantic models (SourceFetch, ExtractionEvent, QueryCache, SourceHealth)."""
from __future__ import annotations

from datetime import datetime

import pytest

from job_discovery.models.source import (
    CircuitState,
    ExtractionEvent,
    ExtractionStatus,
    FetchStatus,
    QueryCache,
    SourceFetch,
    SourceHealth,
)


class TestSourceFetch:
    def test_minimal(self):
        f = SourceFetch(
            run_id="r1",
            source_name="linkedin",
            url="https://linkedin.com/jobs/1",
        )
        assert f.status == FetchStatus.failed
        assert f.http_status_code is None

    def test_success(self):
        f = SourceFetch(
            run_id="r1",
            source_name="indeed",
            url="https://indeed.com/job/99",
            status=FetchStatus.success,
            http_status_code=200,
            response_size_bytes=12_000,
            content_hash="abc",
        )
        assert f.status == FetchStatus.success
        assert f.content_hash == "abc"

    def test_null_fields(self):
        f = SourceFetch(run_id="r1", source_name="s", url="https://x.com")
        assert f.raw_content_path is None
        assert f.error_message is None


class TestExtractionEvent:
    def test_minimal(self):
        e = ExtractionEvent(
            run_id="r1",
            source_name="naukri",
            url="https://naukri.com/job/1",
        )
        assert e.status == ExtractionStatus.failed
        assert e.llm_model is None

    def test_full(self):
        e = ExtractionEvent(
            run_id="r2",
            source_name="linkedin",
            url="https://linkedin.com/jobs/1",
            canonical_url="https://linkedin.com/jobs/1",
            status=ExtractionStatus.success,
            llm_model="gpt-4o-mini",
            prompt_tokens=800,
            completion_tokens=350,
            fields_extracted=["title", "company", "salary"],
        )
        assert e.fields_extracted == ["title", "company", "salary"]
        assert e.prompt_tokens == 800


class TestQueryCache:
    def test_minimal(self):
        q = QueryCache(
            query_hash="h1",
            query_text="python developer",
            source_name="linkedin",
        )
        assert q.result_count == 0
        assert q.hit_count == 0

    def test_with_results(self):
        q = QueryCache(
            query_hash="h2",
            query_text="data scientist",
            source_name="naukri",
            results=[{"url": "https://naukri.com/1"}],
            result_count=1,
        )
        assert len(q.results) == 1


class TestSourceHealth:
    def test_defaults(self):
        h = SourceHealth(source_name="linkedin")
        assert h.circuit_state == CircuitState.closed
        assert h.consecutive_failures == 0
        assert h.success_rate == 1.0

    def test_open_circuit(self):
        h = SourceHealth(
            source_name="indeed",
            circuit_state=CircuitState.open,
            consecutive_failures=5,
            success_rate=0.2,
        )
        assert h.circuit_state == CircuitState.open
        assert h.consecutive_failures == 5

    def test_null_timing(self):
        h = SourceHealth(source_name="wellfound")
        assert h.last_success_at is None
        assert h.last_failure_at is None
