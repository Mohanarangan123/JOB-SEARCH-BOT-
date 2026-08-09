"""Tests for search Pydantic models."""
from __future__ import annotations

from datetime import datetime

import pytest

from job_discovery.models.search import (
    RunStatus,
    SearchResult,
    SearchRun,
    UrlProcessingStatus,
    UrlState,
)


class TestSearchRun:
    def test_minimal(self):
        run = SearchRun(run_id="run-001")
        assert run.status == RunStatus.pending
        assert run.queries_generated == 0
        assert run.urls_discovered == 0
        assert run.errors is None

    def test_full(self):
        run = SearchRun(
            run_id="run-002",
            status=RunStatus.completed,
            started_at=datetime(2026, 1, 1),
            completed_at=datetime(2026, 1, 1, 0, 5),
            queries_generated=4,
            sources_searched=8,
            urls_discovered=45,
            urls_fetched=40,
            failed_fetches=2,
            unique_jobs=38,
        )
        assert run.urls_discovered == 45
        assert run.status == RunStatus.completed

    def test_dict_round_trip(self):
        run = SearchRun(run_id="run-003", status=RunStatus.running)
        restored = SearchRun(**run.model_dump())
        assert restored.run_id == "run-003"

    def test_null_fields(self):
        run = SearchRun(run_id="run-004")
        assert run.started_at is None
        assert run.completed_at is None
        assert run.criteria is None

    def test_queries_issued_list(self):
        run = SearchRun(run_id="run-005")
        assert run.queries_issued is None  # None by default (list populated at runtime)


class TestUrlState:
    def test_minimal(self):
        s = UrlState(run_id="SR_001", url="https://a.com/1")
        assert s.status == UrlProcessingStatus.discovered
        assert s.canonical_url is None

    def test_status_enum(self):
        for status in UrlProcessingStatus:
            s = UrlState(run_id="SR_001", url="https://a.com/1", status=status)
            assert s.status == status


class TestSearchResult:
    def test_minimal(self):
        r = SearchResult(
            run_id="run-001",
            query="python developer india",
            source_name="linkedin",
            url="https://linkedin.com/jobs/1",
        )
        assert r.fetched is False
        assert r.title is None

    def test_full(self):
        r = SearchResult(
            run_id="run-001",
            query="data engineer",
            source_name="naukri",
            url="https://naukri.com/job/42",
            title="Data Engineer",
            snippet="Exciting role...",
            rank=3,
            fetched=True,
        )
        assert r.rank == 3
        assert r.fetched is True
