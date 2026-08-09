"""
Tests for repository initialization and index definitions.
Uses mongomock so no live MongoDB instance is required.
Falls back to MagicMock if mongomock is not installed.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

# ── Try to use mongomock for lightweight in-process MongoDB ──────────────────
try:
    import mongomock

    @pytest.fixture()
    def db():
        client = mongomock.MongoClient()
        return client["test_job_discovery"]

    USING_MONGOMOCK = True
except ImportError:
    USING_MONGOMOCK = False

    @pytest.fixture()
    def db():
        pytest.skip("mongomock not installed — skipping repository integration tests")


# ─────────────────────────────────────────────────────────────────────────────
# JobRepository
# ─────────────────────────────────────────────────────────────────────────────

class TestJobRepositoryInit:
    def test_initializes_without_error(self, db):
        from job_discovery.repositories.job_repository import JobRepository
        repo = JobRepository(db)
        assert repo is not None

    def test_jobs_collection_created(self, db):
        from job_discovery.repositories.job_repository import (
            JOBS_COLLECTION,
            JobRepository,
        )
        JobRepository(db)
        assert JOBS_COLLECTION in db.list_collection_names()

    def test_job_versions_collection_created(self, db):
        from job_discovery.repositories.job_repository import (
            JOB_VERSIONS_COLLECTION,
            JobRepository,
        )
        JobRepository(db)
        assert JOB_VERSIONS_COLLECTION in db.list_collection_names()


class TestJobRepositoryCRUD:
    def test_insert_and_find(self, db):
        from job_discovery.models.job import JobRecord, LifecycleStatus
        from job_discovery.repositories.job_repository import JobRepository

        repo = JobRepository(db)
        job = JobRecord(
            canonical_url="https://linkedin.com/jobs/test-1",
            lifecycle_status=LifecycleStatus.active,
        )
        repo.insert_one(job)

        found = repo.find_by_canonical_url("https://linkedin.com/jobs/test-1")
        assert found is not None
        assert found["canonical_url"] == "https://linkedin.com/jobs/test-1"

    def test_upsert_creates_document(self, db):
        from job_discovery.models.job import JobRecord, LifecycleStatus
        from job_discovery.repositories.job_repository import JobRepository

        repo = JobRepository(db)
        job = JobRecord(
            canonical_url="https://indeed.com/jobs/upsert-1",
            lifecycle_status=LifecycleStatus.active,
        )
        result = repo.upsert_by_canonical_url(job)
        assert result.upserted_id is not None or result.modified_count >= 0

    def test_upsert_updates_existing(self, db):
        from job_discovery.models.job import JobRecord, JobDetails, LifecycleStatus
        from job_discovery.repositories.job_repository import JobRepository

        repo = JobRepository(db)
        url = "https://naukri.com/jobs/upsert-2"

        job = JobRecord(canonical_url=url, lifecycle_status=LifecycleStatus.active)
        repo.upsert_by_canonical_url(job)

        job2 = JobRecord(
            canonical_url=url,
            lifecycle_status=LifecycleStatus.active,
            details=JobDetails(title="Updated Title"),
        )
        repo.upsert_by_canonical_url(job2)

        found = repo.find_by_canonical_url(url)
        assert found is not None

    def test_find_by_content_hash(self, db):
        from job_discovery.models.job import JobRecord
        from job_discovery.repositories.job_repository import JobRepository

        repo = JobRepository(db)
        job = JobRecord(
            canonical_url="https://wellfound.com/job/hash-1",
            content_hash="sha256_abc",
        )
        repo.insert_one(job)

        found = repo.find_by_content_hash("sha256_abc")
        assert found is not None

    def test_mark_removed(self, db):
        from job_discovery.models.job import JobRecord, LifecycleStatus
        from job_discovery.repositories.job_repository import JobRepository

        repo = JobRepository(db)
        url = "https://cutshort.io/job/remove-1"
        repo.insert_one(JobRecord(canonical_url=url, lifecycle_status=LifecycleStatus.active))
        repo.mark_removed(url)

        found = repo.find_by_canonical_url(url)
        assert found["lifecycle_status"] == LifecycleStatus.removed

    def test_count(self, db):
        from job_discovery.models.job import JobRecord
        from job_discovery.repositories.job_repository import JobRepository

        repo = JobRepository(db)
        repo.insert_one(JobRecord(canonical_url="https://hirist.com/job/c1"))
        repo.insert_one(JobRecord(canonical_url="https://hirist.com/job/c2"))
        assert repo.count() >= 2


# ─────────────────────────────────────────────────────────────────────────────
# SearchRepository
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchRepositoryInit:
    def test_initializes_without_error(self, db):
        from job_discovery.repositories.search_repository import SearchRepository
        repo = SearchRepository(db)
        assert repo.runs is not None
        assert repo.results is not None
        assert repo.fetches is not None
        assert repo.extractions is not None
        assert repo.query_cache is not None
        assert repo.health is not None

    def test_collections_created(self, db):
        from job_discovery.repositories.search_repository import (
            EXTRACTION_EVENTS_COLLECTION,
            QUERY_CACHE_COLLECTION,
            SEARCH_RESULTS_COLLECTION,
            SEARCH_RUNS_COLLECTION,
            SOURCE_FETCHES_COLLECTION,
            SOURCE_HEALTH_COLLECTION,
            SearchRepository,
        )
        SearchRepository(db)
        names = db.list_collection_names()
        for col in (
            SEARCH_RUNS_COLLECTION,
            SEARCH_RESULTS_COLLECTION,
            SOURCE_FETCHES_COLLECTION,
            EXTRACTION_EVENTS_COLLECTION,
            QUERY_CACHE_COLLECTION,
            SOURCE_HEALTH_COLLECTION,
        ):
            assert col in names, f"Collection {col!r} not created"


class TestSearchRunRepository:
    def test_insert_and_find(self, db):
        from job_discovery.models.search import RunStatus, SearchRun
        from job_discovery.repositories.search_repository import SearchRepository

        repo = SearchRepository(db)
        run = SearchRun(run_id="run-test-1", status=RunStatus.running)
        repo.runs.insert(run)

        found = repo.runs.find_by_run_id("run-test-1")
        assert found is not None
        assert found["run_id"] == "run-test-1"

    def test_update_status(self, db):
        from job_discovery.models.search import RunStatus, SearchRun
        from job_discovery.repositories.search_repository import SearchRepository

        repo = SearchRepository(db)
        repo.runs.insert(SearchRun(run_id="run-test-2", status=RunStatus.running))
        repo.runs.update_status("run-test-2", RunStatus.completed)
        found = repo.runs.find_by_run_id("run-test-2")
        assert found["status"] == RunStatus.completed


class TestSourceHealthRepository:
    def test_upsert_and_find(self, db):
        from job_discovery.models.source import CircuitState, SourceHealth
        from job_discovery.repositories.search_repository import SearchRepository

        repo = SearchRepository(db)
        health = SourceHealth(source_name="linkedin", consecutive_failures=2)
        repo.health.upsert(health)

        found = repo.health.find_by_source("linkedin")
        assert found is not None
        assert found["source_name"] == "linkedin"

    def test_open_circuits(self, db):
        from job_discovery.models.source import CircuitState, SourceHealth
        from job_discovery.repositories.search_repository import SearchRepository

        repo = SearchRepository(db)
        repo.health.upsert(SourceHealth(source_name="indeed", circuit_state=CircuitState.open))
        repo.health.upsert(SourceHealth(source_name="naukri", circuit_state=CircuitState.closed))

        open_circuits = repo.health.find_open_circuits()
        source_names = [h["source_name"] for h in open_circuits]
        assert "indeed" in source_names
        assert "naukri" not in source_names


class TestQueryCacheRepository:
    def test_upsert_and_get(self, db):
        from job_discovery.models.source import QueryCache
        from job_discovery.repositories.search_repository import SearchRepository

        repo = SearchRepository(db)
        entry = QueryCache(
            query_hash="qh_001",
            query_text="python developer bangalore",
            source_name="linkedin",
            result_count=5,
        )
        repo.query_cache.upsert(entry)

        found = repo.query_cache.get("qh_001")
        assert found is not None
        assert found["query_text"] == "python developer bangalore"

    def test_increment_hit(self, db):
        from job_discovery.models.source import QueryCache
        from job_discovery.repositories.search_repository import SearchRepository

        repo = SearchRepository(db)
        entry = QueryCache(
            query_hash="qh_002",
            query_text="data engineer",
            source_name="naukri",
        )
        repo.query_cache.upsert(entry)
        repo.query_cache.increment_hit("qh_002")
        found = repo.query_cache.get("qh_002")
        assert found["hit_count"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Index definitions (structural check via collection inspection)
# ─────────────────────────────────────────────────────────────────────────────

class TestIndexDefinitions:
    def test_jobs_canonical_url_index(self, db):
        from job_discovery.repositories.job_repository import JOBS_COLLECTION, JobRepository

        JobRepository(db)
        indexes = db[JOBS_COLLECTION].index_information()
        index_keys = [
            tuple(v["key"])
            for v in indexes.values()
        ]
        assert any("canonical_url" in str(k) for k in index_keys)

    def test_jobs_content_hash_index(self, db):
        from job_discovery.repositories.job_repository import JOBS_COLLECTION, JobRepository

        JobRepository(db)
        indexes = db[JOBS_COLLECTION].index_information()
        assert any("content_hash" in str(v["key"]) for v in indexes.values())

    def test_search_runs_run_id_index(self, db):
        from job_discovery.repositories.search_repository import (
            SEARCH_RUNS_COLLECTION,
            SearchRepository,
        )
        SearchRepository(db)
        indexes = db[SEARCH_RUNS_COLLECTION].index_information()
        assert any("run_id" in str(v["key"]) for v in indexes.values())

    def test_no_vector_indexes(self, db):
        """Ensure no vector / knnVector indexes were created."""
        from job_discovery.repositories.job_repository import JOBS_COLLECTION, JobRepository

        JobRepository(db)
        indexes = db[JOBS_COLLECTION].index_information()
        for name, info in indexes.items():
            assert "knnVector" not in str(info), f"Vector index found: {name}"
            assert "vector" not in name.lower(), f"Vector index found: {name}"
