"""
Tests for search run tracking, status transitions,
incremental URL persistence, and crash recovery.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

try:
    import mongomock
    _MONGOMOCK = True
except ImportError:
    _MONGOMOCK = False

from job_discovery.models.search import RunStatus, SearchRun, UrlProcessingStatus, UrlState
from job_discovery.orchestrator.run_tracker import (
    RunTracker,
    generate_run_id,
    reset_daily_counter,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def db():
    if not _MONGOMOCK:
        pytest.skip("mongomock not installed")
    client = mongomock.MongoClient()
    return client["test_search_run"]


@pytest.fixture
def repos(db):
    from job_discovery.repositories.search_repository import SearchRepository
    return SearchRepository(db)


# ─────────────────────────────────────────────────────────────────────────────
# Run ID generation
# ─────────────────────────────────────────────────────────────────────────────

class TestRunIdGeneration:
    def setup_method(self):
        reset_daily_counter()

    def test_format(self):
        rid = generate_run_id()
        assert rid.startswith("SR_")
        parts = rid.split("_")
        assert len(parts) == 3
        assert len(parts[1]) == 8   # YYYYMMDD
        assert parts[2].isdigit()

    def test_sequential(self):
        r1 = generate_run_id()
        r2 = generate_run_id()
        assert r1 != r2
        n1 = int(r1.split("_")[2])
        n2 = int(r2.split("_")[2])
        assert n2 == n1 + 1

    def test_date_in_id(self):
        rid = generate_run_id()
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        assert today in rid


# ─────────────────────────────────────────────────────────────────────────────
# RunTracker lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class TestRunTracker:
    def test_start_creates_run(self, repos):
        tracker = RunTracker(repos.runs)
        run = tracker.start()
        assert run.status == RunStatus.running
        assert run.started_at is not None

    def test_run_persisted_to_db(self, repos):
        tracker = RunTracker(repos.runs)
        tracker.start()
        doc = repos.runs.find_by_run_id(tracker.run_id)
        assert doc is not None
        assert doc["status"] == "running"

    def test_complete_transitions_status(self, repos):
        tracker = RunTracker(repos.runs)
        tracker.start()
        run = tracker.complete()
        assert run.status == RunStatus.completed
        assert run.completed_at is not None

    def test_complete_sets_duration(self, repos):
        tracker = RunTracker(repos.runs)
        tracker.start()
        run = tracker.complete()
        assert run.execution_duration_s is not None
        assert run.execution_duration_s >= 0

    def test_mark_failed(self, repos):
        tracker = RunTracker(repos.runs)
        tracker.start()
        run = tracker.mark_failed("test error")
        assert run.status == RunStatus.failed
        assert run.errors is not None
        assert "test error" in run.errors

    def test_mark_partial(self, repos):
        tracker = RunTracker(repos.runs)
        tracker.start()
        run = tracker.mark_partial("some source failed")
        assert run.status == RunStatus.partial

    def test_counter_increments(self, repos):
        tracker = RunTracker(repos.runs)
        tracker.start()
        tracker.inc_discovered(5)
        tracker.inc_fetched(3)
        tracker.inc_failed_fetch(1)
        assert tracker.state.urls_discovered == 5
        assert tracker.state.urls_fetched == 3
        assert tracker.state.failed_fetches == 1

    def test_query_tracking(self, repos):
        tracker = RunTracker(repos.runs)
        tracker.start()
        tracker.mark_query_issued("Python Developer Chennai")
        tracker.mark_query_issued("Python Engineer Chennai")
        assert "Python Developer Chennai" in tracker.state.queries_issued
        assert len(tracker.state.queries_issued) == 2

    def test_duplicate_query_not_added_twice(self, repos):
        tracker = RunTracker(repos.runs)
        tracker.start()
        tracker.mark_query_issued("same query")
        tracker.mark_query_issued("same query")
        assert tracker.state.queries_issued.count("same query") == 1


# ─────────────────────────────────────────────────────────────────────────────
# URL state / incremental persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestUrlStatePersistence:
    def test_upsert_and_retrieve(self, repos):
        state = UrlState(
            run_id="SR_20260808_001",
            url="https://naukri.com/job/123",
            status=UrlProcessingStatus.discovered,
        )
        repos.url_states.upsert(state)
        result = repos.url_states.get("SR_20260808_001", "https://naukri.com/job/123")
        assert result is not None
        assert result["status"] == "discovered"

    def test_update_status(self, repos):
        run_id = "SR_20260808_002"
        url = "https://indeed.com/viewjob?jk=abc"
        state = UrlState(run_id=run_id, url=url, status=UrlProcessingStatus.discovered)
        repos.url_states.upsert(state)
        repos.url_states.update_status(run_id, url, UrlProcessingStatus.fetched.value)
        result = repos.url_states.get(run_id, url)
        assert result["status"] == "fetched"

    def test_get_processed_urls(self, repos):
        run_id = "SR_20260808_003"
        states = [
            UrlState(run_id=run_id, url=f"https://a.com/{i}",
                     status=UrlProcessingStatus.stored)
            for i in range(3)
        ]
        for s in states:
            repos.url_states.upsert(s)
        # One not yet processed
        repos.url_states.upsert(UrlState(
            run_id=run_id, url="https://a.com/pending",
            status=UrlProcessingStatus.discovered
        ))
        processed = repos.url_states.get_processed_urls(run_id)
        assert len(processed) == 3
        assert "https://a.com/pending" not in processed

    def test_find_by_status(self, repos):
        run_id = "SR_20260808_004"
        repos.url_states.upsert(UrlState(run_id=run_id, url="https://a.com/1",
                                          status=UrlProcessingStatus.fetched))
        repos.url_states.upsert(UrlState(run_id=run_id, url="https://a.com/2",
                                          status=UrlProcessingStatus.discovered))
        fetched = repos.url_states.find_by_status(run_id, "fetched")
        assert len(fetched) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Crash recovery simulation
# ─────────────────────────────────────────────────────────────────────────────

class TestCrashRecovery:
    """
    Simulate: run starts, processes some URLs, crashes, resumes.
    After resume: already-processed URLs must be skipped.
    """

    def test_resume_skips_processed_urls(self, repos):
        run_id = "SR_CRASH_001"

        # ── Run 1: start and process 2 of 5 URLs ──────────────────────
        tracker = RunTracker(repos.runs, run_id=run_id)
        tracker.start()
        tracker.inc_discovered(5)

        urls = [f"https://naukri.com/job/{i}" for i in range(5)]
        for url in urls[:2]:
            repos.url_states.upsert(UrlState(
                run_id=run_id, url=url, status=UrlProcessingStatus.stored
            ))
            tracker.inc_fetched()

        # Simulate crash — status remains "running"
        # (no explicit complete/fail called)

        # ── Run 2: resume ─────────────────────────────────────────────
        processed = repos.url_states.get_processed_urls(run_id)
        assert len(processed) == 2    # 2 already done

        remaining = [u for u in urls if u not in processed]
        assert len(remaining) == 3    # 3 still to do

    def test_already_issued_queries_tracked(self, repos):
        run_id = "SR_CRASH_002"
        tracker = RunTracker(repos.runs, run_id=run_id)
        tracker.start()
        tracker.mark_query_issued("Python Developer Chennai")
        tracker.mark_query_issued("Python Engineer Chennai")

        # Simulate crash → on resume, read back issued queries
        doc = repos.runs.find_by_run_id(run_id)
        already_issued = set(doc.get("queries_issued") or [])
        assert "Python Developer Chennai" in already_issued
        assert "Python Engineer Chennai" in already_issued

    def test_query_cooldown_not_wrong_after_resume(self, repos):
        """
        After resume, already-issued queries should be in already_issued set,
        so they are skipped without re-triggering cooldown logic.
        """
        run_id = "SR_CRASH_003"
        tracker = RunTracker(repos.runs, run_id=run_id)
        tracker.start()
        tracker.mark_query_issued("Python Developer")

        # On resume: query already in issued list → skip it, no cooldown side-effect
        doc = repos.runs.find_by_run_id(run_id)
        already = set(doc.get("queries_issued") or [])
        assert "Python Developer" in already  # → orchestrator will skip it

    def test_completed_run_not_resumed(self, repos):
        """A completed run is not treated as interrupted."""
        run_id = "SR_CRASH_004"
        tracker = RunTracker(repos.runs, run_id=run_id)
        tracker.start()
        tracker.complete()

        doc = repos.runs.find_by_run_id(run_id)
        assert doc["status"] == "completed"
        # Orchestrator checks: only "running"/"partial" → resume
        is_resumable = doc.get("status") in ("running", "partial")
        assert not is_resumable

    def test_no_duplicate_processing_after_resume(self, repos):
        """processed_urls set prevents re-processing."""
        run_id = "SR_CRASH_005"
        urls = [f"https://linkedin.com/jobs/view/{i}" for i in range(10)]

        # Mark first 7 as stored
        for url in urls[:7]:
            repos.url_states.upsert(UrlState(
                run_id=run_id, url=url, status=UrlProcessingStatus.stored
            ))

        processed = repos.url_states.get_processed_urls(run_id)
        new_urls = [u for u in urls if u not in processed]
        assert len(new_urls) == 3
        assert all(u in urls[7:] for u in new_urls)
