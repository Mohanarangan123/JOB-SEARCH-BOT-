"""
RunTracker — generates run IDs and manages SearchRun lifecycle.

Run ID format: SR_YYYYMMDD_NNN (e.g. SR_20260808_001)
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Optional

from job_discovery.models.search import RunStatus, SearchRun

_counter_lock = threading.Lock()
_daily_counter: dict = {}   # date_str → int


def generate_run_id() -> str:
    """Generate a unique run ID in format SR_YYYYMMDD_NNN."""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    with _counter_lock:
        _daily_counter[date_str] = _daily_counter.get(date_str, 0) + 1
        n = _daily_counter[date_str]
    return f"SR_{date_str}_{n:03d}"


def reset_daily_counter():
    """Reset counter — for tests only."""
    with _counter_lock:
        _daily_counter.clear()


class RunTracker:
    """
    Manages the lifecycle of a single SearchRun.
    Persists state changes incrementally via SearchRunRepository.
    """

    def __init__(self, repo, run_id: Optional[str] = None) -> None:
        self._repo = repo
        self.run_id = run_id or generate_run_id()
        self._run = SearchRun(run_id=self.run_id)
        self._start_time: Optional[datetime] = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self, criteria_dict: Optional[dict] = None) -> SearchRun:
        now = datetime.now(timezone.utc)
        self._start_time = now
        self._run.status = RunStatus.running
        self._run.started_at = now
        self._run.criteria = criteria_dict
        self._repo.insert(self._run)
        return self._run

    def complete(self) -> SearchRun:
        now = datetime.now(timezone.utc)
        self._run.status = RunStatus.completed
        self._run.completed_at = now
        if self._start_time:
            self._run.execution_duration_s = (now - self._start_time).total_seconds()
        self._flush()
        return self._run

    def mark_partial(self, reason: str = "") -> SearchRun:
        self._run.status = RunStatus.partial
        self._add_error(reason)
        self._flush()
        return self._run

    def mark_failed(self, reason: str = "") -> SearchRun:
        self._run.status = RunStatus.failed
        self._add_error(reason)
        self._run.completed_at = datetime.now(timezone.utc)
        self._flush()
        return self._run

    # ── Increment counters ───────────────────────────────────────────────

    def inc_queries(self, n: int = 1) -> None:
        self._run.queries_generated += n
        self._flush_counters()

    def inc_sources(self, n: int = 1) -> None:
        self._run.sources_searched += n
        self._flush_counters()

    def inc_discovered(self, n: int = 1) -> None:
        self._run.urls_discovered += n
        self._flush_counters()

    def inc_fetched(self, n: int = 1) -> None:
        self._run.urls_fetched += n
        self._flush_counters()

    def inc_failed_fetch(self, n: int = 1) -> None:
        self._run.failed_fetches += n
        self._flush_counters()

    def inc_extraction_failure(self, n: int = 1) -> None:
        self._run.extraction_failures += n
        self._flush_counters()

    def inc_duplicate(self, n: int = 1) -> None:
        self._run.duplicates_skipped += n
        self._flush_counters()

    def inc_unique_jobs(self, n: int = 1) -> None:
        self._run.unique_jobs += n
        self._flush_counters()

    def mark_query_issued(self, query: str) -> None:
        if self._run.queries_issued is None:
            self._run.queries_issued = []
        if query not in self._run.queries_issued:
            self._run.queries_issued.append(query)
        self._flush_counters()

    @property
    def state(self) -> SearchRun:
        return self._run

    # ── Internal ─────────────────────────────────────────────────────────

    def _add_error(self, msg: str) -> None:
        if msg:
            if self._run.errors is None:
                self._run.errors = []
            self._run.errors.append(msg)

    def _flush(self) -> None:
        """Write all current fields to DB."""
        data = self._run.model_dump(exclude_none=True)
        self._repo.update_status(
            self.run_id, self._run.status.value, **{
                k: v for k, v in data.items()
                if k not in ("run_id", "status", "created_at")
            }
        )

    def _flush_counters(self) -> None:
        """Write counter fields only (lightweight incremental update)."""
        self._repo.update_status(
            self.run_id,
            self._run.status.value,
            queries_generated=self._run.queries_generated,
            sources_searched=self._run.sources_searched,
            urls_discovered=self._run.urls_discovered,
            urls_fetched=self._run.urls_fetched,
            failed_fetches=self._run.failed_fetches,
            extraction_failures=self._run.extraction_failures,
            duplicates_skipped=self._run.duplicates_skipped,
            unique_jobs=self._run.unique_jobs,
            queries_issued=self._run.queries_issued or [],
        )
