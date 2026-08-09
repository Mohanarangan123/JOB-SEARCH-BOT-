"""
Pydantic models for search runs, URL processing state, and results.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RunStatus(str, Enum):
    pending   = "pending"
    running   = "running"
    completed = "completed"
    failed    = "failed"
    partial   = "partial"


class UrlProcessingStatus(str, Enum):
    """Tracks per-URL progress within a run — used for crash recovery."""
    discovered       = "discovered"
    fetch_queued     = "fetch_queued"
    fetched          = "fetched"
    fetch_failed     = "fetch_failed"
    fetch_skipped    = "fetch_skipped"
    extraction_done  = "extraction_done"
    extraction_failed= "extraction_failed"
    stored           = "stored"
    duplicate        = "duplicate"


class SearchRun(BaseModel):
    """
    Represents a single end-to-end orchestrator execution.
    run_id format: SR_YYYYMMDD_NNN
    Stored in the `search_runs` collection.
    """
    run_id: str
    status: RunStatus = RunStatus.pending
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Counters
    queries_generated:    int = 0
    sources_searched:     int = 0
    urls_discovered:      int = 0
    urls_fetched:         int = 0
    failed_fetches:       int = 0
    extraction_failures:  int = 0
    duplicates_skipped:   int = 0
    unique_jobs:          int = 0
    jobs_updated:         int = 0
    execution_duration_s: Optional[float] = None

    # Recovery support
    criteria:             Optional[Dict[str, Any]] = None   # serialised SearchCriteria
    queries_issued:       Optional[List[str]] = None        # queries already executed

    errors: Optional[List[str]] = None

    created_at: datetime = Field(default_factory=_utcnow)

    model_config = {"populate_by_name": True}


class UrlState(BaseModel):
    """
    Per-URL processing state within a run.
    Stored in the `url_states` collection — drives crash recovery.
    """
    run_id:           str
    url:              str
    canonical_url:    Optional[str] = None
    source_name:      Optional[str] = None
    status:           UrlProcessingStatus = UrlProcessingStatus.discovered
    content_hash:     Optional[str] = None
    error_message:    Optional[str] = None
    discovered_at:    datetime = Field(default_factory=_utcnow)
    updated_at:       datetime = Field(default_factory=_utcnow)

    model_config = {"populate_by_name": True}


class SearchResult(BaseModel):
    """
    A single URL / snippet returned by a search provider.
    Stored in the `search_results` collection.
    """
    run_id:        str
    query:         str
    source_name:   str
    url:           str
    title:         Optional[str] = None
    snippet:       Optional[str] = None
    rank:          Optional[int] = None
    fetched:       bool = False
    fetch_skipped: bool = False
    discovered_at: datetime = Field(default_factory=_utcnow)

    model_config = {"populate_by_name": True}
