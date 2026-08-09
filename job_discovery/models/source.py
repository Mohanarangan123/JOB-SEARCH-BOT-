"""
Pydantic models for source-level tracking:
  - SourceFetch      : one HTTP fetch attempt
  - ExtractionEvent  : one LLM/parser extraction attempt
  - QueryCache       : cached search query results
  - SourceHealth     : per-source reliability snapshot
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class FetchStatus(str, Enum):
    success = "success"
    failed = "failed"
    skipped = "skipped"
    blocked = "blocked"


class ExtractionStatus(str, Enum):
    success = "success"
    partial = "partial"
    failed = "failed"


class CircuitState(str, Enum):
    closed = "closed"       # normal operation
    open = "open"           # blocking requests
    half_open = "half_open" # probe allowed


# ─────────────────────────────────────────────────────────────────────────────

class SourceFetch(BaseModel):
    """
    Records a single HTTP page-fetch attempt.
    Stored in the `source_fetches` collection.
    """
    run_id: str
    source_name: str
    url: str
    status: FetchStatus = FetchStatus.failed
    http_status_code: Optional[int] = None
    response_size_bytes: Optional[int] = None
    duration_ms: Optional[float] = None
    error_message: Optional[str] = None
    content_hash: Optional[str] = None
    raw_content_path: Optional[str] = None
    fetched_at: datetime = Field(default_factory=_utcnow)

    model_config = {"populate_by_name": True}


class ExtractionEvent(BaseModel):
    """
    Records a single LLM / parser extraction attempt.
    Stored in the `extraction_events` collection.
    """
    run_id: str
    source_name: str
    url: str
    canonical_url: Optional[str] = None
    status: ExtractionStatus = ExtractionStatus.failed
    llm_model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    duration_ms: Optional[float] = None
    error_message: Optional[str] = None
    fields_extracted: Optional[List[str]] = None
    extracted_at: datetime = Field(default_factory=_utcnow)

    model_config = {"populate_by_name": True}


class QueryCache(BaseModel):
    """
    Caches a search query and its results to avoid redundant requests.
    Stored in the `query_cache` collection.
    """
    query_hash: str             # SHA-256 of normalised query string
    query_text: str
    source_name: str
    results: Optional[List[Dict[str, Any]]] = None
    result_count: int = 0
    cached_at: datetime = Field(default_factory=_utcnow)
    expires_at: Optional[datetime] = None
    hit_count: int = 0

    model_config = {"populate_by_name": True}


class SourceHealth(BaseModel):
    """
    Snapshot of a source's operational health.
    Stored in the `source_health` collection (upserted per source_name).
    """
    source_name: str
    circuit_state: CircuitState = CircuitState.closed
    consecutive_failures: int = 0
    total_requests: int = 0
    total_failures: int = 0
    success_rate: float = 1.0       # 0.0 – 1.0
    avg_response_ms: Optional[float] = None
    last_success_at: Optional[datetime] = None
    last_failure_at: Optional[datetime] = None
    last_error: Optional[str] = None
    updated_at: datetime = Field(default_factory=_utcnow)

    model_config = {"populate_by_name": True}
