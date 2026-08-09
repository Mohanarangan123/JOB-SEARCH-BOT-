"""
Request / response Pydantic schemas for the API layer.
These are separate from the internal domain models to allow independent evolution.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Request schemas
# ─────────────────────────────────────────────────────────────────────────────

class JobSearchRequest(BaseModel):
    """POST /api/jobs/search"""
    title:                Optional[str]        = None
    keywords:             Optional[List[str]]  = None
    location:             Optional[str]        = None
    remote_ok:            bool                 = False
    workplace_type:       Optional[str]        = None
    employment_type:      Optional[str]        = None
    experience_years_min: Optional[int]        = None
    experience_years_max: Optional[int]        = None
    posting_age_days:     Optional[int]        = None
    preferred_sources:    Optional[List[str]]  = None


# ─────────────────────────────────────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class ComponentScoresResponse(BaseModel):
    title:      float
    location:   float
    experience: float
    workplace:  float
    freshness:  float


class RelevanceScoreResponse(BaseModel):
    total:      float
    components: ComponentScoresResponse


class ValidationStatusResponse(BaseModel):
    status:         str
    score:          Optional[float]
    missing_fields: Optional[List[str]]
    warnings:       Optional[List[str]]


class FreshnessResponse(BaseModel):
    first_seen_at:    Optional[datetime]
    last_seen_at:     Optional[datetime]
    last_verified_at: Optional[datetime]


class JobSummaryResponse(BaseModel):
    """Returned in list endpoints — no raw HTML."""
    job_id:           str               # canonical_url as the public ID
    canonical_url:    Optional[str]
    title:            Optional[str]
    company:          Optional[str]
    location:         Optional[str]
    employment_type:  Optional[str]
    work_mode:        Optional[str]
    source_name:      Optional[str]
    lifecycle_status: Optional[str]
    version_count:    Optional[int]
    freshness:        Optional[FreshnessResponse]
    validation:       Optional[ValidationStatusResponse]
    relevance_score:  Optional[RelevanceScoreResponse] = None


class JobDetailResponse(JobSummaryResponse):
    """Returned by GET /api/jobs/{job_id} — includes structured details."""
    description_summary:  Optional[str]
    required_skills:      Optional[List[str]]
    preferred_skills:     Optional[List[str]]
    salary_raw:           Optional[str]
    experience_raw:       Optional[str]
    apply_url:            Optional[str]
    apply_email:          Optional[str]
    seniority:            Optional[str]
    content_hash:         Optional[str]
    # raw_content_path intentionally excluded from API responses


class JobVersionResponse(BaseModel):
    version_number:   int
    content_hash:     str
    retrieved_at:     Optional[datetime]
    changed_fields:   Optional[List[str]]


class ChangeResponse(BaseModel):
    version_number: int
    field_name:     str
    before:         Any
    after:          Any


class SearchRunResponse(BaseModel):
    run_id:               str
    status:               str
    started_at:           Optional[datetime]
    completed_at:         Optional[datetime]
    queries_generated:    int
    sources_searched:     int
    urls_discovered:      int
    urls_fetched:         int
    failed_fetches:       int
    extraction_failures:  int
    duplicates_skipped:   int
    unique_jobs:          int
    execution_duration_s: Optional[float]
    errors:               Optional[List[str]]


class SourceHealthResponse(BaseModel):
    source_name:         str
    total_jobs:          int
    successful_extractions: int
    failed_extractions:  int
    extraction_rate:     float
    field_rates:         Dict[str, float]
    last_updated:        Optional[datetime]


class SearchRunCreatedResponse(BaseModel):
    run_id:  str
    status:  str
    message: str


class ExportResponse(BaseModel):
    message:   str
    run_id:    Optional[str] = None
    job_count: int
