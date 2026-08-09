"""
Pydantic models for job records and related sub-documents.
No semantic-similarity or embedding fields — those are V2.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

from pydantic import BaseModel, Field, HttpUrl


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class LifecycleStatus(str, Enum):
    """Tracks the current state of a job listing in the system."""
    active = "active"
    expired = "expired"
    removed = "removed"
    unknown = "unknown"


class ValidationStatus(str, Enum):
    """Result of the validation pipeline for a given record."""
    valid = "valid"
    partial = "partial"
    invalid = "invalid"
    pending = "pending"


# ─────────────────────────────────────────────────────────────────────────────
# Sub-documents
# ─────────────────────────────────────────────────────────────────────────────

class JobSource(BaseModel):
    """Origin information for a job listing."""
    source_name: str
    source_url: Optional[str] = None
    external_job_id: Optional[str] = None
    canonical_url: Optional[str] = None
    scraped_at: Optional[datetime] = None

    model_config = {"populate_by_name": True}


class Company(BaseModel):
    """Company / employer details."""
    name: Optional[str] = None
    website: Optional[str] = None
    industry: Optional[str] = None
    size: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None


class Description(BaseModel):
    """Raw and structured job description content."""
    raw_text: Optional[str] = None
    summary: Optional[str] = None
    responsibilities: Optional[List[str]] = None
    benefits: Optional[List[str]] = None


class Requirements(BaseModel):
    """Job requirements / qualifications."""
    required_skills: Optional[List[str]] = None
    preferred_skills: Optional[List[str]] = None
    experience_years_min: Optional[int] = None
    experience_years_max: Optional[int] = None
    education: Optional[str] = None
    languages: Optional[List[str]] = None


class Compensation(BaseModel):
    """Salary / compensation information."""
    currency: Optional[str] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    period: Optional[str] = None          # e.g. "annual", "monthly", "hourly"
    equity: Optional[bool] = None
    raw_text: Optional[str] = None


class Application(BaseModel):
    """Application channel details."""
    apply_url: Optional[str] = None
    apply_email: Optional[str] = None
    deadline: Optional[datetime] = None
    process_notes: Optional[str] = None


class JobDetails(BaseModel):
    """Structured details extracted from a job listing."""
    title: Optional[str] = None
    employment_type: Optional[str] = None   # full-time, part-time, contract, …
    work_mode: Optional[str] = None         # remote, hybrid, onsite
    location: Optional[str] = None
    country: Optional[str] = None
    posted_at: Optional[datetime] = None
    company: Optional[Company] = None
    description: Optional[Description] = None
    requirements: Optional[Requirements] = None
    compensation: Optional[Compensation] = None
    application: Optional[Application] = None
    tags: Optional[List[str]] = None
    seniority: Optional[str] = None


class RetrievalMetadata(BaseModel):
    """Tracking data for the fetch / extraction lifecycle."""
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    last_verified_at: Optional[datetime] = None
    content_hash: Optional[str] = None
    raw_content_path: Optional[str] = None
    consecutive_fetch_failures: int = 0
    fetch_attempts: int = 0
    last_fetch_error: Optional[str] = None


class ValidationResult(BaseModel):
    """Outcome of the validation pipeline."""
    status: ValidationStatus = ValidationStatus.pending
    score: Optional[float] = None          # 0.0 – 1.0
    missing_fields: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    validated_at: Optional[datetime] = None


# ─────────────────────────────────────────────────────────────────────────────
# Top-level record
# ─────────────────────────────────────────────────────────────────────────────

class JobRecord(BaseModel):
    """
    Top-level document stored in the `jobs` collection.
    Maps 1-to-1 with a MongoDB document (``_id`` is managed by PyMongo).
    """
    # Identity / deduplication keys
    canonical_url: Optional[str] = None
    external_job_id: Optional[str] = None
    content_hash: Optional[str] = None

    # Lifecycle
    lifecycle_status: LifecycleStatus = LifecycleStatus.unknown

    # Structured sub-documents
    source: Optional[JobSource] = None
    details: Optional[JobDetails] = None
    retrieval: RetrievalMetadata = Field(default_factory=RetrievalMetadata)
    validation: ValidationResult = Field(default_factory=ValidationResult)

    # Timestamps (top-level for fast querying)
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    last_verified_at: Optional[datetime] = None

    # Raw storage pointer (complements retrieval.raw_content_path)
    raw_content_path: Optional[str] = None

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    model_config = {"populate_by_name": True}
