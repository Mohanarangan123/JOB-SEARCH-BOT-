"""Tests for job Pydantic models."""
from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from job_discovery.models.job import (
    Application,
    Company,
    Compensation,
    Description,
    JobDetails,
    JobRecord,
    JobSource,
    LifecycleStatus,
    Requirements,
    RetrievalMetadata,
    ValidationResult,
    ValidationStatus,
)


class TestJobSource:
    def test_minimal(self):
        s = JobSource(source_name="linkedin")
        assert s.source_name == "linkedin"
        assert s.canonical_url is None
        assert s.external_job_id is None

    def test_all_fields(self):
        s = JobSource(
            source_name="naukri",
            source_url="https://naukri.com",
            external_job_id="NKR-123",
            canonical_url="https://naukri.com/job/123",
            scraped_at=datetime(2026, 1, 1),
        )
        assert s.external_job_id == "NKR-123"

    def test_null_optional_fields(self):
        s = JobSource(source_name="indeed")
        for field in ("source_url", "external_job_id", "canonical_url", "scraped_at"):
            assert getattr(s, field) is None


class TestCompany:
    def test_all_none(self):
        c = Company()
        assert c.name is None

    def test_partial(self):
        c = Company(name="Acme Corp", industry="SaaS")
        assert c.name == "Acme Corp"
        assert c.website is None


class TestDescription:
    def test_empty(self):
        d = Description()
        assert d.raw_text is None
        assert d.responsibilities is None

    def test_with_lists(self):
        d = Description(
            raw_text="Build great software.",
            responsibilities=["Design", "Code", "Test"],
        )
        assert len(d.responsibilities) == 3


class TestRequirements:
    def test_defaults(self):
        r = Requirements()
        assert r.required_skills is None
        assert r.experience_years_min is None

    def test_experience_range(self):
        r = Requirements(experience_years_min=2, experience_years_max=5)
        assert r.experience_years_min == 2


class TestCompensation:
    def test_nullable_amounts(self):
        c = Compensation(currency="INR")
        assert c.min_amount is None
        assert c.max_amount is None

    def test_full(self):
        c = Compensation(
            currency="USD",
            min_amount=80_000.0,
            max_amount=120_000.0,
            period="annual",
            equity=True,
        )
        assert c.equity is True


class TestApplication:
    def test_empty(self):
        a = Application()
        assert a.apply_url is None
        assert a.deadline is None


class TestJobDetails:
    def test_minimal(self):
        d = JobDetails()
        assert d.title is None

    def test_nested(self):
        d = JobDetails(
            title="Senior Engineer",
            company=Company(name="TechCorp"),
            compensation=Compensation(currency="USD"),
        )
        assert d.company.name == "TechCorp"
        assert d.compensation.currency == "USD"


class TestRetrievalMetadata:
    def test_defaults(self):
        r = RetrievalMetadata()
        assert r.consecutive_fetch_failures == 0
        assert r.fetch_attempts == 0
        assert r.content_hash is None

    def test_with_hash(self):
        r = RetrievalMetadata(content_hash="abc123", raw_content_path="/tmp/raw.html")
        assert r.content_hash == "abc123"


class TestValidationResult:
    def test_pending_default(self):
        v = ValidationResult()
        assert v.status == ValidationStatus.pending
        assert v.score is None

    def test_valid(self):
        v = ValidationResult(status=ValidationStatus.valid, score=0.95)
        assert v.score == 0.95


class TestJobRecord:
    def test_minimal_creation(self):
        job = JobRecord()
        assert job.lifecycle_status == LifecycleStatus.unknown
        assert job.canonical_url is None
        assert job.content_hash is None
        assert job.retrieval is not None
        assert job.validation is not None

    def test_full_record(self):
        now = datetime(2026, 1, 15, 12, 0, 0)
        job = JobRecord(
            canonical_url="https://linkedin.com/jobs/123",
            external_job_id="LI-123",
            content_hash="sha256abc",
            lifecycle_status=LifecycleStatus.active,
            source=JobSource(source_name="linkedin"),
            details=JobDetails(title="ML Engineer"),
            first_seen_at=now,
            last_seen_at=now,
            last_verified_at=now,
        )
        assert job.lifecycle_status == LifecycleStatus.active
        assert job.details.title == "ML Engineer"
        assert job.first_seen_at == now

    def test_no_semantic_fields(self):
        """No V2 embedding / similarity fields should exist."""
        job = JobRecord()
        assert not hasattr(job, "embedding")
        assert not hasattr(job, "similarity_score")

    def test_dict_round_trip(self):
        job = JobRecord(
            canonical_url="https://example.com/job/1",
            lifecycle_status=LifecycleStatus.active,
        )
        data = job.model_dump()
        restored = JobRecord(**data)
        assert restored.canonical_url == job.canonical_url

    def test_lifecycle_enum_values(self):
        for status in LifecycleStatus:
            job = JobRecord(lifecycle_status=status)
            assert job.lifecycle_status == status
