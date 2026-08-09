"""
Tests for completeness validation, schema validation, and quality checker.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from job_discovery.validation.completeness import (
    CompletenessChecker,
    CompletenessStatus,
    FieldStatus,
)
from job_discovery.validation.quality_checker import QualityChecker
from job_discovery.validation.schema_validator import SchemaValidator
from job_discovery.models.job import ValidationStatus


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _full_job() -> dict:
    """A rich job dict that should score well."""
    return {
        "canonical_url": "https://naukri.com/job-listings-python-dev-12345678",
        "content_hash": "a" * 64,
        "source": {"source_name": "naukri"},
        "last_verified_at": datetime.now(timezone.utc).isoformat(),
        "title": "Senior Python Developer",
        "details": {
            "title": "Senior Python Developer",
            "employment_type": "full-time",
            "work_mode": "hybrid",
            "location": "Chennai",
            "company": {"name": "TechCorp"},
            "description": {"raw_text": "We are hiring a Python developer..."},
            "requirements": {
                "required_skills": ["Python", "FastAPI"],
                "experience_years_min": 3,
                "experience_years_max": 6,
            },
            "compensation": {"min_amount": 800000, "max_amount": 1400000},
            "application": {
                "apply_url": "https://techcorp.in/apply",
                "apply_email": "careers@techcorp.in",
            },
        },
        "raw_content_path": "/tmp/raw",
        "retrieval": {
            "content_hash": "a" * 64,
            "raw_content_path": "/tmp/raw",
            "last_verified_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def _minimal_job() -> dict:
    """Just URL and source — missing most fields."""
    return {
        "canonical_url": "https://linkedin.com/jobs/view/123",
        "source": {"source_name": "linkedin"},
    }


def _extraction_failed_job() -> dict:
    """Page text exists but title/description empty → extraction failed."""
    return {
        "canonical_url": "https://naukri.com/job/1",
        "source": {"source_name": "naukri"},
        "page_text": "A" * 500,   # has content
        "title": None,            # but no title extracted
    }


# ─────────────────────────────────────────────────────────────────────────────
# CompletenessChecker
# ─────────────────────────────────────────────────────────────────────────────

class TestCompletenessChecker:
    def setup_method(self):
        self.checker = CompletenessChecker()

    def test_full_job_high_score(self):
        result = self.checker.check(_full_job())
        assert result.score > 0.7

    def test_full_job_status_complete_or_partial(self):
        result = self.checker.check(_full_job())
        assert result.status in (CompletenessStatus.COMPLETE, CompletenessStatus.PARTIAL)

    def test_minimal_job_low_score(self):
        result = self.checker.check(_minimal_job())
        assert result.score < 0.5

    def test_empty_job_critical(self):
        result = self.checker.check({})
        assert result.status in (CompletenessStatus.CRITICAL, CompletenessStatus.FAILED)

    def test_not_available_for_missing_salary(self):
        """Salary not on page → NOT_AVAILABLE, not EXTRACTION_FAILED."""
        job = _minimal_job()
        result = self.checker.check(job)
        salary_field = next(r for r in result.field_results if r.name == "salary")
        assert salary_field.status == FieldStatus.NOT_AVAILABLE

    def test_extraction_failed_when_page_has_content_but_no_title(self):
        """Page text present + title missing → EXTRACTION_FAILED."""
        result = self.checker.check(_extraction_failed_job())
        title_field = next(r for r in result.field_results if r.name == "title")
        assert title_field.status == FieldStatus.EXTRACTION_FAILED

    def test_not_available_when_no_page_content(self):
        """No page text + no title → NOT_AVAILABLE (nothing to extract from)."""
        job = {"canonical_url": "https://a.com/1"}
        result = self.checker.check(job)
        title_field = next(r for r in result.field_results if r.name == "title")
        assert title_field.status == FieldStatus.NOT_AVAILABLE

    def test_warnings_populated(self):
        result = self.checker.check(_minimal_job())
        assert isinstance(result.warnings, list)

    def test_missing_fields_list(self):
        result = self.checker.check(_minimal_job())
        assert "title" in result.missing_fields

    def test_score_between_0_and_1(self):
        for job in [_full_job(), _minimal_job(), _extraction_failed_job(), {}]:
            result = self.checker.check(job)
            assert 0.0 <= result.score <= 1.0

    def test_validated_at_set(self):
        result = self.checker.check(_full_job())
        assert result.validated_at is not None

    def test_field_results_covers_all_spec_fields(self):
        result = self.checker.check(_full_job())
        field_names = {r.name for r in result.field_results}
        for required_field in ["title", "description", "canonical_url", "source"]:
            assert required_field in field_names


# ─────────────────────────────────────────────────────────────────────────────
# SchemaValidator
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaValidator:
    def setup_method(self):
        self.validator = SchemaValidator()

    def test_valid_full_job_no_errors(self):
        result = self.validator.validate(_full_job())
        assert result.valid

    def test_invalid_url(self):
        job = {"canonical_url": "not-a-url"}
        result = self.validator.validate(job)
        assert not result.valid
        assert any(e.field_name == "canonical_url" for e in result.errors)

    def test_valid_url_passes(self):
        job = {"canonical_url": "https://naukri.com/job/123"}
        result = self.validator.validate(job)
        url_errors = [e for e in result.errors if e.field_name == "canonical_url"]
        assert len(url_errors) == 0

    def test_invalid_content_hash(self):
        job = {"content_hash": "tooshort"}
        result = self.validator.validate(job)
        assert any(e.field_name == "content_hash" for e in result.errors)

    def test_valid_content_hash(self):
        job = {"content_hash": "a" * 64}
        result = self.validator.validate(job)
        hash_errors = [e for e in result.errors if e.field_name == "content_hash"]
        assert len(hash_errors) == 0

    def test_invalid_employment_type(self):
        job = {"details": {"employment_type": "gig-work"}}
        result = self.validator.validate(job)
        assert any(e.field_name == "employment_type" for e in result.errors)

    def test_valid_employment_type(self):
        job = {"details": {"employment_type": "full-time"}}
        result = self.validator.validate(job)
        assert all(e.field_name != "employment_type" for e in result.errors)

    def test_invalid_work_mode(self):
        job = {"details": {"work_mode": "spaceship"}}
        result = self.validator.validate(job)
        assert any(e.field_name == "work_mode" for e in result.errors)

    def test_salary_min_greater_than_max(self):
        job = {"details": {"compensation": {"min_amount": 1000000, "max_amount": 500000}}}
        result = self.validator.validate(job)
        assert any(e.field_name == "salary" for e in result.errors)

    def test_experience_min_greater_than_max(self):
        job = {"details": {"requirements": {
            "experience_years_min": 8,
            "experience_years_max": 3,
        }}}
        result = self.validator.validate(job)
        assert any(e.field_name == "experience" for e in result.errors)

    def test_empty_job_valid(self):
        """Empty dict has nothing to validate — should pass."""
        result = self.validator.validate({})
        assert result.valid

    def test_validated_at_set(self):
        result = self.validator.validate({})
        assert result.validated_at is not None


# ─────────────────────────────────────────────────────────────────────────────
# QualityChecker
# ─────────────────────────────────────────────────────────────────────────────

class TestQualityChecker:
    def setup_method(self):
        self.checker = QualityChecker()

    def test_full_job_valid_or_partial(self):
        result = self.checker.check(_full_job())
        assert result.status in (ValidationStatus.valid, ValidationStatus.partial)

    def test_empty_job_invalid(self):
        result = self.checker.check({})
        assert result.status == ValidationStatus.invalid

    def test_score_present(self):
        result = self.checker.check(_full_job())
        assert result.score is not None
        assert 0.0 <= result.score <= 1.0

    def test_missing_fields_list(self):
        result = self.checker.check(_minimal_job())
        assert isinstance(result.missing_fields, list)

    def test_validated_at_set(self):
        result = self.checker.check(_full_job())
        assert result.validated_at is not None

    def test_schema_errors_appear_in_warnings(self):
        job = {"canonical_url": "BAD_URL", **_full_job()}
        job["canonical_url"] = "BAD_URL"
        result = self.checker.check(job)
        warnings = result.warnings or []
        assert any("Schema error" in w or "canonical_url" in w.lower() for w in warnings)
