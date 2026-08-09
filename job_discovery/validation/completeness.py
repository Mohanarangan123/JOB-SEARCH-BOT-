"""
Completeness validator.

Distinguishes two distinct absence reasons:
  NOT_AVAILABLE     — field was never present on the source page.
  EXTRACTION_FAILED — page content existed but extractor produced no value.

Rules:
  - A job that merely lacks salary is NOT invalid — salary is optional.
  - A job without title AND without description IS flagged.
  - Missing required fields lower the completeness_score.
  - Never fabricate values to fill gaps.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Field status classification
# ─────────────────────────────────────────────────────────────────────────────

class FieldStatus(str, Enum):
    PRESENT           = "present"
    NOT_AVAILABLE     = "not_available"    # genuinely absent from source
    EXTRACTION_FAILED = "extraction_failed" # source had content, extractor failed


@dataclass
class FieldResult:
    name: str
    status: FieldStatus
    value: Any = None
    note: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Validation outcome
# ─────────────────────────────────────────────────────────────────────────────

class CompletenessStatus(str, Enum):
    COMPLETE  = "complete"   # all required fields present
    PARTIAL   = "partial"    # some required fields missing but acceptable
    CRITICAL  = "critical"   # essential fields missing (title + description)
    FAILED    = "failed"     # extraction failed, no usable data


@dataclass
class CompletenessResult:
    status: CompletenessStatus
    score: float                       # 0.0–1.0
    field_results: List[FieldResult]
    warnings: List[str]
    validated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def missing_fields(self) -> List[str]:
        return [
            r.name for r in self.field_results
            if r.status != FieldStatus.PRESENT
        ]

    @property
    def extraction_failures(self) -> List[str]:
        return [
            r.name for r in self.field_results
            if r.status == FieldStatus.EXTRACTION_FAILED
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Field specification
# ─────────────────────────────────────────────────────────────────────────────

# (field_name, weight, required)
# weight: contribution to completeness_score
# required: True means its absence triggers CRITICAL status
_FIELD_SPEC: List[tuple] = [
    # essential
    ("canonical_url",       0.10, True),
    ("source",              0.05, True),
    ("retrieval_timestamp", 0.05, True),
    ("title",               0.15, True),
    ("description",         0.10, True),
    # important but optional
    ("company",             0.10, False),
    ("location",            0.08, False),
    ("requirements",        0.08, False),
    ("salary",              0.06, False),
    ("employment_type",     0.05, False),
    ("work_mode",           0.05, False),
    ("experience",          0.05, False),
    ("application",         0.05, False),
    ("raw_content",         0.08, False),
    ("content_hash",        0.05, False),
]

_MAX_SCORE = sum(w for _, w, _ in _FIELD_SPEC)


# ─────────────────────────────────────────────────────────────────────────────
# CompletenessChecker
# ─────────────────────────────────────────────────────────────────────────────

class CompletenessChecker:
    """
    Validates a job record's completeness.

    Accepts a plain dict representation of a job record so it can be used
    without importing heavy model classes (avoids circular deps).
    """

    def check(self, job: Dict[str, Any]) -> CompletenessResult:
        """
        Validate completeness of a job record dict.

        The dict keys expected are aligned with JobRecord fields.
        """
        results: List[FieldResult] = []
        warnings: List[str] = []
        earned_score = 0.0

        for field_name, weight, required in _FIELD_SPEC:
            value = self._extract_field(job, field_name)
            status = self._classify(field_name, value, job)

            if status == FieldStatus.PRESENT:
                earned_score += weight
            elif status == FieldStatus.EXTRACTION_FAILED:
                warnings.append(
                    f"Extraction failed for field '{field_name}' — "
                    "page content existed but no value was produced."
                )
            elif required:
                warnings.append(f"Required field '{field_name}' is not available.")

            results.append(FieldResult(
                name=field_name,
                status=status,
                value=value,
            ))

        score = round(earned_score / _MAX_SCORE, 3)

        # Determine overall status
        required_missing = [
            r for r in results
            if r.status != FieldStatus.PRESENT
            and next((req for n, _, req in _FIELD_SPEC if n == r.name), False)
        ]

        has_title = any(
            r.name == "title" and r.status == FieldStatus.PRESENT for r in results
        )
        has_desc = any(
            r.name == "description" and r.status == FieldStatus.PRESENT
            for r in results
        )
        all_extraction_failed = all(
            r.status == FieldStatus.EXTRACTION_FAILED for r in results
        )

        if all_extraction_failed:
            status = CompletenessStatus.FAILED
        elif not has_title and not has_desc:
            status = CompletenessStatus.CRITICAL
        elif required_missing:
            status = CompletenessStatus.PARTIAL
        elif score >= 0.85:
            status = CompletenessStatus.COMPLETE
        else:
            status = CompletenessStatus.PARTIAL

        return CompletenessResult(
            status=status,
            score=score,
            field_results=results,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _extract_field(self, job: Dict[str, Any], field_name: str) -> Any:
        """Extract a field value from the job dict using logical mappings."""
        mapping = {
            "canonical_url":       lambda j: j.get("canonical_url"),
            "source":              lambda j: j.get("source"),
            "retrieval_timestamp": lambda j: (
                j.get("last_verified_at") or
                (j.get("retrieval") or {}).get("last_verified_at")
            ),
            "title":               lambda j: (
                j.get("title") or
                (j.get("details") or {}).get("title")
            ),
            "description":         lambda j: (
                ((j.get("details") or {}).get("description") or {}).get("raw_text") or
                j.get("description_text")
            ),
            "company":             lambda j: (
                j.get("company_name") or
                ((j.get("details") or {}).get("company") or {}).get("name")
            ),
            "location":            lambda j: (
                j.get("location") or
                (j.get("details") or {}).get("location")
            ),
            "requirements":        lambda j: (
                ((j.get("details") or {}).get("requirements") or {}).get("required_skills") or
                j.get("requirements_text")
            ),
            "salary":              lambda j: (
                j.get("salary_raw") or
                (j.get("details") or {}).get("compensation")
            ),
            "employment_type":     lambda j: (
                j.get("employment_type") or
                (j.get("details") or {}).get("employment_type")
            ),
            "work_mode":           lambda j: (
                j.get("work_mode") or
                (j.get("details") or {}).get("work_mode")
            ),
            "experience":          lambda j: j.get("experience_raw"),
            "application":         lambda j: (
                j.get("apply_url") or j.get("apply_email") or
                (j.get("details") or {}).get("application")
            ),
            "raw_content":         lambda j: (
                j.get("raw_content_path") or
                (j.get("retrieval") or {}).get("raw_content_path")
            ),
            "content_hash":        lambda j: (
                j.get("content_hash") or
                (j.get("retrieval") or {}).get("content_hash")
            ),
        }
        extractor = mapping.get(field_name)
        return extractor(job) if extractor else None

    def _classify(
        self, field_name: str, value: Any, job: Dict[str, Any]
    ) -> FieldStatus:
        """
        Decide whether value absence is NOT_AVAILABLE or EXTRACTION_FAILED.

        Heuristic: if the raw page text (page_text) is present and non-empty
        but the field is still None, it's an extraction failure for certain
        fields. For most fields, absence just means not available on source.
        """
        if value is not None and value != "" and value != [] and value != {}:
            return FieldStatus.PRESENT

        page_text = job.get("page_text", "")
        has_content = bool(page_text and len(page_text) > 100)

        # Fields where absence with page content = extraction failure
        extraction_sensitive = {"title", "description"}

        if has_content and field_name in extraction_sensitive:
            return FieldStatus.EXTRACTION_FAILED

        return FieldStatus.NOT_AVAILABLE
