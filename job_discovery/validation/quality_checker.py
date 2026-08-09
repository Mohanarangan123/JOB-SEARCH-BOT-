"""
QualityChecker — combines completeness and schema validation into one
pass and attaches the result to a ValidationResult model.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from job_discovery.models.job import ValidationResult, ValidationStatus
from job_discovery.validation.completeness import (
    CompletenessChecker,
    CompletenessStatus,
)
from job_discovery.validation.schema_validator import SchemaValidator


class QualityChecker:
    """
    Single entry-point for full quality validation.

    Returns a ValidationResult (Pydantic model) that can be stored
    directly on a JobRecord.
    """

    def __init__(self) -> None:
        self._completeness = CompletenessChecker()
        self._schema = SchemaValidator()

    def check(self, job: Dict[str, Any]) -> ValidationResult:
        comp   = self._completeness.check(job)
        schema = self._schema.validate(job)

        warnings: List[str] = list(comp.warnings)
        for err in schema.errors:
            warnings.append(f"Schema error [{err.field_name}]: {err.message}")

        # Determine overall status
        if comp.status == CompletenessStatus.FAILED:
            status = ValidationStatus.invalid
        elif comp.status == CompletenessStatus.CRITICAL:
            status = ValidationStatus.invalid
        elif not schema.valid or comp.status == CompletenessStatus.PARTIAL:
            status = ValidationStatus.partial
        else:
            status = ValidationStatus.valid

        return ValidationResult(
            status=status,
            score=comp.score,
            missing_fields=comp.missing_fields,
            warnings=warnings if warnings else None,
            validated_at=datetime.now(timezone.utc),
        )
