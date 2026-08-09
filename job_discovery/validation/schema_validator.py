"""
Schema validator — validates job field values against allowed types,
formats, and ranges.

Separate from completeness (which checks presence);
this checks correctness of values that ARE present.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SchemaError:
    field_name: str
    message: str
    value: Any = None


@dataclass
class SchemaValidationResult:
    valid: bool
    errors: List[SchemaError]
    validated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


_URL_RE = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.I)
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_HASH_RE  = re.compile(r"^[0-9a-f]{64}$", re.I)

_ALLOWED_EMPLOYMENT_TYPES = {
    "full-time", "part-time", "contract", "internship",
    "temporary", "permanent", "freelance",
}
_ALLOWED_WORK_MODES = {"remote", "hybrid", "onsite", "on-site", "in-office"}
_ALLOWED_LIFECYCLE   = {"active", "expired", "removed", "unknown", "unavailable"}


class SchemaValidator:
    """
    Validates field values in a job record dict.
    Does NOT check completeness — only correctness.
    """

    def validate(self, job: Dict[str, Any]) -> SchemaValidationResult:
        errors: List[SchemaError] = []

        # canonical_url
        url = job.get("canonical_url")
        if url and not _URL_RE.match(str(url)):
            errors.append(SchemaError("canonical_url", f"Invalid URL format: {url!r}", url))

        # content_hash
        ch = job.get("content_hash") or (job.get("retrieval") or {}).get("content_hash")
        if ch and not _HASH_RE.match(str(ch)):
            errors.append(SchemaError("content_hash", f"Must be 64-char hex: {ch!r}", ch))

        # employment_type
        et = (job.get("details") or {}).get("employment_type") or job.get("employment_type")
        if et and str(et).lower() not in _ALLOWED_EMPLOYMENT_TYPES:
            errors.append(SchemaError(
                "employment_type",
                f"Unrecognised value {et!r}. Allowed: {sorted(_ALLOWED_EMPLOYMENT_TYPES)}",
                et,
            ))

        # work_mode
        wm = (job.get("details") or {}).get("work_mode") or job.get("work_mode")
        if wm and str(wm).lower() not in _ALLOWED_WORK_MODES:
            errors.append(SchemaError(
                "work_mode",
                f"Unrecognised value {wm!r}. Allowed: {sorted(_ALLOWED_WORK_MODES)}",
                wm,
            ))

        # lifecycle_status
        ls = job.get("lifecycle_status")
        if ls and str(ls).lower() not in _ALLOWED_LIFECYCLE:
            errors.append(SchemaError("lifecycle_status", f"Unknown status: {ls!r}", ls))

        # apply_email
        email_raw = (
            (job.get("details") or {}).get("application") or {}
        ).get("apply_email") or job.get("apply_email")
        if email_raw and not _EMAIL_RE.match(str(email_raw)):
            errors.append(SchemaError("apply_email", f"Invalid email: {email_raw!r}", email_raw))

        # experience range sanity
        details = job.get("details") or {}
        reqs = details.get("requirements") or {}
        exp_min = reqs.get("experience_years_min")
        exp_max = reqs.get("experience_years_max")
        if exp_min is not None and exp_max is not None:
            try:
                if float(exp_min) > float(exp_max):
                    errors.append(SchemaError(
                        "experience",
                        f"experience_years_min ({exp_min}) > experience_years_max ({exp_max})",
                    ))
            except (TypeError, ValueError):
                pass

        # salary sanity
        comp = details.get("compensation") or {}
        sal_min = comp.get("min_amount")
        sal_max = comp.get("max_amount")
        if sal_min is not None and sal_max is not None:
            try:
                if float(sal_min) > float(sal_max):
                    errors.append(SchemaError(
                        "salary",
                        f"salary min ({sal_min}) > max ({sal_max})",
                    ))
            except (TypeError, ValueError):
                pass

        return SchemaValidationResult(valid=len(errors) == 0, errors=errors)
