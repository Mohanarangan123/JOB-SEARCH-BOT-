"""
VersionTracker — job versioning, change detection, freshness, and lifecycle.

Rules:
  - Content unchanged (same hash) → no new version created.
  - Content changed → new version (incrementing number).
  - first_seen_at set once and never changed.
  - last_seen_at updated when job appears in search results.
  - last_verified_at updated after successful fetch.
  - Versions never overwritten; full history retained.
  - No records deleted.

Lifecycle transitions:
  active      ← successful fetch
  unavailable ← temporary fetch failure
  expired     ← source signals expiration (410, "expired" in content)
  removed     ← N consecutive failures ≥ threshold

Change detection fields:
  title, location, salary, experience, skills, description, responsibilities
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Job version record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JobVersion:
    version_number: int
    content_hash: str
    raw_content_path: Optional[str]
    extracted_data: Dict[str, Any]
    retrieved_at: datetime
    changes_from_previous: Optional["ChangeReport"] = None


@dataclass
class FieldChange:
    field_name: str
    before: Any
    after: Any

    def is_meaningful(self) -> bool:
        return self.before != self.after


@dataclass
class ChangeReport:
    """Summary of meaningful changes between two versions."""
    changes: List[FieldChange] = field(default_factory=list)

    def has_changes(self) -> bool:
        return bool(self.changes)

    def changed_fields(self) -> List[str]:
        return [c.field_name for c in self.changes]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "changed_fields": self.changed_fields(),
            "details": [
                {"field": c.field_name, "before": c.before, "after": c.after}
                for c in self.changes
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class JobLifecycle(str, Enum):
    ACTIVE      = "active"
    UNAVAILABLE = "unavailable"
    EXPIRED     = "expired"
    REMOVED     = "removed"
    UNKNOWN     = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Freshness timestamps
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FreshnessRecord:
    """
    Tracks all three freshness timestamps for a job.
    These are system-level timestamps, NOT the source's posted_date.
    """
    first_seen_at: Optional[datetime] = None      # set once; never changed
    last_seen_at:  Optional[datetime] = None      # updated on search result
    last_verified_at: Optional[datetime] = None   # updated on successful fetch

    def mark_seen(self, when: Optional[datetime] = None) -> None:
        """Call when job appears in a search result."""
        now = when or datetime.now(timezone.utc)
        if self.first_seen_at is None:
            self.first_seen_at = now
        self.last_seen_at = now

    def mark_verified(self, when: Optional[datetime] = None) -> None:
        """Call after a successful page fetch."""
        self.last_verified_at = when or datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Change detector (deterministic, field-by-field)
# ─────────────────────────────────────────────────────────────────────────────

# Fields we care about for change detection
_CHANGE_FIELDS = [
    "title",
    "location",
    "salary_raw",
    "experience_raw",
    "employment_type",
    "work_mode",
    "description_text",
    "responsibilities",
    "requirements_text",
    # LLM output fields (when available)
    "required_skills",
    "preferred_skills",
]


def detect_changes(
    old_data: Dict[str, Any],
    new_data: Dict[str, Any],
) -> ChangeReport:
    """
    Compare two extracted-data dicts and return a ChangeReport.
    Only reports fields that are actually different.
    """
    report = ChangeReport()
    for fname in _CHANGE_FIELDS:
        old_val = _flatten(old_data.get(fname))
        new_val = _flatten(new_data.get(fname))
        if old_val != new_val:
            report.changes.append(FieldChange(
                field_name=fname,
                before=old_val,
                after=new_val,
            ))
    return report


def _flatten(value: Any) -> Any:
    """Normalise lists to sorted tuples for stable comparison."""
    if isinstance(value, list):
        return tuple(sorted(str(v) for v in value))
    return value


# ─────────────────────────────────────────────────────────────────────────────
# VersionTracker
# ─────────────────────────────────────────────────────────────────────────────

class VersionTracker:
    """
    Manages versioning, freshness, and lifecycle for a single job URL.

    Designed to be instantiated per job and persisted externally
    (e.g. serialised to MongoDB via the job_repository).

    Args:
        removal_threshold: consecutive fetch failures before marking removed.
    """

    def __init__(self, *, removal_threshold: int = 3) -> None:
        self._threshold = removal_threshold
        self.versions: List[JobVersion] = []
        self.freshness = FreshnessRecord()
        self.lifecycle: JobLifecycle = JobLifecycle.UNKNOWN
        self.consecutive_failures: int = 0

    # ------------------------------------------------------------------ #
    # Versioning
    # ------------------------------------------------------------------ #

    def record_fetch(
        self,
        *,
        content_hash: str,
        extracted_data: Dict[str, Any],
        raw_content_path: Optional[str] = None,
        retrieved_at: Optional[datetime] = None,
    ) -> Optional[JobVersion]:
        """
        Record a successful fetch.

        If content_hash matches the latest version, no new version is created.
        Returns the new JobVersion if created, or None if content unchanged.
        """
        now = retrieved_at or datetime.now(timezone.utc)

        # Freshness
        self.freshness.mark_verified(now)
        self.consecutive_failures = 0
        self.lifecycle = JobLifecycle.ACTIVE

        # Check if content changed
        if self.versions and self.versions[-1].content_hash == content_hash:
            logger.debug(
                "Content unchanged (hash=%s) — no new version created.", content_hash[:8]
            )
            return None

        # Build change report
        prev_data = self.versions[-1].extracted_data if self.versions else {}
        change_report = detect_changes(prev_data, extracted_data) if self.versions else None

        version = JobVersion(
            version_number=len(self.versions) + 1,
            content_hash=content_hash,
            raw_content_path=raw_content_path,
            extracted_data=extracted_data,
            retrieved_at=now,
            changes_from_previous=change_report,
        )
        self.versions.append(version)
        logger.info(
            "New version %d created for hash=%s changed_fields=%s",
            version.version_number,
            content_hash[:8],
            change_report.changed_fields() if change_report else [],
        )
        return version

    def record_seen(self, when: Optional[datetime] = None) -> None:
        """Call when job URL appears in a search result (not a fetch)."""
        self.freshness.mark_seen(when)

    # ------------------------------------------------------------------ #
    # Lifecycle management
    # ------------------------------------------------------------------ #

    def record_failure(
        self,
        *,
        is_permanent: bool = False,
        reason: str = "",
    ) -> JobLifecycle:
        """
        Record a fetch failure and transition lifecycle accordingly.

        Returns the new lifecycle status.
        """
        if is_permanent:
            # 410 Gone or explicit source expiration → expired
            self.lifecycle = JobLifecycle.EXPIRED
            logger.info("Job marked EXPIRED: %s", reason)
            return self.lifecycle

        self.consecutive_failures += 1
        logger.warning(
            "Fetch failure #%d for job (threshold=%d): %s",
            self.consecutive_failures, self._threshold, reason,
        )

        if self.consecutive_failures >= self._threshold:
            self.lifecycle = JobLifecycle.REMOVED
            logger.warning(
                "Job marked REMOVED after %d consecutive failures.",
                self.consecutive_failures,
            )
        else:
            self.lifecycle = JobLifecycle.UNAVAILABLE

        return self.lifecycle

    def mark_expired(self) -> None:
        """Explicitly mark as expired (e.g. source returns 410)."""
        self.lifecycle = JobLifecycle.EXPIRED

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    @property
    def latest_version(self) -> Optional[JobVersion]:
        return self.versions[-1] if self.versions else None

    @property
    def version_count(self) -> int:
        return len(self.versions)

    @property
    def first_seen_at(self) -> Optional[datetime]:
        return self.freshness.first_seen_at

    @property
    def last_seen_at(self) -> Optional[datetime]:
        return self.freshness.last_seen_at

    @property
    def last_verified_at(self) -> Optional[datetime]:
        return self.freshness.last_verified_at

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable snapshot for persistence."""
        return {
            "version_count":          self.version_count,
            "lifecycle":              self.lifecycle.value,
            "consecutive_failures":   self.consecutive_failures,
            "first_seen_at":          self.freshness.first_seen_at,
            "last_seen_at":           self.freshness.last_seen_at,
            "last_verified_at":       self.freshness.last_verified_at,
            "versions": [
                {
                    "version_number":  v.version_number,
                    "content_hash":    v.content_hash,
                    "raw_content_path": v.raw_content_path,
                    "retrieved_at":    v.retrieved_at,
                    "changes":         v.changes_from_previous.to_dict()
                                       if v.changes_from_previous else None,
                }
                for v in self.versions
            ],
        }
