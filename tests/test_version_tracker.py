"""
Tests for VersionTracker — versioning, change detection,
freshness timestamps, and lifecycle transitions.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from job_discovery.versioning.version_tracker import (
    ChangeReport,
    FreshnessRecord,
    JobLifecycle,
    JobVersion,
    VersionTracker,
    detect_changes,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tracker(threshold: int = 3) -> VersionTracker:
    return VersionTracker(removal_threshold=threshold)


def _now() -> datetime:
    return datetime.now(timezone.utc)


_DATA_V1 = {
    "title": "Senior Python Developer",
    "location": "Chennai",
    "salary_raw": "8-14 LPA",
    "experience_raw": "3-6 years",
    "required_skills": ["Python", "FastAPI"],
    "description_text": "We are hiring...",
}

_DATA_V2 = {
    **_DATA_V1,
    "salary_raw": "10-16 LPA",        # changed
    "required_skills": ["Python", "FastAPI", "Kubernetes"],  # added skill
}

_DATA_V3 = {
    **_DATA_V2,
    "description_text": "Updated job description text.",   # changed
}


# ─────────────────────────────────────────────────────────────────────────────
# Versioning
# ─────────────────────────────────────────────────────────────────────────────

class TestVersioning:
    def test_first_fetch_creates_version_1(self):
        t = _tracker()
        v = t.record_fetch(content_hash="hash_a", extracted_data=_DATA_V1)
        assert v is not None
        assert v.version_number == 1

    def test_unchanged_content_no_new_version(self):
        t = _tracker()
        t.record_fetch(content_hash="hash_a", extracted_data=_DATA_V1)
        v2 = t.record_fetch(content_hash="hash_a", extracted_data=_DATA_V1)
        assert v2 is None
        assert t.version_count == 1

    def test_changed_content_creates_version_2(self):
        t = _tracker()
        t.record_fetch(content_hash="hash_a", extracted_data=_DATA_V1)
        v2 = t.record_fetch(content_hash="hash_b", extracted_data=_DATA_V2)
        assert v2 is not None
        assert v2.version_number == 2

    def test_three_versions_sequential(self):
        t = _tracker()
        t.record_fetch(content_hash="h1", extracted_data=_DATA_V1)
        t.record_fetch(content_hash="h2", extracted_data=_DATA_V2)
        v3 = t.record_fetch(content_hash="h3", extracted_data=_DATA_V3)
        assert t.version_count == 3
        assert v3.version_number == 3

    def test_version_retains_hash(self):
        t = _tracker()
        v = t.record_fetch(content_hash="abc123", extracted_data=_DATA_V1)
        assert v.content_hash == "abc123"

    def test_version_retains_extracted_data(self):
        t = _tracker()
        v = t.record_fetch(content_hash="h1", extracted_data=_DATA_V1)
        assert v.extracted_data["title"] == "Senior Python Developer"

    def test_version_retains_raw_content_path(self):
        t = _tracker()
        v = t.record_fetch(
            content_hash="h1",
            extracted_data=_DATA_V1,
            raw_content_path="/tmp/raw/h1",
        )
        assert v.raw_content_path == "/tmp/raw/h1"

    def test_version_retains_retrieved_at(self):
        t = _tracker()
        ts = _now()
        v = t.record_fetch(content_hash="h1", extracted_data=_DATA_V1, retrieved_at=ts)
        assert v.retrieved_at == ts

    def test_old_versions_not_overwritten(self):
        t = _tracker()
        t.record_fetch(content_hash="h1", extracted_data=_DATA_V1)
        t.record_fetch(content_hash="h2", extracted_data=_DATA_V2)
        # Version 1 still has original data
        assert t.versions[0].content_hash == "h1"
        assert t.versions[0].extracted_data == _DATA_V1

    def test_latest_version_accessor(self):
        t = _tracker()
        t.record_fetch(content_hash="h1", extracted_data=_DATA_V1)
        t.record_fetch(content_hash="h2", extracted_data=_DATA_V2)
        assert t.latest_version.version_number == 2


# ─────────────────────────────────────────────────────────────────────────────
# Change detection
# ─────────────────────────────────────────────────────────────────────────────

class TestChangeDetection:
    def test_no_changes_on_identical_data(self):
        report = detect_changes(_DATA_V1, _DATA_V1)
        assert not report.has_changes()

    def test_salary_change_detected(self):
        report = detect_changes(_DATA_V1, _DATA_V2)
        assert "salary_raw" in report.changed_fields()

    def test_skills_added_detected(self):
        report = detect_changes(_DATA_V1, _DATA_V2)
        assert "required_skills" in report.changed_fields()

    def test_description_change_detected(self):
        report = detect_changes(_DATA_V2, _DATA_V3)
        assert "description_text" in report.changed_fields()

    def test_before_after_values_preserved(self):
        report = detect_changes(_DATA_V1, _DATA_V2)
        salary_change = next(c for c in report.changes if c.field_name == "salary_raw")
        assert salary_change.before == "8-14 LPA"
        assert salary_change.after == "10-16 LPA"

    def test_change_report_to_dict(self):
        report = detect_changes(_DATA_V1, _DATA_V2)
        d = report.to_dict()
        assert "changed_fields" in d
        assert "details" in d

    def test_version_2_has_change_report(self):
        t = _tracker()
        t.record_fetch(content_hash="h1", extracted_data=_DATA_V1)
        v2 = t.record_fetch(content_hash="h2", extracted_data=_DATA_V2)
        assert v2.changes_from_previous is not None
        assert v2.changes_from_previous.has_changes()

    def test_version_1_has_no_change_report(self):
        t = _tracker()
        v1 = t.record_fetch(content_hash="h1", extracted_data=_DATA_V1)
        assert v1.changes_from_previous is None


# ─────────────────────────────────────────────────────────────────────────────
# Freshness timestamps
# ─────────────────────────────────────────────────────────────────────────────

class TestFreshness:
    def test_first_seen_at_set_on_first_seen(self):
        t = _tracker()
        t.record_seen()
        assert t.first_seen_at is not None

    def test_first_seen_at_never_changes(self):
        t = _tracker()
        t1 = _now()
        t2 = t1 + timedelta(hours=1)
        t.record_seen(t1)
        t.record_seen(t2)
        assert t.first_seen_at == t1   # never overwritten

    def test_last_seen_at_updates(self):
        t = _tracker()
        t1 = _now()
        t2 = t1 + timedelta(hours=2)
        t.record_seen(t1)
        t.record_seen(t2)
        assert t.last_seen_at == t2

    def test_last_verified_at_set_after_fetch(self):
        t = _tracker()
        ts = _now()
        t.record_fetch(content_hash="h1", extracted_data=_DATA_V1, retrieved_at=ts)
        assert t.last_verified_at == ts

    def test_last_verified_at_updates_on_each_fetch(self):
        t = _tracker()
        ts1 = _now()
        ts2 = ts1 + timedelta(hours=1)
        t.record_fetch(content_hash="h1", extracted_data=_DATA_V1, retrieved_at=ts1)
        t.record_fetch(content_hash="h1", extracted_data=_DATA_V1, retrieved_at=ts2)
        assert t.last_verified_at == ts2

    def test_first_seen_at_independent_of_fetch(self):
        """first_seen_at comes from record_seen, not record_fetch."""
        t = _tracker()
        seen_time = _now() - timedelta(hours=5)
        t.record_seen(seen_time)
        t.record_fetch(content_hash="h1", extracted_data=_DATA_V1)
        assert t.first_seen_at == seen_time

    def test_freshness_record_standalone(self):
        fr = FreshnessRecord()
        t1 = _now()
        t2 = t1 + timedelta(hours=1)
        fr.mark_seen(t1)
        fr.mark_seen(t2)
        fr.mark_verified(t2)
        assert fr.first_seen_at == t1
        assert fr.last_seen_at == t2
        assert fr.last_verified_at == t2


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle transitions
# ─────────────────────────────────────────────────────────────────────────────

class TestLifecycle:
    def test_initial_lifecycle_unknown(self):
        t = _tracker()
        assert t.lifecycle == JobLifecycle.UNKNOWN

    def test_successful_fetch_sets_active(self):
        t = _tracker()
        t.record_fetch(content_hash="h1", extracted_data=_DATA_V1)
        assert t.lifecycle == JobLifecycle.ACTIVE

    def test_first_failure_sets_unavailable(self):
        t = _tracker(threshold=3)
        t.record_failure(reason="HTTP 503")
        assert t.lifecycle == JobLifecycle.UNAVAILABLE

    def test_single_failure_not_removed(self):
        t = _tracker(threshold=3)
        t.record_failure(reason="timeout")
        assert t.lifecycle != JobLifecycle.REMOVED

    def test_n_failures_sets_removed(self):
        t = _tracker(threshold=3)
        for _ in range(3):
            t.record_failure(reason="timeout")
        assert t.lifecycle == JobLifecycle.REMOVED

    def test_permanent_failure_sets_expired(self):
        t = _tracker()
        t.record_failure(is_permanent=True, reason="HTTP 410 Gone")
        assert t.lifecycle == JobLifecycle.EXPIRED

    def test_explicit_mark_expired(self):
        t = _tracker()
        t.mark_expired()
        assert t.lifecycle == JobLifecycle.EXPIRED

    def test_recovery_after_failure(self):
        t = _tracker(threshold=5)
        t.record_failure(reason="timeout")
        assert t.lifecycle == JobLifecycle.UNAVAILABLE
        t.record_fetch(content_hash="h1", extracted_data=_DATA_V1)
        assert t.lifecycle == JobLifecycle.ACTIVE

    def test_consecutive_failures_reset_on_success(self):
        t = _tracker(threshold=5)
        t.record_failure(reason="err")
        t.record_failure(reason="err")
        t.record_fetch(content_hash="h1", extracted_data=_DATA_V1)
        assert t.consecutive_failures == 0

    def test_no_deletion_after_removed(self):
        """Record must still be accessible even after being marked removed."""
        t = _tracker(threshold=2)
        t.record_fetch(content_hash="h1", extracted_data=_DATA_V1)
        t.record_failure(reason="err")
        t.record_failure(reason="err")
        # Still has version history
        assert t.version_count == 1
        assert t.lifecycle == JobLifecycle.REMOVED

    def test_expired_not_deleted(self):
        t = _tracker()
        t.record_fetch(content_hash="h1", extracted_data=_DATA_V1)
        t.mark_expired()
        assert t.version_count == 1
        assert t.latest_version is not None


# ─────────────────────────────────────────────────────────────────────────────
# to_dict serialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestVersionTrackerSerialization:
    def test_to_dict_has_required_keys(self):
        t = _tracker()
        t.record_seen()
        t.record_fetch(content_hash="h1", extracted_data=_DATA_V1)
        d = t.to_dict()
        for key in ("version_count", "lifecycle", "first_seen_at",
                    "last_seen_at", "last_verified_at", "versions"):
            assert key in d

    def test_to_dict_version_entries(self):
        t = _tracker()
        t.record_fetch(content_hash="h1", extracted_data=_DATA_V1)
        t.record_fetch(content_hash="h2", extracted_data=_DATA_V2)
        d = t.to_dict()
        assert d["version_count"] == 2
        assert len(d["versions"]) == 2
        assert d["versions"][0]["version_number"] == 1
        assert d["versions"][1]["version_number"] == 2

    def test_to_dict_changes_preserved(self):
        t = _tracker()
        t.record_fetch(content_hash="h1", extracted_data=_DATA_V1)
        t.record_fetch(content_hash="h2", extracted_data=_DATA_V2)
        d = t.to_dict()
        v2 = d["versions"][1]
        assert v2["changes"] is not None
        assert "salary_raw" in v2["changes"]["changed_fields"]


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic 5-run scenario
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterministicScenario:
    """
    Run 1: new job
    Run 2: same content
    Run 3: modified content
    Run 4: temporary fetch failure
    Run 5: repeated failures → removed
    """

    def test_full_scenario(self):
        t = VersionTracker(removal_threshold=2)

        ts = [_now() + timedelta(hours=i) for i in range(7)]

        # Run 1: job discovered and fetched
        t.record_seen(ts[0])
        v1 = t.record_fetch(content_hash="h1", extracted_data=_DATA_V1, retrieved_at=ts[1])
        assert v1 is not None
        assert v1.version_number == 1
        assert t.lifecycle == JobLifecycle.ACTIVE
        assert t.first_seen_at == ts[0]
        first_seen = t.first_seen_at

        # Run 2: same content — no new version
        t.record_seen(ts[2])
        v2 = t.record_fetch(content_hash="h1", extracted_data=_DATA_V1, retrieved_at=ts[2])
        assert v2 is None                          # no new version
        assert t.version_count == 1               # still 1
        assert t.last_verified_at == ts[2]
        assert t.first_seen_at == first_seen       # UNCHANGED

        # Run 3: content changed
        t.record_seen(ts[3])
        v3 = t.record_fetch(content_hash="h2", extracted_data=_DATA_V2, retrieved_at=ts[3])
        assert v3 is not None
        assert v3.version_number == 2
        assert t.version_count == 2
        assert "salary_raw" in v3.changes_from_previous.changed_fields()

        # Run 4: temporary failure → unavailable
        t.record_failure(reason="HTTP 503 transient")
        assert t.lifecycle == JobLifecycle.UNAVAILABLE
        assert t.consecutive_failures == 1
        assert t.version_count == 2               # versions preserved

        # Run 5: another failure → threshold=2 reached → removed
        t.record_failure(reason="HTTP 503 again")
        assert t.lifecycle == JobLifecycle.REMOVED
        assert t.consecutive_failures == 2

        # No records deleted — full history available
        assert t.version_count == 2
        assert t.versions[0].content_hash == "h1"
        assert t.versions[1].content_hash == "h2"
        assert t.first_seen_at == first_seen   # still unchanged
