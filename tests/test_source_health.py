"""
Tests for SourceHealthMonitor — extraction rate tracking and alerts.
"""
from __future__ import annotations

import logging

import pytest

from job_discovery.health.source_health import SourceHealthMonitor, TRACKED_FIELDS


def _statuses(present=None, failed=None, not_avail=None):
    """Build a field_statuses dict for record()."""
    statuses = {}
    for f in TRACKED_FIELDS:
        if present and f in present:
            statuses[f] = "present"
        elif failed and f in failed:
            statuses[f] = "extraction_failed"
        else:
            statuses[f] = "not_available"
    return statuses


class TestSourceHealthMonitor:
    def setup_method(self):
        self.monitor = SourceHealthMonitor(alert_threshold=0.5, min_samples=3)

    # ── Basic recording ──────────────────────────────────────────────────

    def test_record_creates_metrics(self):
        self.monitor.record("naukri", _statuses(present=["title"]))
        m = self.monitor.get_metrics("naukri")
        assert m is not None
        assert m.source_name == "naukri"

    def test_total_jobs_increments(self):
        for _ in range(5):
            self.monitor.record("indeed", _statuses(present=["title"]))
        m = self.monitor.get_metrics("indeed")
        assert m.total_jobs == 5

    def test_successful_extraction_counted(self):
        self.monitor.record("naukri", _statuses(present=["title", "company"]))
        m = self.monitor.get_metrics("naukri")
        assert m.successful_extractions == 1

    def test_failed_extraction_counted(self):
        self.monitor.record("naukri", _statuses(not_avail=TRACKED_FIELDS))
        m = self.monitor.get_metrics("naukri")
        assert m.failed_extractions == 1

    def test_field_metrics_present_count(self):
        for _ in range(3):
            self.monitor.record("linkedin", _statuses(present=["title", "location"]))
        m = self.monitor.get_metrics("linkedin")
        assert m.get_field("title").present == 3
        assert m.get_field("location").present == 3

    def test_field_metrics_not_available(self):
        self.monitor.record("naukri", _statuses(not_avail=["salary"]))
        m = self.monitor.get_metrics("naukri")
        assert m.get_field("salary").not_available == 1

    def test_extraction_rate_calculation(self):
        # 2 successes, 1 failure out of 3
        for i in range(2):
            self.monitor.record("src", _statuses(present=["title"]))
        self.monitor.record("src", _statuses())  # no title
        m = self.monitor.get_metrics("src")
        assert abs(m.extraction_rate() - 2/3) < 0.01

    # ── Alert threshold ──────────────────────────────────────────────────

    def test_no_alert_below_min_samples(self):
        monitor = SourceHealthMonitor(alert_threshold=0.9, min_samples=5)
        # Only 2 jobs — below min_samples=5, no alert
        for _ in range(2):
            monitor.record("hirist", _statuses())  # all failed
        # No exception raised — alert not triggered yet

    def test_alert_emitted_when_rate_below_threshold(self, caplog):
        monitor = SourceHealthMonitor(alert_threshold=0.8, min_samples=3)
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                monitor.record("wellfound", _statuses())  # all title-absent
        assert any("SOURCE_HEALTH_ALERT" in r.message for r in caplog.records)

    def test_no_alert_when_rate_above_threshold(self, caplog):
        monitor = SourceHealthMonitor(alert_threshold=0.3, min_samples=3)
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                monitor.record("naukri", _statuses(present=["title"]))
        assert not any("SOURCE_HEALTH_ALERT" in r.message for r in caplog.records)

    def test_different_sources_independent(self):
        for _ in range(5):
            self.monitor.record("linkedin", _statuses(present=["title"]))
        for _ in range(5):
            self.monitor.record("naukri", _statuses())   # all failing
        m_li = self.monitor.get_metrics("linkedin")
        m_nk = self.monitor.get_metrics("naukri")
        assert m_li.extraction_rate() == 1.0
        assert m_nk.extraction_rate() == 0.0

    def test_all_metrics_returns_all_sources(self):
        self.monitor.record("linkedin", _statuses(present=["title"]))
        self.monitor.record("naukri",   _statuses(present=["title"]))
        all_m = self.monitor.all_metrics()
        assert "linkedin" in all_m
        assert "naukri" in all_m

    def test_last_updated_set(self):
        self.monitor.record("naukri", _statuses(present=["title"]))
        m = self.monitor.get_metrics("naukri")
        assert m.last_updated is not None

    def test_source_degradation_alert_contains_source_name(self, caplog):
        monitor = SourceHealthMonitor(alert_threshold=0.9, min_samples=3)
        with caplog.at_level(logging.WARNING):
            for _ in range(4):
                monitor.record("hirist", _statuses())
        alert_records = [r for r in caplog.records if "SOURCE_HEALTH_ALERT" in r.message]
        assert any("hirist" in r.message for r in alert_records)
