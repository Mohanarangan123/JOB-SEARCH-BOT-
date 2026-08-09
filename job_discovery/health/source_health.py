"""
Source health monitor.

Tracks per-source extraction quality metrics and emits
SOURCE_HEALTH_ALERT log warnings when extraction rates degrade.

No dashboard — logging only.
No embeddings, no vector operations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Alert log marker — easy to grep in log files
_ALERT_TAG = "SOURCE_HEALTH_ALERT"

# Fields tracked for extraction rate measurement
TRACKED_FIELDS: List[str] = [
    "title",
    "company",
    "location",
    "description",
    "experience",
    "salary",
    "employment_type",
]


@dataclass
class FieldMetrics:
    """Extraction success/failure counts for one field."""
    field_name: str
    total: int = 0
    present: int = 0
    extraction_failed: int = 0
    not_available: int = 0

    @property
    def extraction_rate(self) -> float:
        """Fraction of total attempts where field was present."""
        return self.present / self.total if self.total > 0 else 0.0

    @property
    def failure_rate(self) -> float:
        """Fraction of total attempts that were extraction failures."""
        return self.extraction_failed / self.total if self.total > 0 else 0.0


@dataclass
class SourceMetrics:
    """Aggregated extraction metrics for one source."""
    source_name: str
    total_jobs: int = 0
    successful_extractions: int = 0  # at least title present
    failed_extractions: int = 0
    field_metrics: Dict[str, FieldMetrics] = field(default_factory=dict)
    last_updated: Optional[datetime] = None

    def extraction_rate(self) -> float:
        return (
            self.successful_extractions / self.total_jobs
            if self.total_jobs > 0 else 0.0
        )

    def get_field(self, name: str) -> FieldMetrics:
        if name not in self.field_metrics:
            self.field_metrics[name] = FieldMetrics(field_name=name)
        return self.field_metrics[name]


# ─────────────────────────────────────────────────────────────────────────────
# SourceHealthMonitor
# ─────────────────────────────────────────────────────────────────────────────

class SourceHealthMonitor:
    """
    In-process source health tracker.

    Usage:
        monitor = SourceHealthMonitor(alert_threshold=0.5)
        monitor.record(source_name="naukri", field_results=completeness_result)
        # Alerts logged automatically when threshold crossed.
    """

    def __init__(
        self,
        *,
        alert_threshold: float = 0.5,
        min_samples: int = 5,
    ) -> None:
        """
        Args:
            alert_threshold: Extraction rate below this triggers an alert.
            min_samples:      Minimum jobs seen before alerting.
        """
        self._threshold = alert_threshold
        self._min_samples = min_samples
        self._metrics: Dict[str, SourceMetrics] = {}

    def record(
        self,
        source_name: str,
        field_statuses: Dict[str, str],  # field_name → FieldStatus value
    ) -> None:
        """
        Record extraction outcomes for one job from source_name.

        field_statuses maps field names to FieldStatus string values:
          "present" | "not_available" | "extraction_failed"
        """
        m = self._get_or_create(source_name)
        m.total_jobs += 1

        # Check if this extraction was successful (title present)
        title_status = field_statuses.get("title", "not_available")
        if title_status == "present":
            m.successful_extractions += 1
        else:
            m.failed_extractions += 1

        # Record per-field metrics
        for fname in TRACKED_FIELDS:
            status = field_statuses.get(fname, "not_available")
            fm = m.get_field(fname)
            fm.total += 1
            if status == "present":
                fm.present += 1
            elif status == "extraction_failed":
                fm.extraction_failed += 1
            else:
                fm.not_available += 1

        m.last_updated = datetime.now(timezone.utc)

        # Check alert condition
        self._maybe_alert(m)

    def get_metrics(self, source_name: str) -> Optional[SourceMetrics]:
        return self._metrics.get(source_name)

    def all_metrics(self) -> Dict[str, SourceMetrics]:
        return dict(self._metrics)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _get_or_create(self, source_name: str) -> SourceMetrics:
        if source_name not in self._metrics:
            self._metrics[source_name] = SourceMetrics(source_name=source_name)
        return self._metrics[source_name]

    def _maybe_alert(self, m: SourceMetrics) -> None:
        if m.total_jobs < self._min_samples:
            return
        rate = m.extraction_rate()
        if rate < self._threshold:
            logger.warning(
                "%s source=%r extraction_rate=%.2f total_jobs=%d "
                "failed=%d threshold=%.2f — extraction quality degraded",
                _ALERT_TAG,
                m.source_name,
                rate,
                m.total_jobs,
                m.failed_extractions,
                self._threshold,
            )
