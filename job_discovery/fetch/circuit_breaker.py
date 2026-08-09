"""
Per-source circuit breaker.

States:
  CLOSED    — normal operation; failures are counted.
  OPEN      — requests blocked after threshold exceeded; waits for recovery_timeout.
  HALF_OPEN — one probe request allowed; success → CLOSED, failure → OPEN.

Design:
  - One CircuitBreaker instance per source_name.
  - CircuitBreakerRegistry manages all instances.
  - A failed source does NOT block other sources.
  - No stealth bypass, proxy rotation, or anti-bot circumvention.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Thread-safe per-source circuit breaker.

    Args:
        source_name:       Identifier for the source being protected.
        failure_threshold: Consecutive failures before opening.
        recovery_timeout:  Seconds before transitioning OPEN → HALF_OPEN.
        success_threshold: Consecutive successes in HALF_OPEN before closing.
    """

    def __init__(
        self,
        source_name: str,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 1,
    ) -> None:
        self.source_name = source_name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold

        self._state: CircuitState = CircuitState.CLOSED
        self._consecutive_failures: int = 0
        self._consecutive_successes: int = 0
        self._opened_at: Optional[datetime] = None
        self._last_reason: Optional[str] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # State accessors
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    @property
    def is_open(self) -> bool:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state == CircuitState.OPEN

    @property
    def allows_request(self) -> bool:
        """True when a request is permitted (CLOSED or HALF_OPEN probe)."""
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._consecutive_failures

    @property
    def last_reason(self) -> Optional[str]:
        with self._lock:
            return self._last_reason

    # ------------------------------------------------------------------ #
    # Call outcomes
    # ------------------------------------------------------------------ #

    def record_success(self) -> None:
        """Call after a successful fetch."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self._success_threshold:
                    self._close()
            elif self._state == CircuitState.CLOSED:
                self._consecutive_failures = 0

    def record_failure(self, reason: str = "") -> None:
        """Call after a transient fetch failure."""
        with self._lock:
            self._last_reason = reason
            if self._state == CircuitState.HALF_OPEN:
                # Probe failed — re-open
                self._open(reason)
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._open(reason)

    # ------------------------------------------------------------------ #
    # Internal transitions
    # ------------------------------------------------------------------ #

    def _open(self, reason: str) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = datetime.now(timezone.utc)
        self._consecutive_successes = 0
        self._last_reason = reason
        logger.warning(
            "CircuitBreaker OPEN: source=%r reason=%r", self.source_name, reason
        )

    def _close(self) -> None:
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_reason = None
        self._opened_at = None
        logger.info("CircuitBreaker CLOSED: source=%r", self.source_name)

    def _maybe_transition_to_half_open(self) -> None:
        """Called under lock — move OPEN → HALF_OPEN if recovery_timeout elapsed."""
        if self._state != CircuitState.OPEN:
            return
        if self._opened_at is None:
            return
        elapsed = (datetime.now(timezone.utc) - self._opened_at).total_seconds()
        if elapsed >= self._recovery_timeout:
            self._state = CircuitState.HALF_OPEN
            self._consecutive_successes = 0
            logger.info(
                "CircuitBreaker HALF_OPEN: source=%r (elapsed=%.1fs)",
                self.source_name,
                elapsed,
            )

    def reset(self) -> None:
        """Manually reset to CLOSED (useful for testing)."""
        with self._lock:
            self._close()


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

class CircuitBreakerRegistry:
    """
    Manages one CircuitBreaker per source.
    Ensures a failed source does not affect other sources.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> None:
        self._threshold = failure_threshold
        self._timeout = recovery_timeout
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, source_name: str) -> CircuitBreaker:
        """Return (creating if necessary) the CircuitBreaker for source_name."""
        with self._lock:
            if source_name not in self._breakers:
                self._breakers[source_name] = CircuitBreaker(
                    source_name,
                    failure_threshold=self._threshold,
                    recovery_timeout=self._timeout,
                )
            return self._breakers[source_name]

    def is_open(self, source_name: str) -> bool:
        return self.get(source_name).is_open

    def record_success(self, source_name: str) -> None:
        self.get(source_name).record_success()

    def record_failure(self, source_name: str, reason: str = "") -> None:
        self.get(source_name).record_failure(reason)

    def open_sources(self) -> list[str]:
        """Return list of source names with open circuits."""
        with self._lock:
            return [
                name for name, cb in self._breakers.items()
                if cb.state == CircuitState.OPEN
            ]

    def all_states(self) -> Dict[str, str]:
        with self._lock:
            return {name: cb.state.value for name, cb in self._breakers.items()}
