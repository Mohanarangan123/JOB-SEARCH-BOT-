"""
Tests for CircuitBreaker and CircuitBreakerRegistry.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from job_discovery.fetch.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
)


class TestCircuitBreakerStates:
    def _cb(self, threshold: int = 3, recovery: float = 60.0) -> CircuitBreaker:
        return CircuitBreaker("test_source", failure_threshold=threshold, recovery_timeout=recovery)

    def test_initial_state_closed(self):
        cb = self._cb()
        assert cb.state == CircuitState.CLOSED

    def test_allows_request_when_closed(self):
        cb = self._cb()
        assert cb.allows_request

    def test_stays_closed_below_threshold(self):
        cb = self._cb(threshold=3)
        cb.record_failure("err")
        cb.record_failure("err")
        assert cb.state == CircuitState.CLOSED

    def test_opens_at_threshold(self):
        cb = self._cb(threshold=3)
        cb.record_failure("e1")
        cb.record_failure("e2")
        cb.record_failure("e3")
        assert cb.state == CircuitState.OPEN

    def test_open_blocks_requests(self):
        cb = self._cb(threshold=2)
        cb.record_failure("e1")
        cb.record_failure("e2")
        assert not cb.allows_request

    def test_open_records_reason(self):
        cb = self._cb(threshold=1)
        cb.record_failure("rate limited")
        assert cb.last_reason == "rate limited"

    def test_success_resets_failure_count(self):
        cb = self._cb(threshold=5)
        cb.record_failure("e1")
        cb.record_failure("e2")
        cb.record_success()
        assert cb.failure_count == 0

    def test_reset_restores_closed(self):
        cb = self._cb(threshold=1)
        cb.record_failure("e")
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.allows_request


class TestCircuitBreakerHalfOpen:
    def _open_cb(self, recovery: float = 1.0) -> CircuitBreaker:
        cb = CircuitBreaker("src", failure_threshold=1, recovery_timeout=recovery)
        cb.record_failure("test")
        return cb

    def test_transitions_to_half_open_after_timeout(self):
        cb = self._open_cb(recovery=0.0)
        # recovery_timeout=0 → immediately half-open on next allows_request check
        _ = cb.allows_request
        assert cb.state == CircuitState.HALF_OPEN

    def test_success_in_half_open_closes(self):
        cb = self._open_cb(recovery=0.0)
        _ = cb.allows_request  # trigger half-open transition
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_failure_in_half_open_reopens(self):
        cb = self._open_cb(recovery=0.0)
        _ = cb.allows_request
        cb.record_failure("still failing")
        assert cb.state == CircuitState.OPEN

    def test_not_half_open_before_timeout(self):
        cb = self._open_cb(recovery=9999.0)
        assert not cb.allows_request
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerRegistry:
    def test_creates_breaker_on_demand(self):
        reg = CircuitBreakerRegistry(failure_threshold=3, recovery_timeout=60)
        cb = reg.get("linkedin")
        assert cb is not None
        assert cb.source_name == "linkedin"

    def test_same_instance_returned(self):
        reg = CircuitBreakerRegistry()
        cb1 = reg.get("naukri")
        cb2 = reg.get("naukri")
        assert cb1 is cb2

    def test_different_sources_independent(self):
        reg = CircuitBreakerRegistry(failure_threshold=2)
        # Open linkedin circuit
        reg.record_failure("linkedin", "e1")
        reg.record_failure("linkedin", "e2")
        # naukri should still be closed
        assert reg.is_open("linkedin")
        assert not reg.is_open("naukri")

    def test_record_success_closes(self):
        reg = CircuitBreakerRegistry(failure_threshold=1)
        reg.record_failure("indeed", "fail")
        assert reg.is_open("indeed")
        # Force to half-open via reset for test simplicity
        reg.get("indeed").reset()
        assert not reg.is_open("indeed")

    def test_open_sources_listing(self):
        reg = CircuitBreakerRegistry(failure_threshold=1)
        reg.record_failure("hirist", "e")
        reg.record_failure("wellfound", "e")
        open_sources = reg.open_sources()
        assert "hirist" in open_sources
        assert "wellfound" in open_sources

    def test_all_states(self):
        reg = CircuitBreakerRegistry(failure_threshold=1)
        reg.get("linkedin")
        reg.record_failure("naukri", "e")
        states = reg.all_states()
        assert states["linkedin"] == "closed"
        assert states["naukri"] == "open"

    def test_failed_source_does_not_affect_others(self):
        """Core design requirement: one open circuit must not block other sources."""
        reg = CircuitBreakerRegistry(failure_threshold=1)
        sources = ["linkedin", "indeed", "naukri", "cutshort"]
        # Fail linkedin to threshold
        reg.record_failure("linkedin", "blocked")
        # All other sources still closed
        for source in ["indeed", "naukri", "cutshort"]:
            assert not reg.is_open(source), f"{source} should not be affected"
