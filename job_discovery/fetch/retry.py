"""
Retry configuration using Tenacity.

Provides:
  - build_retry_decorator()  — returns a tenacity @retry decorator configured
                               for HTTP fetching with exponential backoff + jitter.
  - is_retryable_status()    — decides if an HTTP status code warrants a retry.
  - RetryConfig              — typed dataclass for retry parameters.

Design rules:
  - Do NOT hand-roll retry loops; use Tenacity.
  - Only retry transient failures (network errors, 5xx, 429).
  - Do NOT retry permanent failures (400, 403, 404, 410).
  - Jitter prevents thundering-herd on rate-limit windows.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP status classification
# ─────────────────────────────────────────────────────────────────────────────

# Statuses that should be retried (transient / rate-limit)
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

# Statuses that are permanent failures — record and move on
_PERMANENT_FAILURE_STATUSES = {400, 401, 403, 404, 410}


def is_retryable_status(status_code: int) -> bool:
    """Return True if the HTTP status code warrants a retry attempt."""
    return status_code in _RETRYABLE_STATUSES


def is_permanent_failure(status_code: int) -> bool:
    """Return True if the HTTP status code is a permanent failure (do not retry)."""
    return status_code in _PERMANENT_FAILURE_STATUSES


# ─────────────────────────────────────────────────────────────────────────────
# Retry exceptions hierarchy
# ─────────────────────────────────────────────────────────────────────────────

class TransientFetchError(Exception):
    """
    Raised for errors that should be retried:
    network timeouts, connection resets, 5xx, 429.
    """
    def __init__(self, message: str, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


class PermanentFetchError(Exception):
    """
    Raised for errors that must NOT be retried:
    403, 404, 410, robots.txt disallowed.
    """
    def __init__(self, message: str, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


class RobotsDisallowedError(PermanentFetchError):
    """Raised when robots.txt disallows fetching the URL."""


class CircuitOpenError(Exception):
    """Raised when the circuit breaker for a source is open."""


# ─────────────────────────────────────────────────────────────────────────────
# Retry configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetryConfig:
    """Parameters for exponential-backoff retry with jitter."""
    max_attempts: int = 3        # total attempts (1 initial + N-1 retries)
    base_delay: float = 2.0      # initial wait in seconds
    max_delay: float = 30.0      # cap on wait time
    jitter: float = 1.0          # random jitter added to each wait


def _log_retry(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Retry attempt %d — %s",
        retry_state.attempt_number,
        exc,
    )


def build_retry_decorator(config: RetryConfig):
    """
    Return a Tenacity @retry decorator that:
      - Retries only on TransientFetchError.
      - Uses exponential back-off + uniform jitter.
      - Stops after config.max_attempts total attempts.
      - Logs each retry.
    """
    return retry(
        retry=retry_if_exception_type(TransientFetchError),
        stop=stop_after_attempt(config.max_attempts),
        wait=wait_exponential_jitter(
            initial=config.base_delay,
            max=config.max_delay,
            jitter=config.jitter,
        ),
        before_sleep=_log_retry,
        reraise=True,
    )
