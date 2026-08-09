"""
FetchPolicy — governs whether and how a URL should be fetched.

Checks (in order):
  1. Circuit breaker for the source.
  2. robots.txt compliance.
  3. Request budget (max_fetches_per_run).

Returns a FetchDecision explaining the outcome.

Design rules:
  - No CAPTCHA bypass, no proxy rotation for evasion.
  - Robots.txt is always checked for sources that have a domain.
  - The policy does NOT make HTTP calls itself; PageFetcher does.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from job_discovery.fetch.circuit_breaker import CircuitBreakerRegistry
from job_discovery.fetch.retry import CircuitOpenError, RobotsDisallowedError

logger = logging.getLogger(__name__)

# User-agent string sent in the robots.txt request and in fetch headers.
# We identify ourselves honestly — no stealth / spoofed UA.
FETCH_USER_AGENT = "JobDiscoveryBot/1.0 (+https://github.com/job-discovery)"


class FetchDecision(str, Enum):
    ALLOW = "allow"
    BLOCK_CIRCUIT_OPEN = "block_circuit_open"
    BLOCK_ROBOTS = "block_robots"
    BLOCK_BUDGET = "block_budget"
    BLOCK_INVALID = "block_invalid"


@dataclass
class PolicyResult:
    """Outcome of a FetchPolicy check."""
    decision: FetchDecision
    reason: str = ""


class RobotsCache:
    """
    Simple in-process cache for parsed RobotFileParser objects.
    One entry per scheme+netloc.  Thread-unsafe (single-threaded orchestrator).
    """

    def __init__(self, http_client=None) -> None:
        self._cache: dict[str, RobotFileParser] = {}
        self._http = http_client  # optional; if None, fetches via urllib

    def is_allowed(self, url: str, user_agent: str = FETCH_USER_AGENT) -> bool:
        """
        Return True if robots.txt permits fetching the given URL.
        Returns True (permit) on any error fetching/parsing robots.txt,
        to avoid blocking valid pages due to robots.txt unavailability.
        """
        try:
            parsed = urlparse(url)
            key = f"{parsed.scheme}://{parsed.netloc}"
            if key not in self._cache:
                fetched_ok, rp = self._fetch_robots(key)
                self._cache[key] = (fetched_ok, rp)
            fetched_ok, rp = self._cache[key]
            if not fetched_ok:
                return True   # fail open — couldn't load robots.txt
            return rp.can_fetch(user_agent, url)
        except Exception as exc:
            logger.debug("robots.txt check failed for %r: %s — allowing", url, exc)
            return True  # fail open

    def _fetch_robots(self, base_url: str):
        robots_url = f"{base_url}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            if self._http is not None:
                resp = self._http.get(robots_url, timeout=10)
                rp.parse(resp.text.splitlines())
                return True, rp
            else:
                # No HTTP client provided — skip robots.txt fetch, fail open
                return False, rp
        except Exception as exc:
            logger.debug("Could not read robots.txt from %r: %s", robots_url, exc)
            return False, rp  # signal fetch failure

    def clear(self) -> None:
        self._cache.clear()


class FetchPolicy:
    """
    Evaluates whether a URL may be fetched given the current system state.

    Args:
        circuit_registry:  CircuitBreakerRegistry to check per-source state.
        robots_cache:      RobotsCache for robots.txt compliance.
        max_fetches:       Budget cap for this run (0 = unlimited).
    """

    def __init__(
        self,
        circuit_registry: CircuitBreakerRegistry,
        robots_cache: RobotsCache,
        *,
        max_fetches: int = 0,
    ) -> None:
        self._circuits = circuit_registry
        self._robots = robots_cache
        self._max = max_fetches
        self._fetched: int = 0

    def check(self, url: str, source_name: str) -> PolicyResult:
        """
        Evaluate the fetch policy for (url, source_name).

        Returns PolicyResult with decision + reason.
        Does NOT raise; callers check PolicyResult.decision.
        """
        # 1. Circuit breaker
        if self._circuits.is_open(source_name):
            reason = (
                f"Circuit open for source={source_name!r}: "
                f"{self._circuits.get(source_name).last_reason}"
            )
            return PolicyResult(FetchDecision.BLOCK_CIRCUIT_OPEN, reason)

        # 2. robots.txt
        if not self._robots.is_allowed(url):
            return PolicyResult(
                FetchDecision.BLOCK_ROBOTS,
                f"robots.txt disallows {url!r}",
            )

        # 3. Budget
        if self._max > 0 and self._fetched >= self._max:
            return PolicyResult(
                FetchDecision.BLOCK_BUDGET,
                f"Fetch budget exhausted ({self._max} fetches)",
            )

        return PolicyResult(FetchDecision.ALLOW, "")

    def consume(self) -> None:
        """Call once per successful fetch to decrement budget."""
        self._fetched += 1

    @property
    def fetches_remaining(self) -> int:
        if self._max == 0:
            return -1  # unlimited
        return max(0, self._max - self._fetched)
