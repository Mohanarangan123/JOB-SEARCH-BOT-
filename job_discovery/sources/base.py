"""
JobSourceAdapter — abstract base class for all source adapters.

Each adapter is responsible for:
  1. URL recognition    — does this URL belong to this source?
  2. Canonical URL      — normalise a URL to its stable canonical form.
  3. External job ID    — extract the source-specific job identifier.
  4. Source metadata    — name, domain, tier.

The following phases belong to LATER prompts and are declared here as
abstract stubs only:
  - fetch_job()    — Prompt 3: PageFetcher
  - extract()      — Prompt 4: ContentExtractor / LLMInterpreter
  - normalize()    — Prompt 4: Normalisation pipeline

Adapters must NEVER construct search queries (SearchProvider owns that).

Legal / access control note:
  - Adapters must respect robots.txt, rate limits, and ToS for each source.
  - CAPTCHA, authentication bypass, and anti-bot evasion are NOT permitted.
  - LinkedIn in particular: URL recognition is safe; automated page retrieval
    may violate their ToS — document and gate accordingly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SourceTier(str, Enum):
    TIER1 = "tier1"   # LinkedIn, Indeed, Naukri — high-availability, well-structured
    TIER2 = "tier2"   # Cutshort, Instahyre, Hirist, Wellfound, Hirect — may be unavailable


@dataclass(frozen=True)
class SourceMeta:
    """Static metadata about a job source."""
    name: str                   # e.g. "linkedin"
    display_name: str           # e.g. "LinkedIn"
    domain: str                 # e.g. "linkedin.com"
    tier: SourceTier
    canonical_base_url: str     # e.g. "https://www.linkedin.com/jobs/view/"
    # Legal / access notes shown in health reports
    access_notes: str = ""


class UnavailableError(Exception):
    """
    Raised by Tier 2 adapters (and Tier 1 in edge cases) when the source
    is inaccessible: blocked, rate-limited, CAPTCHA gated, or returned
    no results.  The orchestrator records this and moves on.
    """


class JobSourceAdapter(ABC):
    """
    Common interface for all source adapters.

    Only URL-handling methods are fully implemented in this prompt.
    fetch_job / extract / normalize are stubs for future prompts.
    """

    # ------------------------------------------------------------------ #
    # Metadata (must be overridden)
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def meta(self) -> SourceMeta:
        """Return the SourceMeta descriptor for this source."""

    # ------------------------------------------------------------------ #
    # URL handling (Phase 4 — implemented now)
    # ------------------------------------------------------------------ #

    @abstractmethod
    def recognises_url(self, url: str) -> bool:
        """
        Return True if this adapter handles the given URL.
        Used by the source router to dispatch URLs to the right adapter.
        """

    @abstractmethod
    def canonical_url(self, url: str) -> str:
        """
        Return the canonical (stable, de-parameterised) form of a URL.
        For example, strip tracking parameters but keep the job ID.
        Should be idempotent.
        """

    @abstractmethod
    def extract_external_id(self, url: str) -> Optional[str]:
        """
        Extract the source-specific job identifier from a URL.
        Returns None if the ID cannot be determined.
        """

    # ------------------------------------------------------------------ #
    # Fetch stub (Prompt 3)
    # ------------------------------------------------------------------ #

    def fetch_job(self, url: str) -> Dict[str, Any]:
        """
        Fetch raw page content for the given job URL.
        Fully implemented in Prompt 3 via PageFetcher.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.fetch_job() is not yet implemented"
        )

    # ------------------------------------------------------------------ #
    # Extraction stub (Prompt 4)
    # ------------------------------------------------------------------ #

    def extract(self, raw_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured job fields from raw page content.
        Fully implemented in Prompt 4 via ContentExtractor / LLMInterpreter.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.extract() is not yet implemented"
        )

    # ------------------------------------------------------------------ #
    # Normalisation stub (Prompt 4)
    # ------------------------------------------------------------------ #

    def normalize(self, extracted: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalise extracted fields (location, salary, skills, etc.).
        Fully implemented in Prompt 4 via the normalisation pipeline.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.normalize() is not yet implemented"
        )

    # ------------------------------------------------------------------ #
    # Convenience helpers (concrete)
    # ------------------------------------------------------------------ #

    @property
    def source_name(self) -> str:
        return self.meta.name

    @property
    def tier(self) -> SourceTier:
        return self.meta.tier

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} source={self.meta.name!r}>"


# ─────────────────────────────────────────────────────────────────────────────
# Source Registry
# ─────────────────────────────────────────────────────────────────────────────

class SourceRegistry:
    """
    Maintains a list of registered adapters and routes URLs to the
    correct adapter via recognises_url().
    """

    def __init__(self, adapters: Optional[List[JobSourceAdapter]] = None) -> None:
        self._adapters: List[JobSourceAdapter] = list(adapters or [])

    def register(self, adapter: JobSourceAdapter) -> None:
        self._adapters.append(adapter)

    def route(self, url: str) -> Optional[JobSourceAdapter]:
        """Return the first adapter that recognises the URL, or None."""
        for adapter in self._adapters:
            if adapter.recognises_url(url):
                return adapter
        return None

    def all_adapters(self) -> List[JobSourceAdapter]:
        return list(self._adapters)

    def get_by_name(self, name: str) -> Optional[JobSourceAdapter]:
        for adapter in self._adapters:
            if adapter.source_name == name:
                return adapter
        return None
