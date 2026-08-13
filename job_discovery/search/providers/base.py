"""
Abstract base class for all search providers.

Design constraint: SearchProvider OWNS discovery.
Source adapters must NEVER construct search queries.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# SearchResult data-transfer object
# ─────────────────────────────────────────────────────────────────────────────

class ProviderSearchResult(BaseModel):
    """
    A single URL/snippet returned by a SearchProvider.
    This is the *provider-level* DTO — distinct from the MongoDB-backed
    SearchResult model used by the repository layer.
    """
    title: Optional[str] = None
    url: str
    snippet: Optional[str] = None
    source_name: str          # e.g. "linkedin", "naukri", "generic"
    query: str                # the raw query string that produced this result
    query_hash: str           # SHA-256(normalised query)
    rank: Optional[int] = None
    discovered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = {"populate_by_name": True}

    @staticmethod
    def make_query_hash(query: str) -> str:
        """Stable, normalised SHA-256 hash of a query string."""
        normalised = " ".join(query.lower().split())
        return hashlib.sha256(normalised.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Abstract SearchProvider
# ─────────────────────────────────────────────────────────────────────────────

class SearchProvider(ABC):
    """
    Interface every search provider must implement.

    A provider takes a plain query string and returns ranked
    ProviderSearchResult objects.  The provider is responsible for:
      - site:-scoping queries to the desired sources
      - rate-limit awareness (raising RateLimitedError when appropriate)
      - returning an empty list (not raising) when no results are found

    Providers must NOT:
      - construct or expand queries (that is QueryBuilder / QueryExpander's job)
      - parse job page content (that is the extraction pipeline's job)
    """

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        source_filter: Optional[List[str]] = None,
    ) -> List[ProviderSearchResult]:
        """
        Execute a search and return results.

        Args:
            query: Plain-text query string (already expanded/built).
            max_results: Maximum number of results to return.
            source_filter: Optional list of source names to restrict results to.

        Returns:
            List of ProviderSearchResult, ordered by relevance/rank.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique identifier for this provider (e.g. 'ddg', 'google', 'mock')."""


class SearchProviderError(Exception):
    """Generic search provider failure with provider-level classification."""

    def __init__(
        self,
        message: str = "",
        *,
        classification: str = "SEARCH_HTTP_ERROR",
        provider: str = "web_search",
        source: Optional[str] = None,
        query: Optional[str] = None,
        http_status: Optional[int] = None,
        reason: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        super().__init__(message)
        self.classification = classification
        self.provider = provider
        self.source = source
        self.query = query
        self.http_status = http_status
        self.reason = reason or message
        self.timestamp = timestamp or datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "source": self.source,
            "query": self.query,
            "http_status": self.http_status,
            "classification": self.classification,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


class RateLimitedError(SearchProviderError):
    """Raised when the search provider signals rate-limiting."""

    def __init__(
        self,
        message: str = "",
        *,
        provider: str = "web_search",
        source: Optional[str] = None,
        query: Optional[str] = None,
        http_status: Optional[int] = None,
        reason: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        super().__init__(
            message,
            classification="SEARCH_RATE_LIMITED",
            provider=provider,
            source=source,
            query=query,
            http_status=http_status,
            reason=reason or message,
            timestamp=timestamp,
        )
