"""
Identity resolver — maps a raw URL to a NormalizedUrl record,
attaching source detection and canonical form.

This sits between the search result collector and the fetch pipeline:
  ProviderSearchResult.url
      → IdentityResolver.resolve()
      → NormalizedUrl(original, canonical, source_name)
      → FetchPipeline

No semantic deduplication — exact match only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from job_discovery.deduplication.exact_match import (
    ExactMatchDeduplicator,
    UrlValidationError,
    normalize_url,
    validate_url,
)
from job_discovery.search.providers.web_search import detect_source_from_url


@dataclass
class NormalizedUrl:
    """Result of identity resolution for a single URL."""
    original_url: str
    canonical_url: str
    source_name: str        # e.g. "linkedin", "naukri", "generic"
    is_duplicate: bool = False


class IdentityResolver:
    """
    Converts raw URLs from search results into NormalizedUrl records,
    performing source detection and deduplication.

    Uses ExactMatchDeduplicator internally.
    State persists across calls within a run; call reset() between runs.
    """

    def __init__(self, deduplicator: Optional[ExactMatchDeduplicator] = None) -> None:
        self._dedup = deduplicator or ExactMatchDeduplicator()

    def resolve(self, url: str) -> Optional[NormalizedUrl]:
        """
        Resolve one URL.

        Returns:
            NormalizedUrl — always returned even for duplicates
                           (check .is_duplicate to decide whether to fetch).
            None if the URL is structurally invalid.
        """
        try:
            canonical, is_dup = self._dedup.check(url)
        except UrlValidationError:
            return None
        source = detect_source_from_url(canonical)
        return NormalizedUrl(
            original_url=url,
            canonical_url=canonical,
            source_name=source,
            is_duplicate=is_dup,
        )

    def resolve_many(self, urls: List[str]) -> List[NormalizedUrl]:
        """
        Resolve a list of URLs.  Invalid URLs are silently skipped.
        """
        results = []
        for url in urls:
            nr = self.resolve(url)
            if nr is not None:
                results.append(nr)
        return results

    def reset(self) -> None:
        self._dedup.reset()
