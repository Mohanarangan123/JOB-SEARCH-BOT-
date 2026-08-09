"""
Exact-match URL deduplication and normalization.

Responsibilities:
  - Validate URL structure (scheme + netloc required).
  - Normalize to a canonical form (lowercase scheme/host, sorted params,
    remove fragments and known tracking params).
  - Detect obvious duplicates using the normalized URL as a set key.

No semantic or embedding-based deduplication — exact match only.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlparse,
    urlunparse,
)

# Tracking / noise query parameters to strip across all sources
_STRIP_PARAMS: Set[str] = {
    # Generic trackers
    "utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term",
    "utm_id", "gclid", "fbclid", "msclkid", "dclid", "twclid",
    # LinkedIn
    "refId", "trackingId", "trk", "trkCode", "original_referer",
    # Indeed
    "from", "vjs", "advn", "vjfrom", "rref", "tk", "acatk", "pub",
    # Naukri
    "sid",
    # Generic noise
    "ref", "source", "campaign", "medium",
}

# Schemes we accept as valid job page URLs
_VALID_SCHEMES = {"https", "http"}


class UrlValidationError(ValueError):
    """Raised when a URL cannot be validated or normalized."""


def validate_url(url: str) -> None:
    """
    Raise UrlValidationError if the URL is not suitable for fetching.
    Checks: non-empty, valid scheme, host present.
    """
    if not url or not url.strip():
        raise UrlValidationError("URL is empty")
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise UrlValidationError(f"URL parse error: {exc}") from exc
    if parsed.scheme.lower() not in _VALID_SCHEMES:
        raise UrlValidationError(
            f"Invalid scheme {parsed.scheme!r} — only http/https allowed"
        )
    if not parsed.netloc:
        raise UrlValidationError(f"URL has no host: {url!r}")


def normalize_url(url: str) -> str:
    """
    Return a stable, canonical form of the URL suitable for deduplication.

    Operations applied:
      1. Lowercase scheme and host.
      2. Remove default ports (80 for http, 443 for https).
      3. Strip known tracking query parameters.
      4. Sort remaining query parameters alphabetically.
      5. Remove the URL fragment.
      6. Strip trailing slash from path (except bare root '/').
    """
    try:
        parsed = urlparse(url.strip())
    except Exception as exc:
        raise UrlValidationError(f"Cannot normalise URL: {exc}") from exc

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Remove default ports
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    # Strip tracking params; sort remaining
    qs_pairs = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=False)
        if k not in _STRIP_PARAMS
    ]
    qs_pairs.sort(key=lambda kv: kv[0])
    clean_query = urlencode(qs_pairs)

    # Strip trailing slash unless path is '/'
    path = parsed.path
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    normalised = urlunparse((scheme, netloc, path, "", clean_query, ""))
    return normalised


class ExactMatchDeduplicator:
    """
    Stateful deduplicator that tracks seen canonical URLs within a run.

    Usage:
        dedup = ExactMatchDeduplicator()
        for url in candidates:
            canonical, is_dup = dedup.check(url)
            if not is_dup:
                process(canonical)
    """

    def __init__(self) -> None:
        self._seen: Set[str] = set()

    def check(self, url: str) -> Tuple[str, bool]:
        """
        Validate, normalize, and check for duplication.

        Returns:
            (canonical_url, is_duplicate)
        Raises:
            UrlValidationError if the URL is invalid.
        """
        validate_url(url)
        canonical = normalize_url(url)
        is_dup = canonical in self._seen
        if not is_dup:
            self._seen.add(canonical)
        return canonical, is_dup

    def seen_count(self) -> int:
        return len(self._seen)

    def reset(self) -> None:
        """Clear state — call between orchestrator runs."""
        self._seen.clear()

    def is_seen(self, url: str) -> bool:
        """Check without registering."""
        try:
            canonical = normalize_url(url)
        except UrlValidationError:
            return False
        return canonical in self._seen

    # ------------------------------------------------------------------ #
    # Batch helper
    # ------------------------------------------------------------------ #

    def filter_new(self, urls: List[str]) -> List[Tuple[str, str]]:
        """
        Filter a list of raw URLs, returning only new ones.

        Returns:
            List of (original_url, canonical_url) for URLs not yet seen.
        """
        result: List[Tuple[str, str]] = []
        for url in urls:
            try:
                canonical, is_dup = self.check(url)
                if not is_dup:
                    result.append((url, canonical))
            except UrlValidationError:
                pass  # silently skip invalid URLs
        return result
