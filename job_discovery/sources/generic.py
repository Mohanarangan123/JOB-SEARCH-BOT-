"""
Generic source adapter — fallback for any URL not matched by a specific adapter.

Provides minimal URL handling. No external-ID extraction is attempted
beyond reading the full URL as-is.
"""
from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse, urlunparse

from job_discovery.sources.base import (
    JobSourceAdapter,
    SourceMeta,
    SourceTier,
    UnavailableError,
)

_META = SourceMeta(
    name="generic",
    display_name="Generic",
    domain="",
    tier=SourceTier.TIER2,
    canonical_base_url="",
    access_notes="Generic fallback adapter — no source-specific logic.",
)


class GenericAdapter(JobSourceAdapter):
    """
    Fallback adapter.
    Recognises any URL not claimed by a more specific adapter.
    Should always be registered last in the SourceRegistry.
    """

    @property
    def meta(self) -> SourceMeta:
        return _META

    def recognises_url(self, url: str) -> bool:
        # Matches everything — must be last in registry
        return bool(url)

    def canonical_url(self, url: str) -> str:
        """Strip query + fragment; normalise scheme to https."""
        try:
            parsed = urlparse(url)
            clean = parsed._replace(scheme="https", query="", fragment="")
            return urlunparse(clean)
        except Exception:
            return url

    def extract_external_id(self, url: str) -> Optional[str]:
        """Generic: use the full path as the ID proxy."""
        try:
            return urlparse(url).path.strip("/") or None
        except Exception:
            return None

    def fetch_job(self, url: str):  # type: ignore[override]
        raise UnavailableError(
            f"Generic adapter: automated retrieval not yet implemented; url={url!r}"
        )
