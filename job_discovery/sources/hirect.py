"""
Hirect source adapter — Tier 2.

Hirect is a direct-hire startup job board; availability varies.
"""
from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse, urlunparse

from job_discovery.sources.base import (
    JobSourceAdapter,
    SourceMeta,
    SourceTier,
    UnavailableError,
)

_META = SourceMeta(
    name="hirect",
    display_name="Hirect",
    domain="hirect.in",
    tier=SourceTier.TIER2,
    canonical_base_url="https://hirect.in/job/",
    access_notes="Tier 2: Startup-focused; availability varies.",
)

_ID_RE = re.compile(r"/job(?:s)?/([a-zA-Z0-9_-]+)", re.I)


class HirectAdapter(JobSourceAdapter):

    @property
    def meta(self) -> SourceMeta:
        return _META

    def recognises_url(self, url: str) -> bool:
        try:
            return "hirect.in" in urlparse(url).netloc.lower()
        except Exception:
            return False

    def canonical_url(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            clean = parsed._replace(scheme="https", netloc="hirect.in",
                                    query="", fragment="")
            return urlunparse(clean)
        except Exception:
            return url

    def extract_external_id(self, url: str) -> Optional[str]:
        m = _ID_RE.search(urlparse(url).path)
        return m.group(1) if m else None

    def fetch_job(self, url: str):  # type: ignore[override]
        raise UnavailableError(
            f"Hirect: automated retrieval not yet implemented; url={url!r}"
        )
