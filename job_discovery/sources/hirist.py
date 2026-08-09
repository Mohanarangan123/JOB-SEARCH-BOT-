"""
Hirist source adapter — Tier 2.

Hirist is a tech-focused job board; pages are generally public.
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
    name="hirist",
    display_name="Hirist",
    domain="hirist.tech",
    tier=SourceTier.TIER2,
    canonical_base_url="https://www.hirist.tech/j/",
    access_notes="Tier 2: Public listings; availability varies.",
)

_ID_RE = re.compile(r"/j/[^/]*?-(\d+)", re.I)


class HiristAdapter(JobSourceAdapter):

    @property
    def meta(self) -> SourceMeta:
        return _META

    def recognises_url(self, url: str) -> bool:
        try:
            host = urlparse(url).netloc.lower()
            return "hirist.tech" in host or "hirist.com" in host
        except Exception:
            return False

    def canonical_url(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            clean = parsed._replace(scheme="https", netloc="www.hirist.tech",
                                    query="", fragment="")
            return urlunparse(clean)
        except Exception:
            return url

    def extract_external_id(self, url: str) -> Optional[str]:
        m = _ID_RE.search(urlparse(url).path)
        return m.group(1) if m else None

    def fetch_job(self, url: str):  # type: ignore[override]
        raise UnavailableError(
            f"Hirist: automated retrieval not yet implemented; url={url!r}"
        )
