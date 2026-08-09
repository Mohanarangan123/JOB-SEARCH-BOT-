"""
Instahyre source adapter — Tier 2.

Instahyre may require authentication for full job detail pages.
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
    name="instahyre",
    display_name="Instahyre",
    domain="instahyre.com",
    tier=SourceTier.TIER2,
    canonical_base_url="https://www.instahyre.com/job-",
    access_notes=(
        "Tier 2: Full job details may require authentication. "
        "Record UnavailableError when content is gated."
    ),
)

_ID_RE = re.compile(r"/job-(\d+)", re.I)


class InstahyreAdapter(JobSourceAdapter):

    @property
    def meta(self) -> SourceMeta:
        return _META

    def recognises_url(self, url: str) -> bool:
        try:
            return "instahyre.com" in urlparse(url).netloc.lower()
        except Exception:
            return False

    def canonical_url(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            clean = parsed._replace(scheme="https", netloc="www.instahyre.com",
                                    query="", fragment="")
            return urlunparse(clean)
        except Exception:
            return url

    def extract_external_id(self, url: str) -> Optional[str]:
        m = _ID_RE.search(urlparse(url).path)
        return m.group(1) if m else None

    def fetch_job(self, url: str):  # type: ignore[override]
        raise UnavailableError(
            f"Instahyre: automated retrieval not yet implemented; url={url!r}"
        )
