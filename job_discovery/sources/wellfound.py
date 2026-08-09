"""
Wellfound (formerly AngelList Talent) source adapter — Tier 2.

Wellfound startup job listings; some may require sign-in.
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
    name="wellfound",
    display_name="Wellfound",
    domain="wellfound.com",
    tier=SourceTier.TIER2,
    canonical_base_url="https://wellfound.com/jobs/",
    access_notes=(
        "Tier 2: Formerly AngelList Talent. "
        "Some listings require authentication. "
        "Record UnavailableError when gated."
    ),
)

# /l/<slug>/<id> or /jobs/<slug>
_ID_RE = re.compile(r"/l/[^/]+/(\d+)|/jobs/(\d+)", re.I)


class WellfoundAdapter(JobSourceAdapter):

    @property
    def meta(self) -> SourceMeta:
        return _META

    def recognises_url(self, url: str) -> bool:
        try:
            host = urlparse(url).netloc.lower()
            return "wellfound.com" in host or "angel.co" in host
        except Exception:
            return False

    def canonical_url(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            clean = parsed._replace(scheme="https", netloc="wellfound.com",
                                    query="", fragment="")
            return urlunparse(clean)
        except Exception:
            return url

    def extract_external_id(self, url: str) -> Optional[str]:
        m = _ID_RE.search(urlparse(url).path)
        if m:
            return m.group(1) or m.group(2)
        return None

    def fetch_job(self, url: str):  # type: ignore[override]
        raise UnavailableError(
            f"Wellfound: automated retrieval not yet implemented; url={url!r}"
        )
