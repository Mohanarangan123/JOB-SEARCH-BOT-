"""
Cutshort source adapter — Tier 2.

Cutshort is a startup-focused job board; some pages may be behind login.
Gracefully handles unavailable states.
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
    name="cutshort",
    display_name="Cutshort",
    domain="cutshort.io",
    tier=SourceTier.TIER2,
    canonical_base_url="https://cutshort.io/job/",
    access_notes=(
        "Tier 2: Some listings require login. "
        "Record UnavailableError when gated content is encountered."
    ),
)

# /job/<slug>-<id> or /jobs/<id>
_ID_RE = re.compile(r"/(?:job|jobs)/[^/]*?-?(\w{8,})", re.I)


class CutshortAdapter(JobSourceAdapter):

    @property
    def meta(self) -> SourceMeta:
        return _META

    def recognises_url(self, url: str) -> bool:
        try:
            return "cutshort.io" in urlparse(url).netloc.lower()
        except Exception:
            return False

    def canonical_url(self, url: str) -> str:
        """Strip query and fragment; normalise scheme."""
        try:
            parsed = urlparse(url)
            clean = parsed._replace(scheme="https", netloc="cutshort.io",
                                    query="", fragment="")
            return urlunparse(clean)
        except Exception:
            return url

    def extract_external_id(self, url: str) -> Optional[str]:
        m = _ID_RE.search(urlparse(url).path)
        return m.group(1) if m else None

    def fetch_job(self, url: str):  # type: ignore[override]
        # Tier 2: guard — fetch may be gated
        raise UnavailableError(
            f"Cutshort: automated retrieval not yet implemented; url={url!r}"
        )
