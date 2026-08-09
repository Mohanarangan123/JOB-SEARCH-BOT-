"""
Naukri source adapter — Tier 1.

Naukri job URLs use a slug + numeric ID pattern.
"""
from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse

from job_discovery.sources.base import (
    JobSourceAdapter,
    SourceMeta,
    SourceTier,
)

_META = SourceMeta(
    name="naukri",
    display_name="Naukri",
    domain="naukri.com",
    tier=SourceTier.TIER1,
    canonical_base_url="https://www.naukri.com/job-listings-",
    access_notes="Public listings; respect robots.txt and rate limits.",
)

# Naukri job IDs are numeric and appear at the end of the path or in query
_ID_FROM_PATH = re.compile(r"-(\d{6,})(?:\?|$|/)", re.I)
_ID_FROM_QUERY = re.compile(r"[?&]jid=(\d+)", re.I)


class NaukriAdapter(JobSourceAdapter):

    @property
    def meta(self) -> SourceMeta:
        return _META

    def recognises_url(self, url: str) -> bool:
        try:
            host = urlparse(url).netloc.lower()
        except Exception:
            return False
        return "naukri.com" in host

    def canonical_url(self, url: str) -> str:
        """
        Naukri canonical form: strip query parameters, keep the clean path.
        e.g. https://www.naukri.com/job-listings-python-developer-...-12345678
        """
        try:
            parsed = urlparse(url)
            clean = parsed._replace(
                scheme="https",
                netloc="www.naukri.com",
                query="",
                fragment="",
            )
            from urllib.parse import urlunparse
            return urlunparse(clean)
        except Exception:
            return url

    def extract_external_id(self, url: str) -> Optional[str]:
        """Extract Naukri's numeric job ID from URL path or query."""
        m = _ID_FROM_QUERY.search(url)
        if m:
            return m.group(1)
        m = _ID_FROM_PATH.search(url)
        if m:
            return m.group(1)
        return None
