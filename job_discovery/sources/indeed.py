"""
Indeed source adapter — Tier 1.

Indeed has publicly accessible job pages; no authentication is required for
most listings. Respects robots.txt and rate limits.
"""
from __future__ import annotations

import re
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from job_discovery.sources.base import (
    JobSourceAdapter,
    SourceMeta,
    SourceTier,
)

_META = SourceMeta(
    name="indeed",
    display_name="Indeed",
    domain="indeed.com",
    tier=SourceTier.TIER1,
    canonical_base_url="https://www.indeed.com/viewjob?jk=",
    access_notes="Public job listings; respect robots.txt and rate limits.",
)

# Strip noisy tracking params
_STRIP_PARAMS = {"from", "vjs", "advn", "vjfrom", "rref", "tk", "acatk",
                 "pub", "utm_source", "utm_medium", "utm_campaign"}

# Job key is the `jk` query parameter (alphanumeric 16 chars)
_JK_RE = re.compile(r"\bjk=([a-f0-9]+)", re.I)


class IndeedAdapter(JobSourceAdapter):

    @property
    def meta(self) -> SourceMeta:
        return _META

    def recognises_url(self, url: str) -> bool:
        try:
            host = urlparse(url).netloc.lower()
        except Exception:
            return False
        return "indeed.com" in host

    def canonical_url(self, url: str) -> str:
        """Canonical form: https://www.indeed.com/viewjob?jk=<jk>"""
        jk = self.extract_external_id(url)
        if jk:
            return f"https://www.indeed.com/viewjob?jk={jk}"

        parsed = urlparse(url)
        qs = {k: v for k, v in parse_qs(parsed.query).items()
              if k not in _STRIP_PARAMS}
        clean = parsed._replace(
            scheme="https",
            netloc="www.indeed.com",
            query=urlencode(qs, doseq=True),
            fragment="",
        )
        return urlunparse(clean)

    def extract_external_id(self, url: str) -> Optional[str]:
        """Extract the `jk` job-key parameter."""
        try:
            qs = parse_qs(urlparse(url).query)
            if "jk" in qs:
                return qs["jk"][0]
        except Exception:
            pass
        m = _JK_RE.search(url)
        return m.group(1) if m else None
