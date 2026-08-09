"""
LinkedIn source adapter.

LEGAL / RISK NOTICE:
  LinkedIn's User Agreement (Section 8.2) and robots.txt restrict automated
  access to its website.  While search-engine-indexed job URLs can be
  *discovered* via site:linkedin.com/jobs queries, automated *retrieval* of
  those pages may violate LinkedIn's ToS.  This adapter provides URL
  recognition and canonical normalisation only.  Automated page fetching
  against LinkedIn should be gated behind an explicit compliance review
  before enabling in production.

Tier 1: URLs are well-structured, job IDs are numeric, canonical form is stable.
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

_ACCESS_NOTES = (
    "LinkedIn ToS Section 8.2 restricts automated scraping. "
    "Site-scoped search discovery is low-risk; automated page retrieval "
    "requires a compliance review before enabling."
)

_META = SourceMeta(
    name="linkedin",
    display_name="LinkedIn",
    domain="linkedin.com",
    tier=SourceTier.TIER1,
    canonical_base_url="https://www.linkedin.com/jobs/view/",
    access_notes=_ACCESS_NOTES,
)

# Patterns that identify a LinkedIn job listing URL
_JOB_URL_PATTERNS = [
    re.compile(r"linkedin\.com/jobs/view/(\d+)", re.I),
    re.compile(r"linkedin\.com/jobs/collections/[^/]+/\?.*currentJobId=(\d+)", re.I),
    re.compile(r"linkedin\.com/job/[^/]+-(\d+)", re.I),
]

# Parameters to strip from LinkedIn URLs
_STRIP_PARAMS = {"refId", "trackingId", "trk", "position", "pageNum", "utm_source",
                 "utm_medium", "utm_campaign", "original_referer"}


class LinkedInAdapter(JobSourceAdapter):

    @property
    def meta(self) -> SourceMeta:
        return _META

    def recognises_url(self, url: str) -> bool:
        try:
            host = urlparse(url).netloc.lower()
        except Exception:
            return False
        return "linkedin.com" in host

    def canonical_url(self, url: str) -> str:
        """
        Produce canonical LinkedIn job URL:
          https://www.linkedin.com/jobs/view/<job_id>/
        Falls back to a cleaned URL if no job ID is found.
        """
        job_id = self.extract_external_id(url)
        if job_id:
            return f"https://www.linkedin.com/jobs/view/{job_id}/"

        # Strip tracking params and normalise scheme/host
        parsed = urlparse(url)
        qs = {k: v for k, v in parse_qs(parsed.query).items()
              if k not in _STRIP_PARAMS}
        clean_query = urlencode(qs, doseq=True)
        normalised = parsed._replace(
            scheme="https",
            netloc="www.linkedin.com",
            query=clean_query,
            fragment="",
        )
        return urlunparse(normalised)

    def extract_external_id(self, url: str) -> Optional[str]:
        """Extract the numeric LinkedIn job ID from a URL."""
        # Try query param first (currentJobId, jobId)
        try:
            qs = parse_qs(urlparse(url).query)
            for param in ("currentJobId", "jobId", "f_JT"):
                if param in qs:
                    val = qs[param][0]
                    if val.isdigit():
                        return val
        except Exception:
            pass

        # Try path patterns
        for pattern in _JOB_URL_PATTERNS:
            m = pattern.search(url)
            if m:
                return m.group(1)

        return None
