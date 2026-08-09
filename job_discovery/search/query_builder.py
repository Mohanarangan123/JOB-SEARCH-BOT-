"""
QueryBuilder — converts user-supplied search criteria into a set of
concrete query strings ready for a SearchProvider.

Design rules:
  - QueryBuilder does NOT call any external service.
  - Query expansion (LLM-assisted synonyms) is delegated to QueryExpander.
  - Source adapters must NEVER call QueryBuilder; only the orchestrator does.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Search Criteria
# ─────────────────────────────────────────────────────────────────────────────

class SearchCriteria(BaseModel):
    """
    User-supplied job search criteria.
    All fields are optional; at least title/keywords is strongly recommended.
    """
    # Core
    title: Optional[str] = None
    keywords: Optional[List[str]] = None

    # Experience
    experience_years_min: Optional[int] = None
    experience_years_max: Optional[int] = None

    # Location
    location: Optional[str] = None
    remote_ok: bool = False
    workplace_type: Optional[str] = None     # "remote" | "hybrid" | "onsite"

    # Employment
    employment_type: Optional[str] = None    # "full-time" | "part-time" | "contract"

    # Recency
    posting_age_days: Optional[int] = None   # only jobs posted within N days

    # Sources
    preferred_sources: Optional[List[str]] = None  # e.g. ["linkedin", "naukri"]

    model_config = {"populate_by_name": True}


# Supported source identifiers
ALL_SOURCES: List[str] = [
    "linkedin",
    "indeed",
    "naukri",
    "cutshort",
    "instahyre",
    "hirist",
    "wellfound",
    "hirect",
]

# site: domain map — used by WebSearchProvider for site-scoped queries
SOURCE_DOMAINS: Dict[str, str] = {
    "linkedin":  "linkedin.com/jobs",
    "indeed":    "indeed.com",
    "naukri":    "naukri.com",
    "cutshort":  "cutshort.io",
    "instahyre": "instahyre.com",
    "hirist":    "hirist.tech",
    "wellfound": "wellfound.com",
    "hirect":    "hirect.in",
}

# Title synonyms used for lightweight local expansion
_TITLE_SYNONYMS: Dict[str, List[str]] = {
    "developer":  ["developer", "engineer", "programmer"],
    "engineer":   ["engineer", "developer"],
    "scientist":  ["scientist", "analyst", "researcher"],
    "analyst":    ["analyst", "specialist"],
    "manager":    ["manager", "lead", "head"],
    "designer":   ["designer", "ux", "ui"],
    "devops":     ["devops", "sre", "platform engineer"],
    "qa":         ["qa", "quality assurance", "test engineer", "sdet"],
    "fullstack":  ["fullstack", "full stack", "full-stack"],
    "frontend":   ["frontend", "front-end", "ui developer"],
    "backend":    ["backend", "back-end", "server-side"],
    "data":       ["data", "analytics"],
    "ml":         ["ml", "machine learning", "ai"],
}


def _clean(text: str) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", text).strip()


def _extract_tech(title: str) -> Optional[str]:
    """
    Try to extract a leading technology token from a job title.
    E.g.  'Python Backend Developer' -> 'Python'
          'React Frontend Engineer' -> 'React'
    """
    tech_tokens = re.match(
        r"^(python|java|javascript|typescript|react|angular|vue|node|go|rust|"
        r"ruby|scala|kotlin|swift|c\+\+|c#|dotnet|\.net|php|aws|azure|gcp|"
        r"devops|sre|data|ml|ai|android|ios|flutter|django|spring|fastapi|rails)",
        title.lower(),
    )
    return tech_tokens.group(1).title() if tech_tokens else None


class QueryBuilder:
    """
    Builds a ranked list of query strings from SearchCriteria.

    Strategy:
      1. Derive a 'core' phrase from title + location.
      2. Generate role-synonym variants (e.g. developer / engineer / programmer).
      3. Optionally append workplace / employment type qualifiers.
      4. Return 3–5 deduplicated variants, most specific first.
    """

    def build(self, criteria: SearchCriteria) -> List[str]:
        """
        Return 3–5 concrete query strings from the given criteria.
        The list is ordered: most specific first, broadest last.
        """
        base_title = _clean(criteria.title or "")
        if not base_title and criteria.keywords:
            base_title = _clean(" ".join(criteria.keywords[:3]))
        if not base_title:
            base_title = "software engineer"

        location = _clean(criteria.location or "")
        tech = _extract_tech(base_title)

        # ── Role synonyms ────────────────────────────────────────────────
        role_lower = base_title.lower()
        role_variants: List[str] = [base_title]

        for trigger, synonyms in _TITLE_SYNONYMS.items():
            if trigger in role_lower:
                for synonym in synonyms:
                    variant = _clean(re.sub(trigger, synonym, role_lower, flags=re.I))
                    variant = variant.title()
                    if variant.lower() != base_title.lower():
                        role_variants.append(variant)
                break  # only first matching trigger

        # Deduplicate while preserving order
        seen: set = set()
        unique_roles: List[str] = []
        for r in role_variants:
            key = r.lower()
            if key not in seen:
                seen.add(key)
                unique_roles.append(r)

        # ── Qualifiers ───────────────────────────────────────────────────
        qualifiers: List[str] = []
        if criteria.workplace_type == "remote":
            qualifiers.append("remote")
        elif criteria.remote_ok:
            qualifiers.append("remote")
        if criteria.employment_type == "contract":
            qualifiers.append("contract")

        # ── Assemble queries ─────────────────────────────────────────────
        queries: List[str] = []

        def _make(role: str, loc: str, extra: str = "") -> str:
            parts = [p for p in [role, extra, loc] if p]
            return _clean(" ".join(parts))

        # 1. Exact title + location (most specific)
        if location:
            queries.append(_make(base_title, location))

        # 2. Variants with location
        for role in unique_roles[1:3]:
            if location:
                queries.append(_make(role, location))

        # 3. Tech-prefixed variant (only when tech is different from what's already in the title)
        if tech and location:
            tech_lower = tech.lower()
            # Only add if the tech word isn't already the first word of the title
            if not base_title.lower().startswith(tech_lower):
                tech_variant = f"{tech} {base_title}"
                candidate = _make(tech_variant, location)
                if candidate.lower() not in {q.lower() for q in queries}:
                    queries.append(candidate)

        # 4. With qualifier (e.g. "remote")
        for qual in qualifiers:
            candidate = _make(base_title, location, qual)
            if candidate.lower() not in {q.lower() for q in queries}:
                queries.append(candidate)

        # 5. Bare title (broadest fallback, always include)
        bare = _make(base_title, "")
        if bare.lower() not in {q.lower() for q in queries}:
            queries.append(bare)

        # Cap at 5, ensure at least 1
        return queries[:5] if queries else [_make(base_title, location)]

    def get_target_sources(self, criteria: SearchCriteria) -> List[str]:
        """
        Return the ordered list of source names to search.
        Respects preferred_sources if set, otherwise returns all sources.
        """
        if criteria.preferred_sources:
            # Validate against known sources; keep order
            valid = [s for s in criteria.preferred_sources if s in ALL_SOURCES]
            return valid if valid else ALL_SOURCES[:]
        return ALL_SOURCES[:]
