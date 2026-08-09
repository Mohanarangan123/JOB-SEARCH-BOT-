"""
RelevanceScorer — ranks job records against SearchCriteria.

This is NOT resume matching. It never reads:
  - resume / user profile / user skills / candidate experience

Ranking uses only the job content and the search criteria.

Formula (configurable weights):
  title_match    * 0.40
  location_match * 0.20
  experience_match * 0.20
  workplace_match  * 0.10
  freshness        * 0.10

No LLM, no embeddings, no semantic similarity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from job_discovery.normalization.location import LocationNormalizer
from job_discovery.search.query_builder import SearchCriteria


# ─────────────────────────────────────────────────────────────────────────────
# Default weights
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS: Dict[str, float] = {
    "title":      0.40,
    "location":   0.20,
    "experience": 0.20,
    "workplace":  0.10,
    "freshness":  0.10,
}

# Freshness decay: jobs older than this get 0.0
_MAX_FRESHNESS_DAYS = 90


# ─────────────────────────────────────────────────────────────────────────────
# Score result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComponentScores:
    title:      float = 0.0
    location:   float = 0.0
    experience: float = 0.0
    workplace:  float = 0.0
    freshness:  float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "title":      round(self.title, 3),
            "location":   round(self.location, 3),
            "experience": round(self.experience, 3),
            "workplace":  round(self.workplace, 3),
            "freshness":  round(self.freshness, 3),
        }


@dataclass
class RelevanceScore:
    total:      float
    components: ComponentScores
    weights:    Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total":      round(self.total, 4),
            "components": self.components.to_dict(),
            "weights":    self.weights,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Title scoring — token overlap
# ─────────────────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> set:
    """Lowercase word tokens, 2+ chars."""
    return {w for w in re.findall(r"\b\w{2,}\b", text.lower())}


# Stop-words to ignore in title matching
_STOP = {"the", "and", "for", "with", "at", "to", "of", "in", "a", "an"}


def score_title(job_title: Optional[str], criteria_title: Optional[str]) -> float:
    """Token/substring overlap score for job title vs criteria title."""
    if not job_title or not criteria_title:
        return 0.0
    job_tokens      = _tokenise(job_title) - _STOP
    criteria_tokens = _tokenise(criteria_title) - _STOP
    if not criteria_tokens:
        return 0.0
    overlap = len(job_tokens & criteria_tokens)
    return min(1.0, overlap / len(criteria_tokens))


# ─────────────────────────────────────────────────────────────────────────────
# Location scoring
# ─────────────────────────────────────────────────────────────────────────────

_loc_norm = LocationNormalizer()


def score_location(
    job_location: Optional[str],
    job_work_mode: Optional[str],
    criteria: SearchCriteria,
) -> float:
    """
    1.0 — exact normalised city match OR remote+remote_ok
    0.5 — same state, different city
    0.0 — otherwise
    """
    job_wm = (job_work_mode or "").lower()

    # Remote: if criteria asks remote and job is remote → 1.0
    if criteria.remote_ok or criteria.workplace_type == "remote":
        if "remote" in job_wm or "wfh" in job_wm:
            return 1.0

    if not job_location or not criteria.location:
        return 0.0

    job_loc  = _loc_norm.normalize(job_location)
    crit_loc = _loc_norm.normalize(criteria.location)

    # Exact city match
    if job_loc.city and crit_loc.city:
        if job_loc.city.lower() == crit_loc.city.lower():
            return 1.0

    # Same state
    if job_loc.state and crit_loc.state:
        if job_loc.state.lower() == crit_loc.state.lower():
            return 0.5

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Experience scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_experience(
    job_exp_min: Optional[int],
    job_exp_max: Optional[int],
    criteria: SearchCriteria,
) -> float:
    """
    1.0 — criteria range overlaps job range
    partial — near overlap (within 1 year)
    0.0 — clearly outside
    """
    crit_min = criteria.experience_years_min
    crit_max = criteria.experience_years_max

    if crit_min is None and crit_max is None:
        return 0.5  # criteria has no preference → neutral

    if job_exp_min is None and job_exp_max is None:
        return 0.5  # job has no stated experience → neutral

    j_min = job_exp_min or 0
    j_max = job_exp_max or j_min + 5   # assume 5-year window if only min given
    c_min = crit_min or 0
    c_max = crit_max or c_min + 5

    # Overlap check
    overlap_start = max(j_min, c_min)
    overlap_end   = min(j_max, c_max)
    if overlap_end >= overlap_start:
        return 1.0

    # Near miss: gap ≤ 1 year
    gap = max(c_min - j_max, j_min - c_max)
    if gap <= 1:
        return 0.5

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Workplace scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_workplace(
    job_work_mode: Optional[str],
    criteria: SearchCriteria,
) -> float:
    """
    1.0 — exact match
    0.5 — unknown job work mode (can't rule it out)
    0.0 — known mismatch
    """
    crit_wm = criteria.workplace_type
    if not crit_wm:
        return 0.5   # no criteria preference → neutral

    job_wm = (job_work_mode or "").lower().strip()
    if not job_wm:
        return 0.5   # unknown

    crit_wm_l = crit_wm.lower()

    # Normalise remote variants
    if job_wm in ("remote", "wfh", "work from home"):
        job_wm = "remote"

    if job_wm == crit_wm_l:
        return 1.0
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Freshness scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_freshness(last_seen_at: Optional[datetime]) -> float:
    """
    Linear decay from 1.0 (just seen) to 0.0 (_MAX_FRESHNESS_DAYS ago).
    Uses last_seen_at (system timestamp, not posted_date).
    """
    if last_seen_at is None:
        return 0.0
    now = datetime.now(timezone.utc)
    # Ensure both are tz-aware
    if last_seen_at.tzinfo is None:
        last_seen_at = last_seen_at.replace(tzinfo=timezone.utc)
    age_days = (now - last_seen_at).total_seconds() / 86400
    score = max(0.0, 1.0 - age_days / _MAX_FRESHNESS_DAYS)
    return round(score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# RelevanceScorer
# ─────────────────────────────────────────────────────────────────────────────

class RelevanceScorer:
    """
    Scores a job record dict against SearchCriteria.

    Design constraints:
      - No resume, no user profile, no candidate skills.
      - No LLM, no embeddings, no vector similarity.
      - All logic is deterministic and reproducible.
      - Component scores are exposed for debugging.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        raw = weights or dict(DEFAULT_WEIGHTS)
        # Normalise weights to sum to 1.0
        total = sum(raw.values()) or 1.0
        self._weights = {k: v / total for k, v in raw.items()}

    def score(
        self,
        job: Dict[str, Any],
        criteria: SearchCriteria,
    ) -> RelevanceScore:
        """
        Score a job record dict against SearchCriteria.

        Args:
            job:      A dict with fields from JobRecord / JobDetails.
                      Expected keys (all optional): title, location,
                      work_mode, experience_years_min, experience_years_max,
                      last_seen_at.
            criteria: The user search criteria (no profile data).

        Returns:
            RelevanceScore with total and component breakdown.
        """
        # Extract job fields (handles both flat and nested dicts)
        details = job.get("details") or {}
        reqs    = details.get("requirements") or {}

        job_title    = details.get("title")   or job.get("title")
        job_location = details.get("location") or job.get("location")
        job_wm       = details.get("work_mode") or job.get("work_mode")
        job_exp_min  = reqs.get("experience_years_min") or job.get("experience_years_min")
        job_exp_max  = reqs.get("experience_years_max") or job.get("experience_years_max")
        last_seen    = job.get("last_seen_at")

        comp = ComponentScores(
            title      = score_title(job_title, criteria.title),
            location   = score_location(job_location, job_wm, criteria),
            experience = score_experience(job_exp_min, job_exp_max, criteria),
            workplace  = score_workplace(job_wm, criteria),
            freshness  = score_freshness(last_seen),
        )

        w = self._weights
        total = (
            comp.title      * w.get("title",      DEFAULT_WEIGHTS["title"])
            + comp.location   * w.get("location",   DEFAULT_WEIGHTS["location"])
            + comp.experience * w.get("experience", DEFAULT_WEIGHTS["experience"])
            + comp.workplace  * w.get("workplace",  DEFAULT_WEIGHTS["workplace"])
            + comp.freshness  * w.get("freshness",  DEFAULT_WEIGHTS["freshness"])
        )

        return RelevanceScore(
            total=round(total, 4),
            components=comp,
            weights=dict(self._weights),
        )

    def rank(
        self,
        jobs: List[Dict[str, Any]],
        criteria: SearchCriteria,
    ) -> List[tuple]:
        """
        Score and rank a list of job dicts.

        Returns list of (job_dict, RelevanceScore) sorted highest-first.
        """
        scored = [(job, self.score(job, criteria)) for job in jobs]
        scored.sort(key=lambda x: x[1].total, reverse=True)
        return scored
