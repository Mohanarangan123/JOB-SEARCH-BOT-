"""
Tests for RelevanceScorer.
No resume, no embeddings, no LLM — criteria-only ranking.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from job_discovery.ranking.relevance_scorer import (
    DEFAULT_WEIGHTS,
    RelevanceScorer,
    score_experience,
    score_freshness,
    score_location,
    score_title,
    score_workplace,
)
from job_discovery.search.query_builder import SearchCriteria


def _now(offset_days=0):
    return datetime.now(timezone.utc) + timedelta(days=offset_days)


def _job(
    title=None, location=None, work_mode=None,
    exp_min=None, exp_max=None, last_seen=None,
):
    return {
        "details": {
            "title": title,
            "location": location,
            "work_mode": work_mode,
            "requirements": {
                "experience_years_min": exp_min,
                "experience_years_max": exp_max,
            },
        },
        "last_seen_at": last_seen,
    }


def _criteria(
    title=None, location=None, remote_ok=False,
    workplace_type=None, exp_min=None, exp_max=None,
):
    return SearchCriteria(
        title=title,
        location=location,
        remote_ok=remote_ok,
        workplace_type=workplace_type,
        experience_years_min=exp_min,
        experience_years_max=exp_max,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Title scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestTitleScore:
    def test_exact_match(self):
        assert score_title("Python Developer", "Python Developer") == 1.0

    def test_partial_overlap(self):
        s = score_title("Senior Python Developer", "Python Developer")
        assert 0.0 < s <= 1.0

    def test_no_overlap(self):
        s = score_title("Java Engineer", "Python Developer")
        assert s == 0.0

    def test_none_criteria(self):
        assert score_title("Python Dev", None) == 0.0

    def test_none_job_title(self):
        assert score_title(None, "Python Dev") == 0.0

    def test_case_insensitive(self):
        s1 = score_title("python developer", "PYTHON DEVELOPER")
        assert s1 == 1.0

    def test_single_common_token(self):
        s = score_title("Backend Developer", "Python Developer")
        assert s > 0.0   # "developer" in common


# ─────────────────────────────────────────────────────────────────────────────
# Location scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestLocationScore:
    def test_exact_city_match(self):
        c = _criteria(location="Chennai")
        assert score_location("Chennai", None, c) == 1.0

    def test_same_state_different_city(self):
        c = _criteria(location="Chennai")
        assert score_location("Coimbatore", None, c) == 0.5

    def test_different_state(self):
        c = _criteria(location="Chennai")
        assert score_location("Mumbai", None, c) == 0.0

    def test_remote_ok_and_remote_job(self):
        c = _criteria(remote_ok=True)
        assert score_location(None, "remote", c) == 1.0

    def test_remote_workplace_type(self):
        c = _criteria(workplace_type="remote")
        assert score_location(None, "remote", c) == 1.0

    def test_no_criteria_location(self):
        c = _criteria()
        assert score_location("Chennai", None, c) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Experience scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestExperienceScore:
    def test_exact_overlap(self):
        c = _criteria(exp_min=3, exp_max=6)
        assert score_experience(3, 6, c) == 1.0

    def test_partial_overlap(self):
        c = _criteria(exp_min=3, exp_max=6)
        assert score_experience(5, 8, c) == 1.0  # overlap at 5-6

    def test_no_overlap_clearly_outside(self):
        c = _criteria(exp_min=3, exp_max=6)
        assert score_experience(10, 15, c) == 0.0

    def test_near_miss_partial(self):
        c = _criteria(exp_min=3, exp_max=5)
        # Job needs 6-8, criteria wants 3-5 → gap=1 → partial
        assert score_experience(6, 8, c) == 0.5

    def test_no_criteria_neutral(self):
        c = _criteria()
        assert score_experience(3, 6, c) == 0.5

    def test_no_job_experience_neutral(self):
        c = _criteria(exp_min=3, exp_max=6)
        assert score_experience(None, None, c) == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Workplace scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkplaceScore:
    def test_exact_match_remote(self):
        c = _criteria(workplace_type="remote")
        assert score_workplace("remote", c) == 1.0

    def test_exact_match_hybrid(self):
        c = _criteria(workplace_type="hybrid")
        assert score_workplace("hybrid", c) == 1.0

    def test_mismatch(self):
        c = _criteria(workplace_type="remote")
        assert score_workplace("onsite", c) == 0.0

    def test_unknown_job_wm_neutral(self):
        c = _criteria(workplace_type="remote")
        assert score_workplace(None, c) == 0.5

    def test_no_criteria_neutral(self):
        c = _criteria()
        assert score_workplace("remote", c) == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Freshness scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestFreshnessScore:
    def test_just_seen_near_1(self):
        s = score_freshness(_now())
        assert s > 0.99

    def test_old_job_low_score(self):
        s = score_freshness(_now(offset_days=-60))
        assert s < 0.5

    def test_very_old_job_zero(self):
        s = score_freshness(_now(offset_days=-100))
        assert s == 0.0

    def test_none_returns_zero(self):
        assert score_freshness(None) == 0.0

    def test_linear_decay(self):
        s1 = score_freshness(_now(offset_days=-10))
        s2 = score_freshness(_now(offset_days=-20))
        assert s1 > s2


# ─────────────────────────────────────────────────────────────────────────────
# Combined scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestRelevanceScorer:
    def setup_method(self):
        self.scorer = RelevanceScorer()

    def test_returns_relevance_score(self):
        job = _job("Python Developer", "Chennai", None, 3, 6, _now())
        c = _criteria("Python Developer", "Chennai")
        rs = self.scorer.score(job, c)
        assert 0.0 <= rs.total <= 1.0

    def test_component_scores_exposed(self):
        job = _job("Python Developer", "Chennai", "hybrid", 3, 6, _now())
        c = _criteria("Python Developer", "Chennai", workplace_type="hybrid")
        rs = self.scorer.score(job, c)
        assert hasattr(rs.components, "title")
        assert hasattr(rs.components, "location")
        assert hasattr(rs.components, "experience")
        assert hasattr(rs.components, "workplace")
        assert hasattr(rs.components, "freshness")

    def test_perfect_match_high_score(self):
        job = _job("Python Developer", "Chennai", "remote", 3, 6, _now())
        c = _criteria("Python Developer", "Chennai", remote_ok=True, exp_min=3, exp_max=6)
        rs = self.scorer.score(job, c)
        assert rs.total > 0.7

    def test_irrelevant_job_low_score(self):
        job = _job("Java Developer", "Mumbai", "onsite", 10, 15, _now(offset_days=-80))
        c = _criteria("Python Developer", "Chennai", workplace_type="remote", exp_min=0, exp_max=2)
        rs = self.scorer.score(job, c)
        assert rs.total < 0.5

    def test_formula_weights(self):
        """Verify the default weights sum to 1.0."""
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_custom_weights(self):
        scorer = RelevanceScorer(weights={"title": 1.0, "location": 0.0,
                                           "experience": 0.0, "workplace": 0.0,
                                           "freshness": 0.0})
        job = _job("Python Developer", "Chennai", None, None, None, None)
        c = _criteria("Python Developer")
        rs = scorer.score(job, c)
        assert rs.total > 0.5   # title matches perfectly

    def test_rank_returns_sorted_list(self):
        jobs = [
            _job("Java Developer", "Mumbai", None, None, None, _now(offset_days=-80)),
            _job("Python Developer", "Chennai", None, 3, 6, _now()),
            _job("Python Engineer", "Chennai", None, 2, 5, _now(offset_days=-5)),
        ]
        c = _criteria("Python Developer", "Chennai", exp_min=3, exp_max=6)
        ranked = self.scorer.rank(jobs, c)
        scores = [rs.total for _, rs in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_python_dev_highest(self):
        python_job = _job("Python Developer", "Chennai", None, 3, 6, _now())
        java_job   = _job("Java Developer", "Mumbai", None, None, None, _now(offset_days=-80))
        c = _criteria("Python Developer", "Chennai")
        ranked = self.scorer.rank([java_job, python_job], c)
        assert ranked[0][0] is python_job

    def test_no_resume_fields_read(self):
        """Scorer must not read resume/profile/user skills."""
        scorer = RelevanceScorer()
        assert not hasattr(scorer, "resume")
        assert not hasattr(scorer, "user_profile")
        assert not hasattr(scorer, "user_skills")
        assert not hasattr(scorer, "candidate_experience")

    def test_to_dict(self):
        job = _job("Python Developer", "Chennai", None, 3, 6, _now())
        c = _criteria("Python Developer", "Chennai")
        rs = self.scorer.score(job, c)
        d = rs.to_dict()
        assert "total" in d
        assert "components" in d
        assert "weights" in d
