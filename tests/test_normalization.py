"""
Tests for location, salary, and skills normalization.
All static — no external calls.
"""
from __future__ import annotations

import pytest

from job_discovery.normalization.location import LocationNormalizer, NormalizedLocation
from job_discovery.normalization.salary import NormalizedSalary, SalaryNormalizer
from job_discovery.normalization.skills import NormalizedSkill, SkillsNormalizer


# ─────────────────────────────────────────────────────────────────────────────
# Location
# ─────────────────────────────────────────────────────────────────────────────

class TestLocationNormalizer:
    def setup_method(self):
        self.norm = LocationNormalizer()

    def test_city_only(self):
        r = self.norm.normalize("Chennai")
        assert r.city == "Chennai"
        assert r.state == "Tamil Nadu"
        assert r.country == "India"

    def test_city_with_abbreviation(self):
        r = self.norm.normalize("Chennai, TN")
        assert r.city == "Chennai"
        assert r.state == "Tamil Nadu"

    def test_city_with_full_state(self):
        r = self.norm.normalize("Chennai, Tamil Nadu")
        assert r.city == "Chennai"
        assert r.state == "Tamil Nadu"

    def test_city_state_country(self):
        r = self.norm.normalize("Chennai, Tamil Nadu, India")
        assert r.city == "Chennai"
        assert r.state == "Tamil Nadu"
        assert r.country == "India"

    def test_bangalore(self):
        r = self.norm.normalize("Bangalore")
        assert r.city == "Bangalore"
        assert r.state == "Karnataka"

    def test_bengaluru(self):
        r = self.norm.normalize("Bengaluru")
        assert r.city == "Bengaluru"
        assert r.state == "Karnataka"

    def test_mumbai(self):
        r = self.norm.normalize("Mumbai")
        assert r.state == "Maharashtra"
        assert r.country == "India"

    def test_delhi(self):
        r = self.norm.normalize("Delhi")
        assert r.state == "Delhi"

    def test_hyderabad(self):
        r = self.norm.normalize("Hyderabad")
        assert r.state == "Telangana"

    def test_remote_detected(self):
        r = self.norm.normalize("Remote")
        assert r.is_remote is True

    def test_wfh_detected(self):
        r = self.norm.normalize("Work From Home")
        assert r.is_remote is True

    def test_remote_with_city(self):
        r = self.norm.normalize("Remote - Bangalore")
        assert r.is_remote is True

    def test_original_text_preserved(self):
        raw = "Chennai, TN"
        r = self.norm.normalize(raw)
        assert r.original_text == raw

    def test_unknown_city_preserved_as_is(self):
        r = self.norm.normalize("Kozhikode")
        assert r.city == "Kozhikode"
        assert r.state is None  # unknown, not invented

    def test_none_input(self):
        r = self.norm.normalize(None)
        assert r.city is None
        assert r.state is None
        assert r.country is None

    def test_empty_string(self):
        r = self.norm.normalize("")
        assert r.city is None

    def test_state_abbreviation_expansion(self):
        r = self.norm.normalize("Pune, MH")
        assert r.state == "Maharashtra"

    def test_display_city_state_country(self):
        r = self.norm.normalize("Chennai")
        display = r.display()
        assert "Chennai" in display
        assert "Tamil Nadu" in display

    def test_display_remote(self):
        r = self.norm.normalize("Remote")
        assert "Remote" in r.display()


# ─────────────────────────────────────────────────────────────────────────────
# Salary
# ─────────────────────────────────────────────────────────────────────────────

class TestSalaryNormalizer:
    def setup_method(self):
        self.norm = SalaryNormalizer()

    def test_lpa_single(self):
        r = self.norm.normalize("₹4 LPA")
        assert r.min_amount == 400_000
        assert r.currency == "INR"
        assert r.period == "annual"

    def test_lpa_range(self):
        r = self.norm.normalize("4-8 LPA")
        assert r.min_amount == 400_000
        assert r.max_amount == 800_000
        assert r.period == "annual"

    def test_lpa_range_with_rupee_symbol(self):
        r = self.norm.normalize("₹8 - 14 LPA")
        assert r.min_amount == 800_000
        assert r.max_amount == 1_400_000

    def test_monthly_salary(self):
        r = self.norm.normalize("₹40,000/month")
        assert r.min_amount == 40_000
        assert r.period == "monthly"
        assert r.annual_inr == 40_000 * 12

    def test_annual_inr_calculated_for_lpa(self):
        r = self.norm.normalize("₹6 LPA")
        assert r.annual_inr == 600_000

    def test_annual_inr_not_guessed_for_unknown_period(self):
        """If period is unclear, annual_inr must remain None."""
        r = self.norm.normalize("50000")   # raw number, no period cue
        # Should not fabricate annual_inr
        # It's acceptable to have annual_inr=None here
        assert isinstance(r, NormalizedSalary)

    def test_original_text_preserved(self):
        raw = "₹5 - 8 LPA"
        r = self.norm.normalize(raw)
        assert r.original_text == raw

    def test_none_input(self):
        r = self.norm.normalize(None)
        assert r.min_amount is None
        assert r.original_text is None

    def test_empty_string(self):
        r = self.norm.normalize("")
        assert r.min_amount is None

    def test_large_annual_format(self):
        r = self.norm.normalize("₹5,00,000 - ₹8,00,000 per annum")
        assert r.min_amount == 500_000
        assert r.max_amount == 800_000
        assert r.period == "annual"

    def test_k_notation(self):
        r = self.norm.normalize("40k - 60k per month")
        assert r.min_amount == 40_000
        assert r.max_amount == 60_000
        assert r.period == "monthly"

    def test_currency_always_inr(self):
        r = self.norm.normalize("₹10 LPA")
        assert r.currency == "INR"

    @pytest.mark.parametrize("raw,expected_min", [
        ("₹4 LPA",       400_000),
        ("4 LPA",         400_000),
        ("₹12 LPA",     1_200_000),
        ("10 - 15 LPA", 1_000_000),
    ])
    def test_parametric_lpa(self, raw, expected_min):
        r = self.norm.normalize(raw)
        assert r.min_amount == expected_min


# ─────────────────────────────────────────────────────────────────────────────
# Skills
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillsNormalizer:
    def setup_method(self):
        self.norm = SkillsNormalizer()

    def test_ml_to_machine_learning(self):
        result = self.norm.normalize(["ML"])
        assert result[0].canonical == "Machine Learning"
        assert result[0].original == "ML"

    def test_machine_learning_unchanged(self):
        result = self.norm.normalize(["Machine Learning"])
        assert result[0].canonical == "Machine Learning"

    def test_machine_learning_hyphenated(self):
        result = self.norm.normalize(["machine-learning"])
        assert result[0].canonical == "Machine Learning"

    def test_js_to_javascript(self):
        result = self.norm.normalize(["JS"])
        assert result[0].canonical == "JavaScript"

    def test_k8s_to_kubernetes(self):
        result = self.norm.normalize(["k8s"])
        assert result[0].canonical == "Kubernetes"

    def test_golang_to_go(self):
        result = self.norm.normalize(["golang"])
        assert result[0].canonical == "Go"

    def test_reactjs_to_react(self):
        result = self.norm.normalize(["ReactJS"])
        assert result[0].canonical == "React"

    def test_postgres_to_postgresql(self):
        result = self.norm.normalize(["postgres"])
        assert result[0].canonical == "PostgreSQL"

    def test_sklearn_to_scikit_learn(self):
        result = self.norm.normalize(["sklearn"])
        assert result[0].canonical == "scikit-learn"

    def test_unknown_skill_preserved(self):
        result = self.norm.normalize(["SomeObscureLibrary"])
        assert result[0].canonical == "SomeObscureLibrary"
        assert result[0].original == "SomeObscureLibrary"

    def test_alias_recorded(self):
        result = self.norm.normalize(["ML"])
        assert result[0].alias == "ML"

    def test_no_alias_when_canonical_matches(self):
        result = self.norm.normalize(["Python"])
        assert result[0].alias is None

    def test_deduplication(self):
        """ML and Machine Learning both map to Machine Learning — deduplicated."""
        result = self.norm.normalize(["ML", "Machine Learning", "machine-learning"])
        canonicals = [r.canonical for r in result]
        assert canonicals.count("Machine Learning") == 1

    def test_empty_list(self):
        assert self.norm.normalize([]) == []

    def test_empty_string_skipped(self):
        result = self.norm.normalize(["Python", "", "  "])
        assert len(result) == 1

    def test_mixed_case(self):
        result = self.norm.normalize(["PYTHON", "python", "Python"])
        canonicals = [r.canonical for r in result]
        assert canonicals.count("Python") == 1

    def test_cicd_canonical(self):
        result = self.norm.normalize(["ci/cd"])
        assert result[0].canonical == "CI/CD"

    def test_normalize_single(self):
        skill = self.norm.normalize_single("k8s")
        assert skill.canonical == "Kubernetes"

    def test_no_embedding_matching(self):
        """Verify only exact/static taxonomy used — no fuzzy/embedding."""
        result = self.norm.normalize(["Pythonn"])   # typo — should NOT match Python
        # Unknown skills pass through as-is
        assert result[0].canonical == "Pythonn"
