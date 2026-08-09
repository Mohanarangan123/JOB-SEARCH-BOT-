"""
Tests for deterministic HTML extraction: ContentExtractor + SectionDetector.
Uses frozen fixture HTML — no live websites.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from job_discovery.extraction.content_extractor import (
    ContentExtractor,
    ExtractedJob,
    EvidenceField,
    clean_html,
)
from job_discovery.extraction.section_detector import SectionDetector

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> bytes:
    return (FIXTURE_DIR / name).read_bytes()


# ─────────────────────────────────────────────────────────────────────────────
# clean_html
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanHtml:
    def test_removes_script_tags(self):
        html = b"<html><script>alert(1)</script><body>Job Title</body></html>"
        text = clean_html(html)
        assert "alert" not in text
        assert "Job Title" in text

    def test_removes_style_tags(self):
        html = b"<html><style>.foo{color:red}</style><body>hello</body></html>"
        assert ".foo" not in clean_html(html)

    def test_removes_nav(self):
        html = b"<html><nav>Navigation</nav><main>Content</main></html>"
        text = clean_html(html)
        assert "Navigation" not in text
        assert "Content" in text

    def test_collapses_blank_lines(self):
        html = b"<html><body><p>A</p><p>B</p></body></html>"
        text = clean_html(html)
        assert "\n\n\n" not in text

    def test_bytes_and_str_accepted(self):
        html_str = "<html><body>Hello</body></html>"
        assert "Hello" in clean_html(html_str)
        assert "Hello" in clean_html(html_str.encode())


# ─────────────────────────────────────────────────────────────────────────────
# SectionDetector
# ─────────────────────────────────────────────────────────────────────────────

class TestSectionDetector:
    def setup_method(self):
        self.detector = SectionDetector()

    def test_detects_responsibilities(self):
        text = "Responsibilities\n- Design APIs\n- Write tests\n"
        sections = self.detector.detect(text)
        labels = [s.label for s in sections]
        assert "responsibilities" in labels

    def test_detects_requirements(self):
        text = "Requirements\n- 3 years Python\n- REST APIs\n"
        sections = self.detector.detect(text)
        labels = [s.label for s in sections]
        assert "requirements" in labels

    def test_detects_benefits(self):
        text = "Benefits\n- Health insurance\n- Flexible hours\n"
        sections = self.detector.detect(text)
        assert any(s.label == "benefits" for s in sections)

    def test_detects_preferred_qualifications(self):
        text = "Preferred Qualifications\n- Kafka experience\n- AWS knowledge\n"
        sections = self.detector.detect(text)
        assert any(s.label == "preferred_qualifications" for s in sections)

    def test_empty_input_returns_empty(self):
        assert self.detector.detect("") == []
        assert self.detector.detect("   ") == []

    def test_unknown_text_becomes_body(self):
        text = "Some random text that does not match any section heading."
        sections = self.detector.detect(text)
        assert any(s.label == "body" for s in sections)

    def test_multiple_sections_detected(self):
        text = (
            "Responsibilities\n- Task 1\n\n"
            "Requirements\n- Req 1\n\n"
            "Benefits\n- Benefit 1\n"
        )
        sections = self.detector.detect(text)
        labels = {s.label for s in sections}
        assert "responsibilities" in labels
        assert "requirements" in labels
        assert "benefits" in labels

    def test_section_text_content(self):
        text = "Requirements\n- Python 3 years\n- FastAPI knowledge\n"
        sections = self.detector.detect(text)
        req = next((s for s in sections if s.label == "requirements"), None)
        assert req is not None
        assert "Python" in req.text

    def test_get_section_helper(self):
        text = "Benefits\n- Health insurance\n"
        sections = self.detector.detect(text)
        content = self.detector.get_section(sections, "benefits")
        assert content is not None
        assert "Health" in content

    def test_as_dict(self):
        text = "Responsibilities\n- Design\nRequirements\n- Python\n"
        sections = self.detector.detect(text)
        d = self.detector.as_dict(sections)
        assert "responsibilities" in d or "requirements" in d

    def test_what_youll_do_detected_as_responsibilities(self):
        text = "What you'll do\n- Build APIs\n- Write tests\n"
        sections = self.detector.detect(text)
        assert any(s.label == "responsibilities" for s in sections)


# ─────────────────────────────────────────────────────────────────────────────
# ContentExtractor — with golden fixture
# ─────────────────────────────────────────────────────────────────────────────

class TestContentExtractorFixture:
    def setup_method(self):
        self.extractor = ContentExtractor()
        self.html = load_fixture("sample_job.html")
        self.job = self.extractor.extract(self.html)

    def test_returns_extracted_job(self):
        assert isinstance(self.job, ExtractedJob)

    def test_page_text_populated(self):
        assert self.job.page_text is not None
        assert len(self.job.page_text) > 100

    def test_sections_detected(self):
        assert self.job.sections is not None
        assert len(self.job.sections) > 0

    def test_salary_extracted(self):
        """Salary must be extracted from HTML evidence."""
        assert self.job.salary_raw is not None
        assert isinstance(self.job.salary_raw, EvidenceField)
        assert "LPA" in self.job.salary_raw.original_text.upper() or \
               "₹" in self.job.salary_raw.original_text

    def test_salary_evidence_preserved(self):
        """original_text must contain the raw salary text."""
        assert self.job.salary_raw.original_text is not None
        assert len(self.job.salary_raw.original_text) > 0

    def test_employment_type_extracted(self):
        assert self.job.employment_type is not None
        assert "full" in self.job.employment_type.value.lower() or \
               "time" in self.job.employment_type.value.lower()

    def test_work_mode_extracted(self):
        assert self.job.work_mode is not None
        assert "hybrid" in self.job.work_mode.value.lower() or \
               "remote" in self.job.work_mode.value.lower()

    def test_experience_extracted(self):
        assert self.job.experience_raw is not None
        assert "3" in self.job.experience_raw.original_text or \
               "6" in self.job.experience_raw.original_text

    def test_responsibilities_extracted(self):
        assert self.job.responsibilities is not None
        assert len(self.job.responsibilities) >= 3

    def test_responsibilities_are_strings(self):
        for r in self.job.responsibilities:
            assert isinstance(r, str)
            assert len(r) > 4

    def test_requirements_text_extracted(self):
        assert self.job.requirements_text is not None
        assert "Python" in self.job.requirements_text

    def test_benefits_text_extracted(self):
        assert self.job.benefits_text is not None
        assert len(self.job.benefits_text) > 10

    def test_apply_email_extracted(self):
        assert self.job.apply_email is not None
        assert "@" in self.job.apply_email.value

    def test_apply_url_extracted(self):
        assert self.job.apply_url is not None
        assert "https://" in self.job.apply_url.value

    def test_no_fabrication_of_missing_fields(self):
        """Fields not in HTML must be None, never invented."""
        # salary_raw should only be set if the HTML contains salary info
        if self.job.salary_raw is not None:
            # evidence must exist in page_text
            assert self.job.salary_raw.original_text in self.job.page_text

    def test_evidence_traceable_to_page_text(self):
        """All extracted EvidenceField values must appear in page_text."""
        page = self.job.page_text
        for field_name in ["salary_raw", "employment_type", "work_mode", "experience_raw"]:
            ef = getattr(self.job, field_name)
            if ef is not None:
                assert ef.original_text in page, (
                    f"{field_name}.original_text not found in page_text"
                )


class TestContentExtractorMinimalFixture:
    def setup_method(self):
        self.extractor = ContentExtractor()
        self.html = load_fixture("sample_job_minimal.html")
        self.job = self.extractor.extract(self.html)

    def test_missing_fields_are_none(self):
        """Sparse HTML should produce None for missing fields, not invented values."""
        assert self.job.salary_raw is None
        assert self.job.experience_raw is None
        assert self.job.employment_type is None

    def test_page_text_present(self):
        assert self.job.page_text is not None
        assert "Software Engineer" in self.job.page_text


class TestContentExtractorEdgeCases:
    def setup_method(self):
        self.extractor = ContentExtractor()

    def test_empty_html(self):
        job = self.extractor.extract(b"")
        assert isinstance(job, ExtractedJob)

    def test_html_with_only_noise(self):
        html = b"<html><head><style>body{color:red}</style></head><body></body></html>"
        job = self.extractor.extract(html)
        assert isinstance(job, ExtractedJob)

    def test_bytes_input(self):
        html = b"<html><body><h1>Python Dev</h1></body></html>"
        job = self.extractor.extract(html)
        assert isinstance(job, ExtractedJob)

    def test_string_input(self):
        html = "<html><body><h1>Python Dev</h1></body></html>"
        job = self.extractor.extract(html)
        assert isinstance(job, ExtractedJob)
