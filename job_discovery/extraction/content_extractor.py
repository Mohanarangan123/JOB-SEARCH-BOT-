"""
ContentExtractor — deterministic HTML → structured ExtractedJob.

Pipeline:
  raw HTML
    → BeautifulSoup (lxml parser)
    → text cleaning
    → SectionDetector
    → field-specific regex extraction
    → ExtractedJob (with evidence)

Rules:
  - No LLM calls here.
  - Missing fields → None (never invented).
  - Evidence (original_text, source_section) preserved for every field.
  - Raw text is the source of truth.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup

from job_discovery.extraction.section_detector import DetectedSection, SectionDetector


# ─────────────────────────────────────────────────────────────────────────────
# Evidence-carrying field
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvidenceField:
    """Wrapper that preserves the original text evidence for a field value."""
    value: Optional[Any]
    original_text: Optional[str] = None
    source_section: Optional[str] = None
    confidence: float = 1.0

    def is_present(self) -> bool:
        return self.value is not None and self.value != ""


# ─────────────────────────────────────────────────────────────────────────────
# ExtractedJob — output of deterministic extraction
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractedJob:
    """
    All deterministically extracted fields from a job page.
    Every field preserves evidence (original text + source section).
    Fields not found in the HTML are None — never fabricated.
    """
    # Core identity
    title:            Optional[EvidenceField] = None
    company_name:     Optional[EvidenceField] = None
    location:         Optional[EvidenceField] = None
    employment_type:  Optional[EvidenceField] = None
    work_mode:        Optional[EvidenceField] = None
    posted_date:      Optional[EvidenceField] = None
    updated_date:     Optional[EvidenceField] = None
    salary_raw:       Optional[EvidenceField] = None
    experience_raw:   Optional[EvidenceField] = None

    # Sections
    description_text:  Optional[str] = None
    responsibilities:  Optional[List[str]] = None
    requirements_text: Optional[str] = None
    preferred_text:    Optional[str] = None
    benefits_text:     Optional[str] = None

    # Application
    apply_url:        Optional[EvidenceField] = None
    apply_email:      Optional[EvidenceField] = None

    # Meta
    page_text:        Optional[str] = None      # full cleaned text for LLM
    sections:         Optional[List[DetectedSection]] = None
    extracted_at:     datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ─────────────────────────────────────────────────────────────────────────────
# HTML cleaning helpers
# ─────────────────────────────────────────────────────────────────────────────

_NOISE_TAGS = {"script", "style", "noscript", "header", "footer", "nav",
               "aside", "iframe", "svg", "button", "form"}


def clean_html(html: bytes | str) -> str:
    """
    Parse HTML with BeautifulSoup (lxml), remove noise tags, return clean text.
    """
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Field-specific regex extractors
# ─────────────────────────────────────────────────────────────────────────────

_SALARY_RE = re.compile(
    r"(₹\s?[\d,]+(?:\.\d+)?\s*(?:LPA|L\.P\.A\.|lakh|lakhs|k|/month|/yr|/year|per\s+annum)?"
    r"(?:\s*[-–—to]+\s*₹?\s*[\d,]+(?:\.\d+)?\s*(?:LPA|L\.P\.A\.|lakh|lakhs|k|/month|/yr|/year|per\s+annum)?)?)"
    r"|(\d+(?:\.\d+)?\s*[-–—to]+\s*\d+(?:\.\d+)?\s*LPA)"
    r"|(INR\s*[\d,]+(?:\.\d+)?(?:\s*[-–—to]+\s*INR?\s*[\d,]+(?:\.\d+)?)?)",
    re.I,
)

_EXPERIENCE_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*[-–—+]\s*\d+(?:\.\d+)?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience)?)"
    r"|(\d+\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience)?)",
    re.I,
)

_EMPLOYMENT_TYPE_RE = re.compile(
    r"\b(full[- ]time|part[- ]time|contract|freelance|intern(ship)?|temporary|permanent)\b",
    re.I,
)

_WORK_MODE_RE = re.compile(
    r"\b(remote|work\s+from\s+home|wfh|hybrid|on[- ]?site|in[- ]?office)\b",
    re.I,
)

_POSTED_DATE_RE = re.compile(
    r"(?:posted|published|listed)\s*(?:on|:)?\s*"
    r"((?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})"
    r"|(?:\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4})"
    r"|(?:\d+\s+(?:day|hour|minute|week)s?\s+ago))",
    re.I,
)

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
)

_APPLY_URL_RE = re.compile(
    r'(?:apply|application)\s+(?:here|now|at|url|link)?[:\s]*'
    r'(https?://[^\s"\'<>]+)',
    re.I,
)


def _first_match(pattern: re.Pattern, text: str, section: str) -> Optional[EvidenceField]:
    m = pattern.search(text)
    if m:
        return EvidenceField(
            value=m.group(0).strip(),
            original_text=m.group(0).strip(),
            source_section=section,
        )
    return None


def _extract_list_items(text: str) -> List[str]:
    """Extract bullet-point or numbered items from text."""
    items = []
    for line in text.splitlines():
        line = line.strip()
        # Strip bullets: •, -, *, ·, >, –, numbers
        clean = re.sub(r"^[•\-\*·>–\d]+[.)]\s*", "", line).strip()
        if clean and len(clean) > 4:
            items.append(clean)
    return items


# ─────────────────────────────────────────────────────────────────────────────
# ContentExtractor
# ─────────────────────────────────────────────────────────────────────────────

class ContentExtractor:
    """
    Deterministic HTML → ExtractedJob converter.

    Design rules:
      - No LLM calls.
      - No invented values.
      - All fields backed by text evidence.
    """

    def __init__(self) -> None:
        self._detector = SectionDetector()

    def extract(self, html: bytes | str, *, source_url: str = "") -> ExtractedJob:
        """
        Main entry point. Parse HTML and return an ExtractedJob.

        Args:
            html:       Raw HTML bytes or string.
            source_url: URL of the page (for context only; not used for values).
        """
        page_text = clean_html(html)
        sections = self._detector.detect(page_text)
        sec_dict = self._detector.as_dict(sections)

        job = ExtractedJob(
            page_text=page_text,
            sections=sections,
        )

        # ── Title ────────────────────────────────────────────────────────
        job.title = self._extract_title(page_text, sec_dict)

        # ── Company ──────────────────────────────────────────────────────
        job.company_name = self._extract_company(page_text, sec_dict)

        # ── Location ─────────────────────────────────────────────────────
        job.location = self._extract_location(page_text, sec_dict)

        # ── Employment type + work mode ──────────────────────────────────
        full_text = page_text[:2000]  # search near top of page
        emp_match = _first_match(_EMPLOYMENT_TYPE_RE, full_text, "page_top")
        if emp_match:
            job.employment_type = emp_match

        wm_match = _first_match(_WORK_MODE_RE, full_text, "page_top")
        if wm_match:
            job.work_mode = wm_match

        # ── Salary ───────────────────────────────────────────────────────
        salary_text = sec_dict.get("salary") or page_text[:4000]
        sal_match = _first_match(_SALARY_RE, salary_text, "salary")
        if sal_match:
            job.salary_raw = sal_match

        # ── Experience ───────────────────────────────────────────────────
        req_text = sec_dict.get("requirements", page_text[:3000])
        exp_match = _first_match(_EXPERIENCE_RE, req_text, "requirements")
        if exp_match:
            job.experience_raw = exp_match

        # ── Posted date ──────────────────────────────────────────────────
        date_match = _first_match(_POSTED_DATE_RE, page_text, "page_top")
        if date_match:
            job.posted_date = date_match

        # ── Sections ─────────────────────────────────────────────────────
        job.description_text = sec_dict.get("description") or sec_dict.get("body")
        job.responsibilities = (
            _extract_list_items(sec_dict["responsibilities"])
            if "responsibilities" in sec_dict else None
        )
        job.requirements_text = sec_dict.get("requirements")
        job.preferred_text    = sec_dict.get("preferred_qualifications")
        job.benefits_text     = sec_dict.get("benefits")

        # ── Application ──────────────────────────────────────────────────
        app_section = sec_dict.get("application", page_text)
        url_match = _first_match(_APPLY_URL_RE, app_section, "application")
        if url_match:
            job.apply_url = url_match

        email_match = _first_match(_EMAIL_RE, app_section, "application")
        if email_match:
            job.apply_email = email_match

        return job

    # ------------------------------------------------------------------ #
    # Field-specific extractors
    # ------------------------------------------------------------------ #

    def _extract_title(self, page_text: str, sec_dict: Dict[str, str]) -> Optional[EvidenceField]:
        """Try to find job title in the first few lines of the page."""
        lines = [l.strip() for l in page_text.splitlines() if l.strip()]
        for line in lines[:10]:
            # Heuristic: title is a short capitalised line
            if 3 < len(line) < 120 and not line.endswith(":"):
                # Skip obvious non-titles
                if not re.search(r"(?i)(login|sign in|register|home|menu|search)", line):
                    return EvidenceField(
                        value=line,
                        original_text=line,
                        source_section="page_top",
                        confidence=0.7,
                    )
        return None

    def _extract_company(self, page_text: str, sec_dict: Dict[str, str]) -> Optional[EvidenceField]:
        """Try common patterns for company name."""
        patterns = [
            re.compile(r"(?i)(?:company|employer|organization|organisation)\s*[:\-]\s*(.+)", re.M),
            re.compile(r"(?i)at\s+([A-Z][A-Za-z\s&.,]+?)(?:\s+we\b|\s+is\b|\.|,)", re.M),
        ]
        for pattern in patterns:
            m = pattern.search(page_text[:3000])
            if m:
                val = m.group(1).strip()
                if 2 < len(val) < 80:
                    return EvidenceField(
                        value=val,
                        original_text=m.group(0).strip(),
                        source_section="page_top",
                        confidence=0.6,
                    )
        return None

    def _extract_location(self, page_text: str, sec_dict: Dict[str, str]) -> Optional[EvidenceField]:
        """Extract location from structured patterns."""
        patterns = [
            re.compile(r"(?i)(?:location|place|city|office)\s*[:\-]\s*([^\n]+)", re.M),
            # Indian cities
            re.compile(
                r"\b((?:Bangalore|Bengaluru|Mumbai|Delhi|Chennai|Hyderabad|Pune|Kolkata|"
                r"Noida|Gurgaon|Gurugram|Ahmedabad|Jaipur|Kochi|Coimbatore|Chandigarh|"
                r"Indore|Bhopal|Nagpur|Lucknow|Visakhapatnam|Surat|Vadodara|Thiruvananthapuram)"
                r"(?:\s*,\s*(?:TN|MH|DL|KA|TS|AP|GJ|RJ|UP|WB|Punjab|Maharashtra|Tamil Nadu|"
                r"Karnataka|Telangana|Andhra Pradesh|Gujarat|Rajasthan))?)",
                re.M,
            ),
        ]
        for pattern in patterns:
            m = pattern.search(page_text[:3000])
            if m:
                val = m.group(1).strip() if m.lastindex and m.lastindex >= 1 else m.group(0).strip()
                if 2 < len(val) < 100:
                    return EvidenceField(
                        value=val,
                        original_text=m.group(0).strip(),
                        source_section="page_top",
                    )
        return None
