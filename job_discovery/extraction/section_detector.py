"""
SectionDetector — identifies semantic sections in cleaned job-page text.

Responsibilities:
  - Split text into labelled sections (description, requirements, etc.)
  - Return section content with its label and character-offset evidence.
  - No LLM calls — purely deterministic regex/heuristic matching.
  - Missing sections remain None; no values are invented.

Detection ordering matters: more-specific patterns are checked first.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DetectedSection:
    """A section of text identified by label."""
    label: str
    text: str
    start: int
    end: int
    confidence: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Section heading patterns
# Order matters: more specific labels checked before broader ones.
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_PATTERNS: List[tuple] = [
    # ── Preferred qualifications (BEFORE requirements — more specific) ──
    ("preferred_qualifications", [
        r"(?i)^preferred\s+(qualifications?|skills?|requirements?)\s*$",
        r"(?i)^(nice[\s\-]to[\s\-]have|good[\s\-]to[\s\-]have|bonus\s+skills?|"
        r"additional\s+(qualifications?|skills?))\s*$",
        r"(?i)^(preferred|bonus|plus)\s*$",
        r"(?i)would\s+be\s+(an?\s+)?advantage",
    ]),

    # ── Responsibilities ─────────────────────────────────────────────────
    ("responsibilities",  [
        r"(?i)^(key\s+)?responsibilit(y|ies)\s*$",
        r"(?i)^what\s+you.ll\s+do\s*$",
        r"(?i)^your\s+role\s*$",
        r"(?i)^day[- ]to[- ]day\s*$",
        r"(?i)^job\s+duties\s*$",
        r"(?i)^duties\s+and\s+responsibilities\s*$",
    ]),

    # ── Requirements (anchored to avoid matching "Preferred...") ────────
    ("requirements",      [
        r"(?i)^(required|minimum|must[\s\-]have)\s+(qualifications?|skills?|"
        r"requirements?)\s*$",
        r"(?i)^requirements?\s*$",
        r"(?i)^qualifications?\s*$",
        r"(?i)^what\s+we.re?\s+looking\s+for\s*$",
        r"(?i)^you\s+should\s+have\s*$",
        r"(?i)^(skills?\s+(required|needed|we\s+need))\s*$",
    ]),

    # ── Benefits (anchored — don't grab "Competitive salary: ₹8 LPA") ──
    ("benefits",          [
        r"(?i)^benefits?\s*$",
        r"(?i)^perks?\s*$",
        r"(?i)^what\s+we\s+offer\s*$",
        r"(?i)^why\s+(join|work\s+(at|with|for))\s*",
        r"(?i)^compensation\s+and\s+benefits\s*$",
        r"(?i)^what.s\s+in\s+it\s+for\s+you\s*$",
    ]),

    # ── Description ──────────────────────────────────────────────────────
    ("description",       [
        r"(?i)^about\s+the\s+(role|job|position|opportunity)\s*$",
        r"(?i)^(job|role|position)\s+(overview|summary|description)\s*$",
        r"(?i)^overview\s*$",
        r"(?i)^about\s+(us|the\s+company|techcorp|our\s+company)\s*$",
    ]),

    # ── Salary (standalone heading only) ────────────────────────────────
    ("salary",            [
        r"(?i)^(salary|compensation|ctc|pay|package|remuneration)\s*$",
    ]),

    # ── Application ──────────────────────────────────────────────────────
    ("application",       [
        r"(?i)^(how\s+to\s+apply|apply\s+now|application\s+(process|instructions?))\s*$",
    ]),
]

# Compile all patterns once
_COMPILED: List[tuple] = [
    (label, [re.compile(p, re.MULTILINE) for p in patterns])
    for label, patterns in _SECTION_PATTERNS
]

# Maximum heading length to avoid matching a long prose sentence
_MAX_HEADING_LEN = 80


class SectionDetector:
    """
    Splits a block of text into labelled sections using heading detection.
    """

    def detect(self, text: str) -> List[DetectedSection]:
        if not text or not text.strip():
            return []

        lines = text.splitlines()
        sections: List[DetectedSection] = []
        current_label: Optional[str] = None
        current_start: int = 0
        current_lines: List[str] = []
        char_offset = 0

        for line in lines:
            line_end = char_offset + len(line) + 1
            stripped = line.strip()

            matched_label = self._match_heading(stripped)

            if matched_label and len(stripped) < _MAX_HEADING_LEN:
                # Flush previous block
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append(DetectedSection(
                        label=current_label or "body",
                        text=body,
                        start=current_start,
                        end=char_offset,
                    ))
                current_label = matched_label
                current_start = char_offset
                current_lines = []
            else:
                current_lines.append(line)

            char_offset = line_end

        # Flush final block
        body = "\n".join(current_lines).strip()
        if body:
            sections.append(DetectedSection(
                label=current_label or "body",
                text=body,
                start=current_start,
                end=char_offset,
            ))

        return sections

    def get_section(self, sections: List[DetectedSection], label: str) -> Optional[str]:
        for s in sections:
            if s.label == label:
                return s.text
        return None

    def as_dict(self, sections: List[DetectedSection]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for s in sections:
            if s.label not in result:
                result[s.label] = s.text
        return result

    def _match_heading(self, line: str) -> Optional[str]:
        if not line:
            return None
        for label, patterns in _COMPILED:
            for pattern in patterns:
                if pattern.search(line):
                    return label
        return None
