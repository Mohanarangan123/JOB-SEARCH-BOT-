"""
Salary normalizer.

Parses free-text Indian salary strings into structured NormalizedSalary.

Supported formats:
  ₹4 LPA               → min=400000  max=None    period=annual  currency=INR
  4-8 LPA              → min=400000  max=800000  period=annual  currency=INR
  ₹40,000/month        → min=40000   max=None    period=monthly currency=INR
  ₹5,00,000 - ₹8,00,000 per annum → min=500000  max=800000  period=annual
  40k - 60k per month  → min=40000   max=60000   period=monthly

Rules:
  - Never guess a value.
  - Only calculate annual_inr when the period is unambiguously annual or monthly.
  - Preserve original_text always.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class NormalizedSalary:
    currency:      Optional[str]   = None
    min_amount:    Optional[float] = None
    max_amount:    Optional[float] = None
    period:        Optional[str]   = None    # annual | monthly | hourly
    annual_inr:    Optional[float] = None    # calculated only when unambiguous
    original_text: Optional[str]   = None


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_LAKH  = 100_000
_CRORE = 10_000_000

# Period keywords
_ANNUAL_RE  = re.compile(
    r"\b(lpa|l\.p\.a\.|l\s*p\s*a|per\s+annum|p\.a\.|annual(?:ly)?|yearly)\b", re.I
)
_MONTHLY_RE = re.compile(r"\b(per\s+month|/month|monthly|p\.m\.)\b", re.I)
_HOURLY_RE  = re.compile(r"\b(per\s+hour|/hour|hourly)\b", re.I)

# Range separator (must come BEFORE we strip to identify two tokens)
_RANGE_SEP = re.compile(r"\s*[-–—]+\s*|\s+to\s+", re.I)

# A number token, optionally followed by a multiplier unit
_TOKEN_RE = re.compile(
    r"([\d,]+(?:\.\d+)?)\s*(lpa|l\.p\.a\.|lakh(?:s)?|lac|crore|cr|k|m)?",
    re.I,
)


def _apply_multiplier(num: float, unit: str) -> float:
    """Scale num by the unit suffix."""
    u = (unit or "").lower().strip().rstrip("s")  # strip plural 's'
    if u in ("lpa", "l.p.a.", "lakh", "lac"):
        return num * _LAKH
    if u in ("crore", "cr"):
        return num * _CRORE
    if u == "k":
        return num * 1_000
    if u == "m":
        return num * 1_000_000
    return num


def _parse_token(token: str, global_unit: Optional[str] = None) -> Optional[float]:
    """
    Parse a single numeric token.

    token        — cleaned number string, may contain commas (e.g. '5,00,000')
    global_unit  — unit detected from the full raw string (e.g. 'lpa')
    """
    token = token.strip()
    if not token:
        return None

    m = _TOKEN_RE.search(token)
    if not m:
        return None

    num_str = m.group(1).replace(",", "")
    try:
        num = float(num_str)
    except ValueError:
        return None

    local_unit = m.group(2) or ""
    unit = local_unit or global_unit or ""
    return _apply_multiplier(num, unit)


def _detect_global_unit(raw: str) -> Optional[str]:
    """Return the dominant unit keyword found anywhere in raw."""
    m = re.search(r"\b(lpa|l\.p\.a\.|lakh(?:s)?|lac|crore|cr|k|m)\b", raw, re.I)
    return m.group(1).lower() if m else None


class SalaryNormalizer:
    """Parse a free-text Indian salary string into NormalizedSalary."""

    def normalize(self, raw: Optional[str]) -> NormalizedSalary:
        if not raw or not raw.strip():
            return NormalizedSalary(original_text=raw)

        original = raw.strip()
        result = NormalizedSalary(original_text=original, currency="INR")

        # ── Detect period ────────────────────────────────────────────────
        if _ANNUAL_RE.search(original):
            result.period = "annual"
        elif _MONTHLY_RE.search(original):
            result.period = "monthly"
        elif _HOURLY_RE.search(original):
            result.period = "hourly"

        # ── Detect global unit (LPA, lakh, k, …) ────────────────────────
        global_unit = _detect_global_unit(original)

        # ── Strip currency symbols and noise words to isolate numbers ────
        stripped = re.sub(r"[₹$€£]", "", original)
        stripped = re.sub(
            r"\b(lpa|l\.p\.a\.|l\s*p\s*a|per\s+annum|annual(?:ly)?|yearly|"
            r"per\s+month|/month|monthly|per\s+hour|/hour|hourly|"
            r"inr|rupees?|rs\.?|p\.a\.|p\.m\.)\b",
            " ", stripped, flags=re.I,
        )
        stripped = stripped.strip()

        # ── Try to split into range ──────────────────────────────────────
        parts = _RANGE_SEP.split(stripped, maxsplit=1)

        if len(parts) == 2 and parts[1].strip():
            result.min_amount = _parse_token(parts[0].strip(), global_unit)
            result.max_amount = _parse_token(parts[1].strip(), global_unit)
        else:
            result.min_amount = _parse_token(stripped, global_unit)

        # ── Calculate annual_inr only when unambiguous ───────────────────
        if result.min_amount is not None:
            if result.period == "annual":
                result.annual_inr = result.min_amount
            elif result.period == "monthly":
                result.annual_inr = result.min_amount * 12

        return result
