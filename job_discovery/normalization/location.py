"""
Location normalizer.

Converts free-text location strings into a structured NormalizedLocation,
preserving the original text as evidence.

Examples:
  "Chennai"                      → city=Chennai, state=Tamil Nadu, country=India
  "Chennai, TN"                  → city=Chennai, state=Tamil Nadu, country=India
  "Chennai, Tamil Nadu"          → city=Chennai, state=Tamil Nadu, country=India
  "Chennai, Tamil Nadu, India"   → city=Chennai, state=Tamil Nadu, country=India
  "Bangalore"                    → city=Bangalore, state=Karnataka, country=India
  "Remote"                       → is_remote=True

Rules:
  - Never invent values not present in the input.
  - Abbreviations expanded only from the known static map.
  - Unknown cities/states preserved as-is in city field.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Static reference data
# ─────────────────────────────────────────────────────────────────────────────

# Indian state abbreviations → full name
_STATE_ABBR: dict[str, str] = {
    "AN": "Andaman and Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Daman and Diu",
    "DL": "Delhi",
    "DN": "Dadra and Nagar Haveli",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OR": "Odisha",
    "OD": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TG": "Telangana",
    "TS": "Telangana",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "UK": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}

# City → (state, country) — major Indian tech hubs
_CITY_LOOKUP: dict[str, tuple[str, str]] = {
    "bangalore":         ("Karnataka", "India"),
    "bengaluru":         ("Karnataka", "India"),
    "mumbai":            ("Maharashtra", "India"),
    "pune":              ("Maharashtra", "India"),
    "delhi":             ("Delhi", "India"),
    "new delhi":         ("Delhi", "India"),
    "gurgaon":           ("Haryana", "India"),
    "gurugram":          ("Haryana", "India"),
    "noida":             ("Uttar Pradesh", "India"),
    "hyderabad":         ("Telangana", "India"),
    "secunderabad":      ("Telangana", "India"),
    "chennai":           ("Tamil Nadu", "India"),
    "coimbatore":        ("Tamil Nadu", "India"),
    "madurai":           ("Tamil Nadu", "India"),
    "kolkata":           ("West Bengal", "India"),
    "ahmedabad":         ("Gujarat", "India"),
    "surat":             ("Gujarat", "India"),
    "vadodara":          ("Gujarat", "India"),
    "jaipur":            ("Rajasthan", "India"),
    "kochi":             ("Kerala", "India"),
    "thiruvananthapuram":("Kerala", "India"),
    "trivandrum":        ("Kerala", "India"),
    "chandigarh":        ("Punjab", "India"),
    "bhopal":            ("Madhya Pradesh", "India"),
    "indore":            ("Madhya Pradesh", "India"),
    "nagpur":            ("Maharashtra", "India"),
    "lucknow":           ("Uttar Pradesh", "India"),
    "visakhapatnam":     ("Andhra Pradesh", "India"),
    "vizag":             ("Andhra Pradesh", "India"),
    "patna":             ("Bihar", "India"),
    "ranchi":            ("Jharkhand", "India"),
}

_REMOTE_PATTERNS = re.compile(
    r"\b(remote|work\s+from\s+home|wfh|anywhere|pan\s+india)\b", re.I
)


# ─────────────────────────────────────────────────────────────────────────────
# Output model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NormalizedLocation:
    city:          Optional[str]  = None
    state:         Optional[str]  = None
    country:       Optional[str]  = None
    is_remote:     bool           = False
    original_text: Optional[str]  = None

    def display(self) -> str:
        parts = [p for p in [self.city, self.state, self.country] if p]
        base = ", ".join(parts)
        if self.is_remote:
            return f"Remote — {base}" if base else "Remote"
        return base or self.original_text or ""


# ─────────────────────────────────────────────────────────────────────────────
# Normalizer
# ─────────────────────────────────────────────────────────────────────────────

class LocationNormalizer:
    """
    Normalize a free-text location string.

    Preserves original_text always.
    Expands abbreviations only from the static map.
    Never invents state/country for unknown cities.
    """

    def normalize(self, raw: Optional[str]) -> NormalizedLocation:
        if not raw or not raw.strip():
            return NormalizedLocation(original_text=raw)

        original = raw.strip()
        result = NormalizedLocation(original_text=original)

        # Remote check
        if _REMOTE_PATTERNS.search(original):
            result.is_remote = True

        # Split on comma
        parts = [p.strip() for p in original.split(",")]

        city_raw = parts[0] if parts else ""
        state_raw = parts[1].strip() if len(parts) > 1 else ""
        country_raw = parts[2].strip() if len(parts) > 2 else ""

        # Resolve city
        city_key = city_raw.lower()
        if city_key in _CITY_LOOKUP:
            result.city = city_raw
            default_state, default_country = _CITY_LOOKUP[city_key]
        else:
            result.city = city_raw if city_raw else None
            default_state = None
            default_country = None

        # Resolve state
        if state_raw:
            upper = state_raw.upper()
            if upper in _STATE_ABBR:
                result.state = _STATE_ABBR[upper]
            else:
                result.state = state_raw  # use as-is
        elif default_state:
            result.state = default_state

        # Resolve country
        if country_raw:
            result.country = country_raw
        elif default_country:
            result.country = default_country

        return result
