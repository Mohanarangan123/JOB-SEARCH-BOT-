"""
LLMInterpreter — interprets already-extracted job content using a local
Ollama model (Qwen2.5 by default).

Design rules:
  - The LLM receives ONLY text already present in the raw page.
  - The LLM never invents job facts; missing information → null.
  - Strict JSON schema enforced via Pydantic.
  - Validation retry loop: re-prompt with error message up to LLM_MAX_RETRIES.
  - LLM cache keyed on (content_hash, schema_version, model_name).
  - No external API calls — Ollama runs locally.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schema for LLM output
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTION_SCHEMA_VERSION = "1.0"


class LLMJobOutput(BaseModel):
    """
    Schema that the LLM must return as JSON.
    All fields are optional — missing → null, never fabricated.
    """
    title:               Optional[str] = None
    company_name:        Optional[str] = None
    location:            Optional[str] = None
    employment_type:     Optional[str] = None   # full-time | part-time | contract | …
    work_mode:           Optional[str] = None   # remote | hybrid | onsite
    experience_min_years: Optional[int] = None
    experience_max_years: Optional[int] = None
    salary_raw:          Optional[str] = None
    salary_currency:     Optional[str] = None
    salary_min:          Optional[float] = None
    salary_max:          Optional[float] = None
    salary_period:       Optional[str] = None   # annual | monthly | hourly
    required_skills:     Optional[List[str]] = None
    preferred_skills:    Optional[List[str]] = None
    responsibilities:    Optional[List[str]] = None
    description_summary: Optional[str] = None
    benefits:            Optional[List[str]] = None
    posted_date:         Optional[str] = None   # ISO date string or null
    apply_url:           Optional[str] = None
    apply_email:         Optional[str] = None
    seniority:           Optional[str] = None   # junior | mid | senior | lead | …

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @field_validator("employment_type", mode="before")
    @classmethod
    def normalise_employment_type(cls, v):
        if v is None:
            return None
        mapping = {
            "fulltime": "full-time", "full time": "full-time",
            "parttime": "part-time", "part time": "part-time",
        }
        return mapping.get(str(v).lower().strip(), str(v).lower().strip())

    @field_validator("work_mode", mode="before")
    @classmethod
    def normalise_work_mode(cls, v):
        if v is None:
            return None
        s = str(v).lower().strip()
        if s in ("wfh", "work from home"):
            return "remote"
        return s


# ─────────────────────────────────────────────────────────────────────────────
# Extraction status
# ─────────────────────────────────────────────────────────────────────────────

class ExtractionStatus:
    SUCCESS               = "success"
    PARTIAL               = "partial"
    FAILED_SCHEMA         = "extraction_failed"
    FAILED_LLM            = "llm_unavailable"
    CACHE_HIT             = "cache_hit"


class LLMExtractionResult:
    """
    Result of one LLM extraction attempt.
    """
    def __init__(
        self,
        *,
        status: str,
        output: Optional[LLMJobOutput] = None,
        raw_response: Optional[str] = None,
        validation_errors: Optional[str] = None,
        retry_count: int = 0,
        cache_hit: bool = False,
        model: Optional[str] = None,
        schema_version: str = EXTRACTION_SCHEMA_VERSION,
    ) -> None:
        self.status = status
        self.output = output
        self.raw_response = raw_response
        self.validation_errors = validation_errors
        self.retry_count = retry_count
        self.cache_hit = cache_hit
        self.model = model
        self.schema_version = schema_version


# ─────────────────────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a precise job-data extraction assistant.
Extract structured information from the provided job posting text.

CRITICAL RULES:
1. Return ONLY valid JSON matching the schema below.
2. Use null for any field that is NOT explicitly mentioned in the text.
3. Do NOT invent, guess, or assume any information.
4. Do NOT add markdown, code fences, or commentary — raw JSON only.
5. Skills must come directly from the text — do not add unlisted skills.

JSON Schema:
{
  "title":               string | null,
  "company_name":        string | null,
  "location":            string | null,
  "employment_type":     "full-time"|"part-time"|"contract"|"internship"|null,
  "work_mode":           "remote"|"hybrid"|"onsite"|null,
  "experience_min_years": integer | null,
  "experience_max_years": integer | null,
  "salary_raw":          string | null,
  "salary_currency":     string | null,
  "salary_min":          number | null,
  "salary_max":          number | null,
  "salary_period":       "annual"|"monthly"|"hourly"|null,
  "required_skills":     [string] | null,
  "preferred_skills":    [string] | null,
  "responsibilities":    [string] | null,
  "description_summary": string (max 300 chars) | null,
  "benefits":            [string] | null,
  "posted_date":         "YYYY-MM-DD" | null,
  "apply_url":           string | null,
  "apply_email":         string | null,
  "seniority":           "junior"|"mid"|"senior"|"lead"|"principal"|"staff"|null
}
"""

_USER_TEMPLATE = """\
Job posting text:
---
{text}
---

Extract all available information. Missing fields must be null.
"""

_RETRY_TEMPLATE = """\
Your previous response failed JSON schema validation.
Validation error: {error}

Please provide a corrected JSON response.
Previous response was:
{previous}

Requirements:
- Fix the validation error.
- Keep all correctly extracted values.
- Set invalid/missing fields to null.
- Return raw JSON only, no markdown.
"""


# ─────────────────────────────────────────────────────────────────────────────
# In-process LLM cache
# ─────────────────────────────────────────────────────────────────────────────

class LLMCache:
    """
    In-memory LLM response cache.
    Key: (content_hash, schema_version, model_name)
    """

    def __init__(self) -> None:
        self._store: Dict[str, LLMExtractionResult] = {}

    def _key(self, content_hash: str, schema_version: str, model: str) -> str:
        return hashlib.sha256(
            f"{content_hash}|{schema_version}|{model}".encode()
        ).hexdigest()

    def get(self, content_hash: str, schema_version: str, model: str) -> Optional[LLMExtractionResult]:
        return self._store.get(self._key(content_hash, schema_version, model))

    def put(self, content_hash: str, schema_version: str, model: str, result: LLMExtractionResult) -> None:
        self._store[self._key(content_hash, schema_version, model)] = result

    def size(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()


# ─────────────────────────────────────────────────────────────────────────────
# OllamaClient — thin wrapper around the Ollama HTTP API
# ─────────────────────────────────────────────────────────────────────────────

class OllamaClient:
    """
    Thin synchronous wrapper around the Ollama /api/generate endpoint.
    Inject a mock for tests.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:7b") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def complete(self, system_prompt: str, user_message: str, *, timeout: int = 120) -> str:
        """
        Send a completion request to Ollama.
        Returns the model's raw text response.
        Raises OllamaError on failure.
        """
        import httpx  # import here to keep this class testable without httpx at module level

        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}\n\n{user_message}",
            "stream": False,
            "options": {"temperature": 0.0},
        }
        try:
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except Exception as exc:
            raise OllamaError(f"Ollama call failed: {exc}") from exc


class OllamaError(Exception):
    """Raised when the Ollama server is unreachable or returns an error."""


# ─────────────────────────────────────────────────────────────────────────────
# LLMInterpreter
# ─────────────────────────────────────────────────────────────────────────────

class LLMInterpreter:
    """
    Interprets a cleaned job page text using a local Ollama LLM.

    Flow:
      1. Check LLMCache → return cached result if hit.
      2. Build prompt from page text.
      3. Call Ollama.
      4. Parse + validate JSON → LLMJobOutput.
      5. On validation failure, re-prompt with error (up to max_retries).
      6. If still invalid → status = extraction_failed.
      7. Store result in cache.

    Args:
        ollama_client:    OllamaClient instance (or mock).
        cache:            LLMCache instance (shared across runs).
        max_retries:      Maximum validation-retry attempts.
        schema_version:   Bump to invalidate all cached entries.
        max_text_chars:   Truncate page text to this length before prompting.
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        cache: Optional[LLMCache] = None,
        *,
        max_retries: int = 3,
        schema_version: str = EXTRACTION_SCHEMA_VERSION,
        max_text_chars: int = 6000,
    ) -> None:
        self._client = ollama_client
        self._cache = cache or LLMCache()
        self._max_retries = max_retries
        self._schema_version = schema_version
        self._max_chars = max_text_chars

    def interpret(
        self,
        page_text: str,
        content_hash: str,
    ) -> LLMExtractionResult:
        """
        Interpret a job page text.

        Args:
            page_text:    Cleaned text (from ContentExtractor).
            content_hash: SHA-256 of raw content (for cache key).

        Returns:
            LLMExtractionResult.
        """
        model = self._client.model

        # ── Cache check ──────────────────────────────────────────────────
        cached = self._cache.get(content_hash, self._schema_version, model)
        if cached is not None:
            cached.cache_hit = True
            cached.status = ExtractionStatus.CACHE_HIT
            logger.debug("LLM cache hit: hash=%s model=%s", content_hash[:8], model)
            return cached

        # ── Truncate text ────────────────────────────────────────────────
        text = page_text[:self._max_chars]

        # ── Attempt 1 ───────────────────────────────────────────────────
        user_msg = _USER_TEMPLATE.format(text=text)
        raw_response = ""
        last_error = ""
        retry_count = 0

        try:
            raw_response = self._client.complete(_SYSTEM_PROMPT, user_msg)
        except OllamaError as exc:
            result = LLMExtractionResult(
                status=ExtractionStatus.FAILED_LLM,
                raw_response="",
                validation_errors=str(exc),
                model=model,
                schema_version=self._schema_version,
            )
            return result

        # ── Parse + validate loop ────────────────────────────────────────
        output, last_error = self._parse_and_validate(raw_response)

        while output is None and retry_count < self._max_retries:
            retry_count += 1
            logger.warning(
                "LLM schema validation failed (attempt %d): %s",
                retry_count, last_error
            )
            retry_msg = _RETRY_TEMPLATE.format(
                error=last_error,
                previous=raw_response[:1000],
            )
            try:
                raw_response = self._client.complete(_SYSTEM_PROMPT, retry_msg)
            except OllamaError as exc:
                last_error = str(exc)
                break
            output, last_error = self._parse_and_validate(raw_response)

        if output is None:
            result = LLMExtractionResult(
                status=ExtractionStatus.FAILED_SCHEMA,
                raw_response=raw_response,
                validation_errors=last_error,
                retry_count=retry_count,
                model=model,
                schema_version=self._schema_version,
            )
        else:
            status = ExtractionStatus.SUCCESS
            result = LLMExtractionResult(
                status=status,
                output=output,
                raw_response=raw_response,
                retry_count=retry_count,
                model=model,
                schema_version=self._schema_version,
            )

        # ── Store in cache ───────────────────────────────────────────────
        self._cache.put(content_hash, self._schema_version, model, result)
        return result

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _parse_and_validate(self, raw: str) -> tuple[Optional[LLMJobOutput], str]:
        """
        Strip markdown fences, parse JSON, validate with Pydantic.
        Returns (LLMJobOutput, "") on success or (None, error_message) on failure.
        """
        cleaned = raw.strip()
        # Strip markdown code fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.M)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.M)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return None, f"JSON parse error: {exc}"

        try:
            output = LLMJobOutput.model_validate(data)
            return output, ""
        except ValidationError as exc:
            return None, str(exc)
