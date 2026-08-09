"""
Tests for LLMInterpreter — Ollama mocked throughout.
No live LLM calls.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import pytest

from job_discovery.extraction.llm_interpreter import (
    EXTRACTION_SCHEMA_VERSION,
    ExtractionStatus,
    LLMCache,
    LLMExtractionResult,
    LLMInterpreter,
    LLMJobOutput,
    OllamaClient,
    OllamaError,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

VALID_OUTPUT = {
    "title": "Senior Python Developer",
    "company_name": "TechCorp India",
    "location": "Chennai, Tamil Nadu",
    "employment_type": "full-time",
    "work_mode": "hybrid",
    "experience_min_years": 3,
    "experience_max_years": 6,
    "salary_raw": "₹8 - 14 LPA",
    "salary_currency": "INR",
    "salary_min": 800000,
    "salary_max": 1400000,
    "salary_period": "annual",
    "required_skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
    "preferred_skills": ["Kafka", "AWS"],
    "responsibilities": ["Design REST APIs", "Write tests"],
    "description_summary": "Backend engineering role using Python and FastAPI.",
    "benefits": ["Health insurance", "Flexible hours"],
    "posted_date": "2025-01-15",
    "apply_url": "https://techcorp.in/careers/apply",
    "apply_email": "careers@techcorp.in",
    "seniority": "senior",
}

SAMPLE_TEXT = "Senior Python Developer at TechCorp India, Chennai. Full-time hybrid role."
SAMPLE_HASH = "abc123def456"


def _mock_ollama(response: str) -> MagicMock:
    mock = MagicMock(spec=OllamaClient)
    mock.model = "qwen2.5:7b"
    mock.complete.return_value = response
    return mock


def _interpreter(ollama=None, max_retries=3, cache=None) -> LLMInterpreter:
    if ollama is None:
        ollama = _mock_ollama(json.dumps(VALID_OUTPUT))
    return LLMInterpreter(
        ollama_client=ollama,
        cache=cache or LLMCache(),
        max_retries=max_retries,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLMJobOutput schema validation
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMJobOutputSchema:
    def test_valid_complete_output(self):
        o = LLMJobOutput.model_validate(VALID_OUTPUT)
        assert o.title == "Senior Python Developer"
        assert o.employment_type == "full-time"
        assert o.experience_min_years == 3

    def test_all_none_valid(self):
        o = LLMJobOutput.model_validate({})
        assert o.title is None
        assert o.required_skills is None

    def test_employment_type_normalised(self):
        o = LLMJobOutput.model_validate({"employment_type": "Full Time"})
        assert o.employment_type == "full-time"

    def test_work_mode_wfh_normalised(self):
        o = LLMJobOutput.model_validate({"work_mode": "wfh"})
        assert o.work_mode == "remote"

    def test_extra_fields_ignored(self):
        data = {**VALID_OUTPUT, "fabricated_field": "should not appear"}
        o = LLMJobOutput.model_validate(data)
        assert not hasattr(o, "fabricated_field")

    def test_skills_are_list(self):
        o = LLMJobOutput.model_validate({"required_skills": ["Python", "Go"]})
        assert isinstance(o.required_skills, list)

    def test_null_skills_ok(self):
        o = LLMJobOutput.model_validate({"required_skills": None})
        assert o.required_skills is None


# ─────────────────────────────────────────────────────────────────────────────
# LLMInterpreter — valid JSON
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMInterpreterValidJson:
    def test_returns_success_status(self):
        interp = _interpreter()
        result = interp.interpret(SAMPLE_TEXT, SAMPLE_HASH)
        assert result.status == ExtractionStatus.SUCCESS

    def test_output_populated(self):
        interp = _interpreter()
        result = interp.interpret(SAMPLE_TEXT, SAMPLE_HASH)
        assert result.output is not None
        assert result.output.title == "Senior Python Developer"

    def test_model_recorded(self):
        interp = _interpreter()
        result = interp.interpret(SAMPLE_TEXT, SAMPLE_HASH)
        assert result.model == "qwen2.5:7b"

    def test_schema_version_recorded(self):
        interp = _interpreter()
        result = interp.interpret(SAMPLE_TEXT, SAMPLE_HASH)
        assert result.schema_version == EXTRACTION_SCHEMA_VERSION

    def test_zero_retries_on_valid_json(self):
        interp = _interpreter()
        result = interp.interpret(SAMPLE_TEXT, SAMPLE_HASH)
        assert result.retry_count == 0

    def test_strips_markdown_fences(self):
        ollama = _mock_ollama(f"```json\n{json.dumps(VALID_OUTPUT)}\n```")
        interp = _interpreter(ollama=ollama)
        result = interp.interpret(SAMPLE_TEXT, SAMPLE_HASH)
        assert result.status == ExtractionStatus.SUCCESS

    def test_missing_fields_are_none(self):
        sparse = {"title": "Data Engineer", "company_name": None}
        ollama = _mock_ollama(json.dumps(sparse))
        interp = _interpreter(ollama=ollama)
        result = interp.interpret(SAMPLE_TEXT, SAMPLE_HASH)
        assert result.output.title == "Data Engineer"
        assert result.output.location is None
        assert result.output.required_skills is None


# ─────────────────────────────────────────────────────────────────────────────
# LLMInterpreter — invalid JSON → retry
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMInterpreterRetry:
    def test_invalid_json_triggers_retry(self):
        call_count = {"n": 0}
        valid_json = json.dumps(VALID_OUTPUT)

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "not valid json {"
            return valid_json

        ollama = MagicMock(spec=OllamaClient)
        ollama.model = "qwen2.5:7b"
        ollama.complete.side_effect = side_effect

        interp = _interpreter(ollama=ollama, max_retries=3)
        result = interp.interpret(SAMPLE_TEXT, SAMPLE_HASH)

        assert result.status == ExtractionStatus.SUCCESS
        assert result.retry_count == 1
        assert ollama.complete.call_count == 2

    def test_retry_sends_error_in_prompt(self):
        """Second call must include the validation error."""
        responses = [
            "invalid json {{{",
            json.dumps(VALID_OUTPUT),
        ]
        call_args_list = []

        def side_effect(*args, **kwargs):
            call_args_list.append(args)
            return responses.pop(0)

        ollama = MagicMock(spec=OllamaClient)
        ollama.model = "qwen2.5:7b"
        ollama.complete.side_effect = side_effect

        interp = _interpreter(ollama=ollama, max_retries=3)
        interp.interpret(SAMPLE_TEXT, SAMPLE_HASH)

        # Second call's user message should mention the error
        second_user_msg = call_args_list[1][1]
        assert "validation" in second_user_msg.lower() or "error" in second_user_msg.lower()

    def test_exhausted_retries_returns_failed_schema(self):
        ollama = _mock_ollama("not json at all {{{")
        interp = _interpreter(ollama=ollama, max_retries=2)
        result = interp.interpret(SAMPLE_TEXT, SAMPLE_HASH)

        assert result.status == ExtractionStatus.FAILED_SCHEMA
        assert result.validation_errors is not None
        assert result.retry_count == 2

    def test_exhausted_retries_does_not_silently_drop(self):
        """Failure must be recorded, not silently discarded."""
        ollama = _mock_ollama("{}")
        # {} is valid JSON but passes Pydantic (all null) — this should succeed
        # Use truly invalid JSON to force failure
        ollama = _mock_ollama("not-json")
        interp = _interpreter(ollama=ollama, max_retries=1)
        result = interp.interpret(SAMPLE_TEXT, SAMPLE_HASH)
        assert result.status == ExtractionStatus.FAILED_SCHEMA
        assert result.validation_errors is not None

    def test_schema_failure_reason_set(self):
        ollama = _mock_ollama("broken")
        interp = _interpreter(ollama=ollama, max_retries=1)
        result = interp.interpret(SAMPLE_TEXT, SAMPLE_HASH)
        assert result.status == ExtractionStatus.FAILED_SCHEMA


# ─────────────────────────────────────────────────────────────────────────────
# LLMInterpreter — Ollama unavailable
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMInterpreterOllamaError:
    def test_ollama_error_returns_failed_llm(self):
        ollama = MagicMock(spec=OllamaClient)
        ollama.model = "qwen2.5:7b"
        ollama.complete.side_effect = OllamaError("connection refused")

        interp = _interpreter(ollama=ollama)
        result = interp.interpret(SAMPLE_TEXT, SAMPLE_HASH)

        assert result.status == ExtractionStatus.FAILED_LLM
        assert "refused" in result.validation_errors.lower()


# ─────────────────────────────────────────────────────────────────────────────
# LLM Cache
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMCache:
    def test_cache_miss_calls_ollama(self):
        ollama = _mock_ollama(json.dumps(VALID_OUTPUT))
        cache = LLMCache()
        interp = LLMInterpreter(ollama_client=ollama, cache=cache, max_retries=1)
        interp.interpret(SAMPLE_TEXT, SAMPLE_HASH)
        assert ollama.complete.call_count == 1

    def test_cache_hit_does_not_call_ollama(self):
        ollama = _mock_ollama(json.dumps(VALID_OUTPUT))
        cache = LLMCache()
        interp = LLMInterpreter(ollama_client=ollama, cache=cache, max_retries=1)

        # First call — cache miss
        interp.interpret(SAMPLE_TEXT, SAMPLE_HASH)
        assert ollama.complete.call_count == 1

        # Second call — cache hit
        result = interp.interpret(SAMPLE_TEXT, SAMPLE_HASH)
        assert ollama.complete.call_count == 1  # no new call
        assert result.cache_hit is True

    def test_different_hash_causes_new_call(self):
        ollama = _mock_ollama(json.dumps(VALID_OUTPUT))
        cache = LLMCache()
        interp = LLMInterpreter(ollama_client=ollama, cache=cache)

        interp.interpret(SAMPLE_TEXT, "hash_A")
        interp.interpret(SAMPLE_TEXT, "hash_B")

        assert ollama.complete.call_count == 2

    def test_cache_hit_status(self):
        ollama = _mock_ollama(json.dumps(VALID_OUTPUT))
        cache = LLMCache()
        interp = LLMInterpreter(ollama_client=ollama, cache=cache)
        interp.interpret(SAMPLE_TEXT, SAMPLE_HASH)
        result = interp.interpret(SAMPLE_TEXT, SAMPLE_HASH)
        assert result.status == ExtractionStatus.CACHE_HIT

    def test_cache_size(self):
        cache = LLMCache()
        assert cache.size() == 0
        ollama = _mock_ollama(json.dumps(VALID_OUTPUT))
        interp = LLMInterpreter(ollama_client=ollama, cache=cache)
        interp.interpret(SAMPLE_TEXT, "h1")
        interp.interpret(SAMPLE_TEXT, "h2")
        assert cache.size() == 2

    def test_cache_clear(self):
        ollama = _mock_ollama(json.dumps(VALID_OUTPUT))
        cache = LLMCache()
        interp = LLMInterpreter(ollama_client=ollama, cache=cache)
        interp.interpret(SAMPLE_TEXT, SAMPLE_HASH)
        cache.clear()
        assert cache.size() == 0
        # Next call should miss cache
        interp.interpret(SAMPLE_TEXT, SAMPLE_HASH)
        assert ollama.complete.call_count == 2

    def test_different_schema_version_causes_miss(self):
        ollama = _mock_ollama(json.dumps(VALID_OUTPUT))
        cache = LLMCache()
        i1 = LLMInterpreter(ollama_client=ollama, cache=cache, schema_version="1.0")
        i2 = LLMInterpreter(ollama_client=ollama, cache=cache, schema_version="2.0")

        i1.interpret(SAMPLE_TEXT, SAMPLE_HASH)
        i2.interpret(SAMPLE_TEXT, SAMPLE_HASH)

        assert ollama.complete.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# Non-fabrication guard
# ─────────────────────────────────────────────────────────────────────────────

class TestNonFabrication:
    def test_empty_page_text_all_null(self):
        """LLM output with all null fields must be accepted."""
        all_null = {k: None for k in LLMJobOutput.model_fields}
        ollama = _mock_ollama(json.dumps(all_null))
        interp = _interpreter(ollama=ollama)
        result = interp.interpret("", "empty_hash")
        assert result.status == ExtractionStatus.SUCCESS
        assert result.output is not None
        assert result.output.title is None
        assert result.output.required_skills is None
