"""Tests for typed Settings configuration loading."""
from __future__ import annotations

import sys

import pytest


def _reload_config():
    """Force a fresh import of config (bypasses lru_cache)."""
    for mod in list(sys.modules.keys()):
        if mod == "config":
            del sys.modules[mod]
    import config as cfg
    cfg.get_settings.cache_clear()
    return cfg


class TestDefaultSettings:
    def test_defaults_load_without_env_file(self, tmp_path, monkeypatch):
        """Settings should load with sane defaults even if no .env file exists."""
        monkeypatch.chdir(tmp_path)
        cfg = _reload_config()
        s = cfg.Settings()

        assert s.mongodb_uri == "mongodb://localhost:27017"
        assert s.mongodb_database == "job_discovery"
        assert s.max_search_results == 100
        assert s.max_fetches_per_run == 50
        assert s.request_timeout == 30
        assert s.retry_count == 3
        assert s.backoff_base == 2.0
        assert s.circuit_breaker_threshold == 5
        assert s.query_cooldown == 60
        assert s.llm_model == "qwen2.5:7b"   # pinned Ollama model (Prompt 4)
        assert s.llm_max_retries == 3
        assert s.llm_cache_ttl == 86400
        assert s.raw_retention_days == 30
        assert s.export_output_path == "./exports"
        assert s.job_removed_after_consecutive_failures == 3
        assert isinstance(s.ranking_weights, dict)

    def test_no_v2_fields(self, tmp_path, monkeypatch):
        """V2 fields must NOT exist on the Settings model."""
        monkeypatch.chdir(tmp_path)
        cfg = _reload_config()
        s = cfg.Settings()
        assert not hasattr(s, "embedding_model"), "embedding_model must not exist"
        assert not hasattr(s, "dedup_similarity_threshold"), (
            "dedup_similarity_threshold must not exist"
        )

    def test_env_override(self, tmp_path, monkeypatch):
        """Environment variables should override defaults."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MAX_SEARCH_RESULTS", "250")
        monkeypatch.setenv("LLM_MODEL", "qwen2.5:14b")
        monkeypatch.setenv("MONGODB_DATABASE", "test_db")

        cfg = _reload_config()
        s = cfg.Settings()

        assert s.max_search_results == 250
        assert s.llm_model == "qwen2.5:14b"
        assert s.mongodb_database == "test_db"

    def test_ranking_weights_type(self, tmp_path, monkeypatch):
        """ranking_weights must be a dict of str -> float."""
        monkeypatch.chdir(tmp_path)
        cfg = _reload_config()
        s = cfg.Settings()
        for k, v in s.ranking_weights.items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_get_settings_singleton(self, tmp_path, monkeypatch):
        """get_settings() should return the same object on repeated calls."""
        monkeypatch.chdir(tmp_path)
        cfg = _reload_config()
        s1 = cfg.get_settings()
        s2 = cfg.get_settings()
        assert s1 is s2

    def test_extraction_schema_version_present(self, tmp_path, monkeypatch):
        """extraction_schema_version must exist (added Prompt 4)."""
        monkeypatch.chdir(tmp_path)
        cfg = _reload_config()
        s = cfg.Settings()
        assert hasattr(s, "extraction_schema_version")
        assert s.extraction_schema_version == "1.0"

    def test_llm_base_url_present(self, tmp_path, monkeypatch):
        """llm_base_url must point to local Ollama by default."""
        monkeypatch.chdir(tmp_path)
        cfg = _reload_config()
        s = cfg.Settings()
        assert s.llm_base_url == "http://localhost:11434"
