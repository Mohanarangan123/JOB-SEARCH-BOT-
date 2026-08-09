"""
Typed application settings loaded from environment / .env file.
Single source of truth — never use os.getenv() elsewhere.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # MongoDB
    # ------------------------------------------------------------------ #
    mongodb_uri: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI",
        validation_alias=AliasChoices(
            "MONGODB_URI",
            "MONGO_URI",
            "mongodb_uri",
        ),
    )
    mongodb_database: str = Field(
        default="job_discovery",
        description="Name of the MongoDB database",
        validation_alias=AliasChoices(
            "MONGODB_DATABASE",
            "MONGO_DB",
            "mongodb_database",
        ),
    )
    mongo_collection: str = Field(
        default="jobs",
        description="Name of the jobs collection",
        validation_alias=AliasChoices(
            "MONGO_COLLECTION",
            "mongo_collection",
        ),
    )

    # ------------------------------------------------------------------ #
    # Search & Fetch
    # ------------------------------------------------------------------ #
    max_search_results: int = Field(
        default=100,
        ge=1,
        description="Maximum number of search results to collect per run",
    )
    max_fetches_per_run: int = Field(
        default=50,
        ge=1,
        description="Maximum number of pages to fetch per orchestrator run",
    )
    request_timeout: int = Field(
        default=30,
        ge=1,
        description="HTTP request timeout in seconds",
    )
    retry_count: int = Field(
        default=3,
        ge=0,
        description="Number of HTTP retry attempts",
    )
    backoff_base: float = Field(
        default=2.0,
        gt=0,
        description="Exponential backoff base multiplier (seconds)",
    )
    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        description="Consecutive failures before opening the circuit breaker",
    )
    query_cooldown: int = Field(
        default=60,
        ge=0,
        description="Seconds to wait before re-issuing the same query",
    )

    # ------------------------------------------------------------------ #
    # LLM
    # ------------------------------------------------------------------ #
    llm_model: str = Field(
        default="qwen2.5:7b",
        description="Ollama model tag used for job content interpretation",
    )
    llm_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the local Ollama server",
    )
    llm_max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum LLM schema-validation retries per extraction",
    )
    llm_cache_ttl: int = Field(
        default=86400,
        ge=0,
        description="TTL in seconds for LLM response cache entries",
    )
    extraction_schema_version: str = Field(
        default="1.0",
        description="Version tag for the extraction schema — bump to invalidate cache",
    )

    # ------------------------------------------------------------------ #
    # Storage & Export
    # ------------------------------------------------------------------ #
    raw_retention_days: int = Field(
        default=30,
        ge=1,
        description="Days to retain raw HTML/content before purging",
    )
    export_output_path: str = Field(
        default="./exports",
        description="Directory path for XLSX export output files",
    )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    job_removed_after_consecutive_failures: int = Field(
        default=3,
        ge=1,
        description="Mark job as removed after this many consecutive fetch failures",
    )

    # ------------------------------------------------------------------ #
    # Ranking
    # ------------------------------------------------------------------ #
    ranking_weights: Dict[str, float] = Field(
        default={"recency": 0.4, "completeness": 0.3, "source_reliability": 0.3},
        description="Weights used by the relevance scorer (keys must sum to 1.0)",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()
