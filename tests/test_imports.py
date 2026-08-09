"""Smoke-test that all top-level modules import cleanly."""
from __future__ import annotations

import importlib

import pytest

MODULES = [
    "config",
    "job_discovery",
    "job_discovery.models.job",
    "job_discovery.models.search",
    "job_discovery.models.source",
    "job_discovery.repositories.job_repository",
    "job_discovery.repositories.search_repository",
    "job_discovery.api.routes",
    # Placeholder stubs
    "job_discovery.orchestrator.search_orchestrator",
    "job_discovery.sources.base",
    "job_discovery.sources.linkedin",
    "job_discovery.sources.indeed",
    "job_discovery.sources.naukri",
    "job_discovery.sources.cutshort",
    "job_discovery.sources.instahyre",
    "job_discovery.sources.hirist",
    "job_discovery.sources.wellfound",
    "job_discovery.sources.hirect",
    "job_discovery.sources.generic",
    "job_discovery.search.providers.base",
    "job_discovery.search.providers.web_search",
    "job_discovery.search.query_builder",
    "job_discovery.search.query_expander",
    "job_discovery.search.result_collector",
    "job_discovery.fetch.page_fetcher",
    "job_discovery.fetch.fetch_policy",
    "job_discovery.fetch.retry",
    "job_discovery.fetch.circuit_breaker",
    "job_discovery.extraction.content_extractor",
    "job_discovery.extraction.section_detector",
    "job_discovery.extraction.llm_interpreter",
    "job_discovery.normalization.location",
    "job_discovery.normalization.salary",
    "job_discovery.normalization.skills",
    "job_discovery.validation.completeness",
    "job_discovery.validation.schema_validator",
    "job_discovery.validation.quality_checker",
    "job_discovery.deduplication.exact_match",
    "job_discovery.deduplication.identity",
    "job_discovery.versioning.version_tracker",
    "job_discovery.ranking.relevance_scorer",
    "job_discovery.health.source_health",
    "job_discovery.export.xlsx_writer",
    "job_discovery.export.run_export",
    # Prompt 3
    "job_discovery.storage.raw_store",
    "job_discovery.deduplication.exact_match",
    "job_discovery.deduplication.identity",
    "job_discovery.fetch.retry",
    "job_discovery.fetch.circuit_breaker",
    "job_discovery.fetch.fetch_policy",
    "job_discovery.fetch.page_fetcher",
    # Prompt 4
    "job_discovery.extraction.section_detector",
    "job_discovery.extraction.content_extractor",
    "job_discovery.extraction.llm_interpreter",
    "job_discovery.normalization.location",
    "job_discovery.normalization.salary",
    "job_discovery.normalization.skills",
    # Prompt 5
    "job_discovery.validation.completeness",
    "job_discovery.validation.schema_validator",
    "job_discovery.validation.quality_checker",
    "job_discovery.versioning.version_tracker",
    "job_discovery.health.source_health",
    # Prompt 6
    "job_discovery.ranking.relevance_scorer",
    "job_discovery.orchestrator.run_tracker",
    "job_discovery.orchestrator.search_orchestrator",
    "job_discovery.api.schemas",
    "job_discovery.api.dependencies",
    "job_discovery.api.routes",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str):
    """Each module must import without raising exceptions."""
    importlib.import_module(module)
