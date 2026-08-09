"""
FastAPI dependency injection helpers.
Provides shared DB / repo / scorer instances per request.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from job_discovery.db import get_db
from job_discovery.ranking.relevance_scorer import RelevanceScorer
from job_discovery.repositories.job_repository import JobRepository
from job_discovery.repositories.search_repository import SearchRepository


@lru_cache(maxsize=1)
def get_job_repo() -> JobRepository:
    return JobRepository(get_db())


@lru_cache(maxsize=1)
def get_search_repo() -> SearchRepository:
    return SearchRepository(get_db())


@lru_cache(maxsize=1)
def get_scorer() -> RelevanceScorer:
    return RelevanceScorer()
