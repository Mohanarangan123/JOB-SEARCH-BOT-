"""
ResultCollector — orchestrates query execution, cooldown enforcement,
result deduplication, and persistence of SearchResult records.

Phase 3 scope: query fan-out + cooldown.
Phase 4 (fetch/extraction) is intentionally NOT wired here yet.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from job_discovery.models.source import QueryCache
from job_discovery.search.providers.base import (
    ProviderSearchResult,
    RateLimitedError,
    SearchProvider,
)
from job_discovery.search.query_builder import QueryBuilder, SearchCriteria
from job_discovery.search.query_expander import QueryExpander

logger = logging.getLogger(__name__)


class ResultCollector:
    """
    Drives the search phase for one orchestrator run:
      1. Build queries from criteria.
      2. Optionally expand via QueryExpander.
      3. Check per-query cooldown (skip if still cooling down).
      4. Execute via SearchProvider.
      5. Deduplicate by URL.
      6. Persist cache entries (caller supplies the repository).

    The collector does NOT:
      - Fetch job pages  (Prompt 3 / page_fetcher)
      - Extract or normalise content  (Prompt 4)
      - Write to the jobs collection
    """

    def __init__(
        self,
        search_provider: SearchProvider,
        query_builder: QueryBuilder,
        query_expander: Optional[QueryExpander] = None,
        *,
        cooldown_seconds: int = 60,
        max_results_per_query: int = 10,
    ) -> None:
        self._provider = search_provider
        self._builder = query_builder
        self._expander = query_expander or QueryExpander()
        self._cooldown = cooldown_seconds
        self._max_per_query = max_results_per_query

        # In-memory cooldown store: query_hash -> last_run datetime
        # The orchestrator should supplement this with the DB QueryCache
        self._cooldown_cache: dict[str, datetime] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def collect(
        self,
        criteria: SearchCriteria,
        *,
        external_cache: Optional[dict[str, datetime]] = None,
    ) -> List[ProviderSearchResult]:
        """
        Run the full search phase for the given criteria.

        Args:
            criteria:       User search criteria.
            external_cache: Optional mapping {query_hash: last_run_at} sourced
                            from the DB QueryCache, merged with the in-memory store.

        Returns:
            Deduplicated list of ProviderSearchResults.
        """
        # Merge in-memory + external cooldown data
        effective_cache = dict(self._cooldown_cache)
        if external_cache:
            effective_cache.update(external_cache)

        # Build + expand queries
        base_queries = self._builder.build(criteria)
        queries = self._expander.expand(base_queries, max_total=5)

        logger.debug("Generated %d queries: %s", len(queries), queries)

        all_results: List[ProviderSearchResult] = []
        seen_urls: set[str] = set()

        for query in queries:
            q_hash = ProviderSearchResult.make_query_hash(query)

            # Cooldown check
            if self._is_cooling(q_hash, effective_cache):
                logger.debug("Skipping cooled-down query: %r (hash=%s)", query, q_hash)
                continue

            source_filter = self._builder.get_target_sources(criteria) or None

            try:
                results = self._provider.search(
                    query,
                    max_results=self._max_per_query,
                    source_filter=source_filter,
                )
            except RateLimitedError:
                logger.warning("Rate limited on query %r — stopping search loop", query)
                break
            except Exception as exc:
                logger.error("Search error for query %r: %s", query, exc)
                continue

            # Deduplicate by URL
            for r in results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    all_results.append(r)

            # Update cooldown
            effective_cache[q_hash] = datetime.now(timezone.utc)
            self._cooldown_cache[q_hash] = effective_cache[q_hash]

        logger.info(
            "ResultCollector: %d unique results from %d queries",
            len(all_results),
            len(queries),
        )
        return all_results

    def make_cache_entry(
        self,
        query: str,
        source_name: str,
        results: List[ProviderSearchResult],
        *,
        cooldown_seconds: Optional[int] = None,
    ) -> QueryCache:
        """
        Build a QueryCache model ready for persistence.
        Caller is responsible for calling repository.query_cache.upsert().
        """
        ttl = cooldown_seconds if cooldown_seconds is not None else self._cooldown
        now = datetime.now(timezone.utc)
        q_hash = ProviderSearchResult.make_query_hash(query)
        return QueryCache(
            query_hash=q_hash,
            query_text=query,
            source_name=source_name,
            results=[r.model_dump() for r in results],
            result_count=len(results),
            cached_at=now,
            expires_at=now + timedelta(seconds=ttl),
        )

    def is_on_cooldown(self, query: str) -> bool:
        """Public helper: check whether a query is currently cooling down."""
        q_hash = ProviderSearchResult.make_query_hash(query)
        return self._is_cooling(q_hash, self._cooldown_cache)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _is_cooling(
        self, query_hash: str, cache: dict[str, datetime]
    ) -> bool:
        last_run = cache.get(query_hash)
        if last_run is None:
            return False
        elapsed = (datetime.now(timezone.utc) - last_run).total_seconds()
        return elapsed < self._cooldown
