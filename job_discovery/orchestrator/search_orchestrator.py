"""
SearchOrchestrator — connects all pipeline stages for one search run.

Pipeline:
  SearchCriteria
    → QueryBuilder + QueryExpander  (query generation)
    → SearchProvider                (URL discovery, site:-scoped)
    → IdentityResolver              (URL normalisation + exact dedup)
    → PageFetcher (async)           (HTTP retrieval + robots.txt + circuit breaker)
    → RawStore                      (raw HTML storage)
    → ContentExtractor              (deterministic HTML → structured fields)
    → LLMInterpreter                (Ollama interpretation, cached, optional)
    → QualityChecker                (completeness + schema validation)
    → VersionTracker                (versioning + change detection)
    → JobRepository                 (MongoDB upsert)

Crash recovery:
  - Run state persisted to MongoDB (search_runs + url_states).
  - On resume, already-processed URLs are skipped.
  - Already-executed queries are skipped via already_issued set.

One source failure does NOT terminate the run.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from config import get_settings
from job_discovery.deduplication.identity import IdentityResolver
from job_discovery.extraction.content_extractor import ContentExtractor
from job_discovery.extraction.llm_interpreter import LLMCache, LLMInterpreter, OllamaClient
from job_discovery.fetch.circuit_breaker import CircuitBreakerRegistry
from job_discovery.fetch.fetch_policy import FetchDecision, FetchPolicy, RobotsCache
from job_discovery.fetch.page_fetcher import FetchResult, FetchStatus, PageFetcher
from job_discovery.fetch.retry import RetryConfig
from job_discovery.health.source_health import SourceHealthMonitor
from job_discovery.models.job import (
    Application, Company, Compensation, Description,
    JobDetails, JobRecord, JobSource, LifecycleStatus,
    Requirements, RetrievalMetadata,
)
from job_discovery.models.search import RunStatus, SearchRun, UrlProcessingStatus, UrlState
from job_discovery.orchestrator.run_tracker import RunTracker
from job_discovery.repositories.job_repository import JobRepository
from job_discovery.repositories.search_repository import SearchRepository
from job_discovery.search.providers.base import SearchProvider
from job_discovery.search.query_builder import QueryBuilder, SearchCriteria
from job_discovery.search.query_expander import QueryExpander
from job_discovery.storage.raw_store import RawStore
from job_discovery.validation.quality_checker import QualityChecker
from job_discovery.versioning.version_tracker import VersionTracker

logger = logging.getLogger(__name__)


class SearchOrchestrator:
    """Coordinates a complete search run from criteria to stored JobRecords."""

    def __init__(
        self,
        search_provider: SearchProvider,
        job_repo: JobRepository,
        search_repo: SearchRepository,
        raw_store: RawStore,
        *,
        http_client: Optional[httpx.AsyncClient] = None,
        ollama_client: Optional[OllamaClient] = None,
        llm_cache: Optional[LLMCache] = None,
    ) -> None:
        settings = get_settings()
        self._provider      = search_provider
        self._job_repo      = job_repo
        self._search_repo   = search_repo
        self._raw_store     = raw_store
        self._http          = http_client
        self._settings      = settings

        self._query_builder  = QueryBuilder()
        self._query_expander = QueryExpander()
        self._resolver       = IdentityResolver()
        self._extractor      = ContentExtractor()
        self._quality        = QualityChecker()
        self._health_monitor = SourceHealthMonitor(alert_threshold=0.5, min_samples=5)
        self._circuits       = CircuitBreakerRegistry(
            failure_threshold=settings.circuit_breaker_threshold,
            recovery_timeout=60.0,
        )

        self._ollama = ollama_client
        self._llm_cache = llm_cache or LLMCache()
        self._interpreter: Optional[LLMInterpreter] = (
            LLMInterpreter(
                ollama_client=self._ollama,
                cache=self._llm_cache,
                max_retries=settings.llm_max_retries,
                schema_version=settings.extraction_schema_version,
            ) if self._ollama else None
        )
        self._retry_cfg = RetryConfig(
            max_attempts=settings.retry_count,
            base_delay=settings.backoff_base,
            jitter=1.0,
        )
        self._version_trackers: Dict[str, VersionTracker] = {}

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    async def run(self, criteria: SearchCriteria, *, run_id: Optional[str] = None) -> SearchRun:
        """Execute a full search run with crash-recovery support."""
        tracker = RunTracker(self._search_repo.runs, run_id=run_id)

        existing = self._search_repo.runs.find_by_run_id(tracker.run_id)
        is_resume = existing is not None and existing.get("status") in (
            RunStatus.running.value, RunStatus.partial.value
        )

        if is_resume:
            logger.info("Resuming interrupted run %s", tracker.run_id)
            tracker._run = SearchRun(**{k: v for k, v in existing.items() if k != "_id"})
            tracker._start_time = existing.get("started_at") or datetime.now(timezone.utc)
        else:
            tracker.start(criteria.model_dump())

        processed_urls = self._search_repo.url_states.get_processed_urls(tracker.run_id)
        already_issued: set = set(tracker.state.queries_issued or [])

        try:
            await self._execute(criteria, tracker, processed_urls, already_issued)
            tracker.complete()
        except Exception as exc:
            logger.exception("Run %s failed: %s", tracker.run_id, exc)
            tracker.mark_failed(str(exc))

        return tracker.state

    # ------------------------------------------------------------------ #
    # Pipeline stages
    # ------------------------------------------------------------------ #

    async def _execute(
        self,
        criteria: SearchCriteria,
        tracker: RunTracker,
        processed_urls: set,
        already_issued: set,
    ) -> None:
        settings = self._settings

        # 1. Query generation
        queries = self._query_expander.expand(
            self._query_builder.build(criteria), max_total=5
        )
        tracker.inc_queries(len(queries))

        # 2. Search — skip already-issued queries
        all_results = []
        for query in queries:
            if query in already_issued:
                continue
            try:
                results = self._provider.search(
                    query,
                    max_results=settings.max_search_results,
                    source_filter=self._query_builder.get_target_sources(criteria),
                )
                all_results.extend(results)
                tracker.mark_query_issued(query)
                tracker.inc_sources(1)
            except Exception as exc:
                logger.warning("Search failed for %r: %s", query, exc)

        tracker.inc_discovered(len(all_results))

        # 3. URL normalisation + dedup
        new_urls = []
        for r in all_results:
            norm = self._resolver.resolve(r.url)
            if norm is None:
                continue
            if norm.is_duplicate or norm.canonical_url in processed_urls:
                tracker.inc_duplicate()
                continue
            try:
                self._search_repo.url_states.upsert(UrlState(
                    run_id=tracker.run_id,
                    url=r.url,
                    canonical_url=norm.canonical_url,
                    source_name=norm.source_name,
                    status=UrlProcessingStatus.discovered,
                ))
            except Exception:
                pass
            new_urls.append(norm)

        # 4. Fetch + extract + store (one source failure does not stop others)
        robots_cache = RobotsCache(http_client=None)   # fail-open: no real network call
        fetch_policy = FetchPolicy(
            self._circuits, robots_cache,
            max_fetches=settings.max_fetches_per_run,
        )

        # Use injected client directly (avoid async context manager on mock)
        if self._http is not None:
            client = self._http
            await self._process_urls(new_urls, tracker, fetch_policy, client, robots_cache)
        else:
            async with httpx.AsyncClient() as client:
                await self._process_urls(new_urls, tracker, fetch_policy, client, robots_cache)

    async def _process_urls(
        self,
        new_urls: list,
        tracker: RunTracker,
        fetch_policy: FetchPolicy,
        client: httpx.AsyncClient,
        robots_cache: RobotsCache,
    ) -> None:
        """Fetch, extract, validate, version, and store each URL."""
        settings = self._settings
        fetcher = PageFetcher(
            raw_store=self._raw_store,
            circuit_registry=self._circuits,
            robots_cache=robots_cache,
            retry_config=self._retry_cfg,
            timeout=float(settings.request_timeout),
            max_fetches=settings.max_fetches_per_run,
            http_client=client,
        )

        for norm in new_urls:
            url    = norm.canonical_url
            source = norm.source_name

            # Policy check
            policy = fetch_policy.check(url, source)
            if policy.decision != FetchDecision.ALLOW:
                self._update_url(tracker, norm, UrlProcessingStatus.fetch_skipped, policy.reason)
                tracker.inc_failed_fetch()
                continue

            # Fetch
            try:
                fetch_result: FetchResult = await fetcher.fetch(norm)
            except Exception as exc:
                logger.warning("Fetch exception %r: %s", url, exc)
                self._update_url(tracker, norm, UrlProcessingStatus.fetch_failed, str(exc))
                tracker.inc_failed_fetch()
                continue

            if fetch_result.status != FetchStatus.SUCCESS:
                self._update_url(tracker, norm, UrlProcessingStatus.fetch_failed,
                                 fetch_result.error or "")
                tracker.inc_failed_fetch()
                continue

            tracker.inc_fetched()
            self._update_url(tracker, norm, UrlProcessingStatus.fetched,
                             content_hash=fetch_result.content_hash, canonical_url=url)

            # Extract
            try:
                raw_html  = self._raw_store.load_html(source, fetch_result.content_hash[:16]) or b""
                extracted = self._extractor.extract(raw_html, source_url=url)
            except Exception as exc:
                logger.warning("Extraction error %r: %s", url, exc)
                tracker.inc_extraction_failure()
                continue

            # LLM (optional)
            llm_output = None
            if self._interpreter and extracted.page_text:
                try:
                    res = self._interpreter.interpret(extracted.page_text, fetch_result.content_hash)
                    llm_output = res.output
                except Exception as exc:
                    logger.warning("LLM failed %r: %s", url, exc)

            # Build + validate + version + store
            job_record = self._build_job_record(url, source, extracted, llm_output, fetch_result)
            job_dict   = job_record.model_dump()
            job_dict["page_text"] = extracted.page_text
            job_record.validation = self._quality.check(job_dict)

            vt = self._get_version_tracker(url)
            vt.record_seen()
            version = vt.record_fetch(
                content_hash=fetch_result.content_hash,
                extracted_data=job_dict,
                raw_content_path=fetch_result.raw_content_path,
            )
            job_record.lifecycle_status = LifecycleStatus(vt.lifecycle.value)
            job_record.first_seen_at    = vt.first_seen_at
            job_record.last_seen_at     = vt.last_seen_at
            job_record.last_verified_at = vt.last_verified_at

            try:
                self._job_repo.upsert_by_canonical_url(job_record)
                if version:
                    self._job_repo.save_version(url, {
                        "version_number":   version.version_number,
                        "content_hash":     version.content_hash,
                        "raw_content_path": version.raw_content_path,
                        "retrieved_at":     version.retrieved_at,
                    })
                    tracker.inc_unique_jobs()
                else:
                    tracker.state.jobs_updated = (tracker.state.jobs_updated or 0) + 1
            except Exception as exc:
                logger.error("MongoDB upsert failed %r: %s", url, exc)

            self._update_url(tracker, norm, UrlProcessingStatus.stored)

            # Health metrics
            missing = (job_record.validation.missing_fields or [])
            field_statuses = {f: "not_available" for f in missing}
            for f in ["title", "company", "location", "description",
                      "experience", "salary", "employment_type"]:
                field_statuses.setdefault(f, "present")
            self._health_monitor.record(source, field_statuses)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _update_url(self, tracker, norm, status: UrlProcessingStatus,
                    error_message: str = "", **extra) -> None:
        try:
            self._search_repo.url_states.update_status(
                tracker.run_id, norm.original_url, status.value,
                error_message=error_message, **extra,
            )
        except Exception:
            pass

    def _build_job_record(self, url, source, extracted, llm_output, fetch_result) -> JobRecord:
        title    = (llm_output.title     if llm_output else None) or \
                   (extracted.title.value if extracted.title else None)
        location = (llm_output.location  if llm_output else None) or \
                   (extracted.location.value if extracted.location else None)

        details = JobDetails(
            title=title,
            location=location,
            employment_type=(llm_output.employment_type if llm_output else None) or
                            (extracted.employment_type.value if extracted.employment_type else None),
            work_mode=(llm_output.work_mode if llm_output else None) or
                      (extracted.work_mode.value if extracted.work_mode else None),
            company=Company(
                name=(llm_output.company_name if llm_output else None) or
                     (extracted.company_name.value if extracted.company_name else None)
            ),
            description=Description(
                raw_text=extracted.description_text,
                responsibilities=extracted.responsibilities or
                                 (llm_output.responsibilities if llm_output else None),
            ),
            requirements=Requirements(
                required_skills=llm_output.required_skills if llm_output else None,
                preferred_skills=llm_output.preferred_skills if llm_output else None,
                experience_years_min=llm_output.experience_min_years if llm_output else None,
                experience_years_max=llm_output.experience_max_years if llm_output else None,
            ),
            compensation=Compensation(
                raw_text=extracted.salary_raw.value if extracted.salary_raw else None,
                currency="INR",
            ),
            application=Application(
                apply_url=extracted.apply_url.value if extracted.apply_url else None,
                apply_email=extracted.apply_email.value if extracted.apply_email else None,
            ),
            seniority=llm_output.seniority if llm_output else None,
        )
        return JobRecord(
            canonical_url=url,
            content_hash=fetch_result.content_hash,
            lifecycle_status=LifecycleStatus.active,
            source=JobSource(source_name=source, canonical_url=url,
                             scraped_at=fetch_result.retrieved_at),
            details=details,
            raw_content_path=fetch_result.raw_content_path,
            retrieval=RetrievalMetadata(
                content_hash=fetch_result.content_hash,
                raw_content_path=fetch_result.raw_content_path,
                last_verified_at=fetch_result.retrieved_at,
            ),
        )

    def _get_version_tracker(self, canonical_url: str) -> VersionTracker:
        if canonical_url not in self._version_trackers:
            self._version_trackers[canonical_url] = VersionTracker(
                removal_threshold=self._settings.job_removed_after_consecutive_failures
            )
        return self._version_trackers[canonical_url]
