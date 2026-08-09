"""
FastAPI API routes for the Job Discovery System — V1 complete.

Endpoints:
  POST /api/jobs/search          — trigger a search run
  GET  /api/jobs                 — list/rank stored jobs
  GET  /api/jobs/detail          — single job (canonical_url query param)
  GET  /api/jobs/versions        — version history
  GET  /api/jobs/changes         — change log
  GET  /api/search-runs/{run_id} — run state
  GET  /api/sources/health       — source health metrics
  POST /api/jobs/export          — export to XLSX
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from job_discovery.api.dependencies import get_job_repo, get_scorer, get_search_repo
from job_discovery.api.schemas import (
    ChangeResponse,
    ComponentScoresResponse,
    ExportResponse,
    FreshnessResponse,
    JobDetailResponse,
    JobSearchRequest,
    JobSummaryResponse,
    JobVersionResponse,
    RelevanceScoreResponse,
    SearchRunCreatedResponse,
    SearchRunResponse,
    SourceHealthResponse,
    ValidationStatusResponse,
)
from job_discovery.health.source_health import SourceHealthMonitor
from job_discovery.ranking.relevance_scorer import RelevanceScorer
from job_discovery.repositories.job_repository import JobRepository
from job_discovery.repositories.search_repository import SearchRepository
from job_discovery.search.query_builder import SearchCriteria

logger = logging.getLogger(__name__)
router = APIRouter()

_health_monitor = SourceHealthMonitor(alert_threshold=0.5, min_samples=5)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _doc_to_summary(doc: Dict[str, Any], score=None) -> JobSummaryResponse:
    details = doc.get("details") or {}
    source  = doc.get("source") or {}
    comp_d  = details.get("company") or {}
    freshness = FreshnessResponse(
        first_seen_at=doc.get("first_seen_at"),
        last_seen_at=doc.get("last_seen_at"),
        last_verified_at=doc.get("last_verified_at"),
    )
    val_raw = doc.get("validation") or {}
    validation = ValidationStatusResponse(
        status=val_raw.get("status", "pending"),
        score=val_raw.get("score"),
        missing_fields=val_raw.get("missing_fields"),
        warnings=val_raw.get("warnings"),
    )
    rel_score = None
    if score is not None:
        rel_score = RelevanceScoreResponse(
            total=score.total,
            components=ComponentScoresResponse(**score.components.to_dict()),
        )
    return JobSummaryResponse(
        job_id=doc.get("canonical_url", ""),
        canonical_url=doc.get("canonical_url"),
        title=details.get("title"),
        company=comp_d.get("name"),
        location=details.get("location"),
        employment_type=details.get("employment_type"),
        work_mode=details.get("work_mode"),
        source_name=source.get("source_name"),
        lifecycle_status=doc.get("lifecycle_status"),
        version_count=doc.get("version_count"),
        freshness=freshness,
        validation=validation,
        relevance_score=rel_score,
    )


def _doc_to_detail(doc: Dict[str, Any], score=None) -> JobDetailResponse:
    summary  = _doc_to_summary(doc, score)
    details  = doc.get("details") or {}
    reqs     = details.get("requirements") or {}
    desc     = details.get("description") or {}
    comp_doc = details.get("compensation") or {}
    app_doc  = details.get("application") or {}
    return JobDetailResponse(
        **summary.model_dump(),
        description_summary=desc.get("summary"),
        required_skills=reqs.get("required_skills"),
        preferred_skills=reqs.get("preferred_skills"),
        salary_raw=comp_doc.get("raw_text"),
        experience_raw=None,
        apply_url=app_doc.get("apply_url"),
        apply_email=app_doc.get("apply_email"),
        seniority=details.get("seniority"),
        content_hash=doc.get("content_hash"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Status
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/status", tags=["meta"])
async def status() -> dict:
    return {"service": "job_discovery", "version": "1.0.0"}


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/jobs/search
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/jobs/search", response_model=SearchRunCreatedResponse, tags=["search"])
async def trigger_search(
    request: JobSearchRequest,
    background_tasks: BackgroundTasks,
    search_repo: SearchRepository = Depends(get_search_repo),
) -> SearchRunCreatedResponse:
    """Trigger a new search run. Returns run_id immediately; pipeline runs in background."""
    from job_discovery.orchestrator.run_tracker import generate_run_id
    from job_discovery.models.search import SearchRun, RunStatus

    run_id = generate_run_id()
    run = SearchRun(run_id=run_id, status=RunStatus.pending)
    try:
        search_repo.runs.insert(run)
    except Exception as exc:
        logger.error("Failed to create search run: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create search run")

    # Build SearchCriteria from request
    from job_discovery.search.query_builder import SearchCriteria
    criteria = SearchCriteria(
        title=request.title,
        keywords=request.keywords,
        location=request.location,
        employment_type=request.employment_type,
        workplace_type=request.workplace_type,
        experience_years_min=request.experience_years_min,
        experience_years_max=request.experience_years_max,
        preferred_sources=request.preferred_sources,
    )

    background_tasks.add_task(_run_search_pipeline, run_id, criteria)

    return SearchRunCreatedResponse(
        run_id=run_id,
        status=RunStatus.pending.value,
        message="Search run created. Poll GET /api/search-runs/{run_id} for status.",
    )


def _run_search_pipeline(run_id: str, criteria) -> None:
    """Background task: execute the full search orchestrator pipeline."""
    import asyncio
    import httpx
    from job_discovery.db import get_db
    from job_discovery.repositories.job_repository import JobRepository
    from job_discovery.repositories.search_repository import SearchRepository
    from job_discovery.search.providers.web_search import WebSearchProvider
    from job_discovery.orchestrator.search_orchestrator import SearchOrchestrator
    from job_discovery.storage.raw_store import RawStore
    from config import get_settings

    settings = get_settings()
    db = get_db()
    job_repo = JobRepository(db)
    search_repo = SearchRepository(db)
    raw_store = RawStore(base_dir=settings.export_output_path.replace("exports", "raw_store"))

    http_client = httpx.Client(timeout=float(settings.request_timeout))
    provider = WebSearchProvider(http_client=http_client, search_engine="ddg", request_delay=1.5)

    async def _run():
        async with httpx.AsyncClient(timeout=float(settings.request_timeout)) as async_client:
            orchestrator = SearchOrchestrator(
                search_provider=provider,
                job_repo=job_repo,
                search_repo=search_repo,
                raw_store=raw_store,
                http_client=async_client,
            )
            await orchestrator.run(criteria, run_id=run_id)

    try:
        asyncio.run(_run())
    except Exception as exc:
        logger.error("Background search pipeline failed for run %s: %s", run_id, exc)


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/jobs
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/jobs", response_model=List[JobSummaryResponse], tags=["jobs"])
async def list_jobs(
    sort:                 str           = Query("relevance", pattern="^(relevance|newest|posted_date)$"),
    title:                Optional[str] = Query(None),
    location:             Optional[str] = Query(None),
    remote_ok:            bool          = Query(False),
    workplace_type:       Optional[str] = Query(None),
    employment_type:      Optional[str] = Query(None),
    experience_years_min: Optional[int] = Query(None),
    experience_years_max: Optional[int] = Query(None),
    lifecycle_status:     Optional[str] = Query("active"),
    limit:                int           = Query(50, ge=1, le=500),
    skip:                 int           = Query(0, ge=0),
    job_repo:  JobRepository   = Depends(get_job_repo),
    scorer:    RelevanceScorer = Depends(get_scorer),
) -> List[JobSummaryResponse]:
    from pymongo import DESCENDING
    mongo_filter: Dict[str, Any] = {}
    if lifecycle_status:
        mongo_filter["lifecycle_status"] = lifecycle_status
    try:
        cursor = job_repo._jobs.find(
            mongo_filter, {"retrieval.raw_content_path": 0}, limit=limit, skip=skip,
        )
        if sort == "newest":
            cursor = cursor.sort("last_seen_at", DESCENDING)
        elif sort == "posted_date":
            cursor = cursor.sort("details.posted_at", DESCENDING)
        else:
            cursor = cursor.sort("last_seen_at", DESCENDING)
        docs = list(cursor)
    except Exception as exc:
        logger.error("list_jobs DB error: %s", exc)
        raise HTTPException(status_code=500, detail="Database error")

    criteria = SearchCriteria(
        title=title, location=location, remote_ok=remote_ok,
        workplace_type=workplace_type, employment_type=employment_type,
        experience_years_min=experience_years_min,
        experience_years_max=experience_years_max,
    )
    has_criteria = bool(title or location or workplace_type or employment_type)
    if sort == "relevance" and has_criteria:
        ranked = scorer.rank(docs, criteria)
        return [_doc_to_summary(doc, sc) for doc, sc in ranked]
    return [_doc_to_summary(doc) for doc in docs]


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/jobs/detail
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/jobs/detail", response_model=JobDetailResponse, tags=["jobs"])
async def get_job(
    canonical_url: str = Query(...),
    job_repo: JobRepository = Depends(get_job_repo),
) -> JobDetailResponse:
    doc = job_repo.find_by_canonical_url(canonical_url)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {canonical_url!r}")
    return _doc_to_detail(doc)


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/jobs/versions
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/jobs/versions", response_model=List[JobVersionResponse], tags=["jobs"])
async def get_job_versions(
    canonical_url: str = Query(...),
    job_repo: JobRepository = Depends(get_job_repo),
) -> List[JobVersionResponse]:
    versions = job_repo.get_versions(canonical_url, limit=50)
    return [
        JobVersionResponse(
            version_number=v.get("version_number", 0),
            content_hash=v.get("content_hash", ""),
            retrieved_at=v.get("retrieved_at") or v.get("versioned_at"),
            changed_fields=(v.get("changes") or {}).get("changed_fields"),
        )
        for v in versions
    ]


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/jobs/changes
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/jobs/changes", response_model=List[ChangeResponse], tags=["jobs"])
async def get_job_changes(
    canonical_url: str = Query(...),
    job_repo: JobRepository = Depends(get_job_repo),
) -> List[ChangeResponse]:
    versions = job_repo.get_versions(canonical_url, limit=50)
    changes: List[ChangeResponse] = []
    for v in versions:
        for detail in (v.get("changes") or {}).get("details") or []:
            changes.append(ChangeResponse(
                version_number=v.get("version_number", 0),
                field_name=detail.get("field", ""),
                before=detail.get("before"),
                after=detail.get("after"),
            ))
    return changes


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/search-runs/{run_id}
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/search-runs/{run_id}", response_model=SearchRunResponse, tags=["search"])
async def get_search_run(
    run_id: str,
    search_repo: SearchRepository = Depends(get_search_repo),
) -> SearchRunResponse:
    doc = search_repo.runs.find_by_run_id(run_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Search run not found: {run_id!r}")
    return SearchRunResponse(
        run_id=doc.get("run_id", ""),
        status=doc.get("status", "unknown"),
        started_at=doc.get("started_at"),
        completed_at=doc.get("completed_at"),
        queries_generated=doc.get("queries_generated", 0),
        sources_searched=doc.get("sources_searched", 0),
        urls_discovered=doc.get("urls_discovered", 0),
        urls_fetched=doc.get("urls_fetched", 0),
        failed_fetches=doc.get("failed_fetches", 0),
        extraction_failures=doc.get("extraction_failures", 0),
        duplicates_skipped=doc.get("duplicates_skipped", 0),
        unique_jobs=doc.get("unique_jobs", 0),
        execution_duration_s=doc.get("execution_duration_s"),
        errors=doc.get("errors"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/sources/health
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/sources/health", response_model=List[SourceHealthResponse], tags=["health"])
async def get_sources_health() -> List[SourceHealthResponse]:
    all_metrics = _health_monitor.all_metrics()
    return [
        SourceHealthResponse(
            source_name=source_name,
            total_jobs=m.total_jobs,
            successful_extractions=m.successful_extractions,
            failed_extractions=m.failed_extractions,
            extraction_rate=round(m.extraction_rate(), 3),
            field_rates={fn: round(m.get_field(fn).extraction_rate, 3) for fn in m.field_metrics},
            last_updated=m.last_updated,
        )
        for source_name, m in all_metrics.items()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/jobs/export
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/jobs/export", response_model=ExportResponse, tags=["export"])
async def export_jobs_endpoint(
    source:             Optional[str] = Query(None, description="Filter by source name"),
    location:           Optional[str] = Query(None, description="Filter by location (regex)"),
    posted_within_days: Optional[int] = Query(None, description="Jobs seen within N days"),
    exp_min:            Optional[int] = Query(None, description="Min experience years"),
    exp_max:            Optional[int] = Query(None, description="Max experience years"),
    overwrite_target:   bool          = Query(False, description="Back up and overwrite existing"),
    output_path:        Optional[str] = Query(None, description="Override output directory"),
    job_repo: JobRepository = Depends(get_job_repo),
) -> ExportResponse:
    """Export active jobs to a timestamped XLSX workbook."""
    from job_discovery.export.run_export import export_jobs as _do_export
    try:
        path = _do_export(
            source=source,
            location=location,
            posted_within_days=posted_within_days,
            exp_min=exp_min,
            exp_max=exp_max,
            output_path=output_path,
            overwrite=overwrite_target,
            db=job_repo._db,
        )
        count = job_repo.count({"lifecycle_status": {"$in": ["active", "unknown"]}})
        return ExportResponse(
            message=f"Export complete: {path}",
            run_id=None,
            job_count=count,
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        logger.error("Export failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
