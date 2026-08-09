"""
Repository for search-related collections:
  - search_runs
  - search_results
  - source_fetches
  - extraction_events
  - query_cache
  - source_health

Uses PyMongo directly — no ODM.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.results import InsertOneResult, UpdateResult

from job_discovery.models.search import SearchRun, SearchResult, UrlState, UrlProcessingStatus
from job_discovery.models.source import (
    ExtractionEvent,
    QueryCache,
    SourceFetch,
    SourceHealth,
)

# Collection name constants
SEARCH_RUNS_COLLECTION = "search_runs"
SEARCH_RESULTS_COLLECTION = "search_results"
SOURCE_FETCHES_COLLECTION = "source_fetches"
EXTRACTION_EVENTS_COLLECTION = "extraction_events"
QUERY_CACHE_COLLECTION = "query_cache"
SOURCE_HEALTH_COLLECTION = "source_health"


def _ensure_indexes(db: Database) -> None:
    """Create indexes for all search-related collections."""

    # search_runs
    sr: Collection = db[SEARCH_RUNS_COLLECTION]
    sr.create_indexes([
        IndexModel([("run_id", ASCENDING)], unique=True, name="idx_run_id"),
        IndexModel([("status", ASCENDING)], name="idx_run_status"),
        IndexModel([("started_at", DESCENDING)], name="idx_run_started_at"),
    ])

    # url_states (crash recovery)
    us: Collection = db["url_states"]
    us.create_indexes([
        IndexModel([("run_id", ASCENDING), ("url", ASCENDING)], unique=True, name="idx_us_run_url"),
        IndexModel([("run_id", ASCENDING), ("status", ASCENDING)], name="idx_us_run_status"),
        IndexModel([("canonical_url", ASCENDING)], sparse=True, name="idx_us_canonical"),
    ])

    # search_results
    sres: Collection = db[SEARCH_RESULTS_COLLECTION]
    sres.create_indexes([
        IndexModel([("run_id", ASCENDING)], name="idx_sres_run_id"),
        IndexModel([("url", ASCENDING)], name="idx_sres_url"),
        IndexModel(
            [("run_id", ASCENDING), ("url", ASCENDING)],
            unique=True,
            name="idx_sres_run_url",
        ),
    ])

    # source_fetches
    sf: Collection = db[SOURCE_FETCHES_COLLECTION]
    sf.create_indexes([
        IndexModel([("run_id", ASCENDING)], name="idx_sf_run_id"),
        IndexModel([("url", ASCENDING)], name="idx_sf_url"),
        IndexModel([("source_name", ASCENDING)], name="idx_sf_source_name"),
        IndexModel([("content_hash", ASCENDING)], sparse=True, name="idx_sf_content_hash"),
        IndexModel([("fetched_at", DESCENDING)], name="idx_sf_fetched_at"),
    ])

    # extraction_events
    ee: Collection = db[EXTRACTION_EVENTS_COLLECTION]
    ee.create_indexes([
        IndexModel([("run_id", ASCENDING)], name="idx_ee_run_id"),
        IndexModel([("url", ASCENDING)], name="idx_ee_url"),
        IndexModel([("canonical_url", ASCENDING)], sparse=True, name="idx_ee_canonical_url"),
        IndexModel([("extracted_at", DESCENDING)], name="idx_ee_extracted_at"),
    ])

    # query_cache
    qc: Collection = db[QUERY_CACHE_COLLECTION]
    qc.create_indexes([
        IndexModel([("query_hash", ASCENDING)], unique=True, name="idx_qc_query_hash"),
        IndexModel([("expires_at", ASCENDING)], name="idx_qc_expires_at"),
        IndexModel(
            [("source_name", ASCENDING), ("query_hash", ASCENDING)],
            name="idx_qc_source_hash",
        ),
    ])

    # source_health
    sh: Collection = db[SOURCE_HEALTH_COLLECTION]
    sh.create_indexes([
        IndexModel([("source_name", ASCENDING)], unique=True, name="idx_sh_source_name"),
        IndexModel([("circuit_state", ASCENDING)], name="idx_sh_circuit_state"),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# SearchRunRepository
# ─────────────────────────────────────────────────────────────────────────────

class SearchRunRepository:
    def __init__(self, db: Database) -> None:
        self._col: Collection = db[SEARCH_RUNS_COLLECTION]

    def insert(self, run: SearchRun) -> InsertOneResult:
        doc = run.model_dump(exclude_none=True)
        return self._col.insert_one(doc)

    def update_status(self, run_id: str, status: str, **kwargs: Any) -> UpdateResult:
        fields: Dict[str, Any] = {"status": status, **kwargs}
        return self._col.update_one({"run_id": run_id}, {"$set": fields})

    def find_by_run_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self._col.find_one({"run_id": run_id})

    def find_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(self._col.find({}, limit=limit).sort("started_at", DESCENDING))


# ─────────────────────────────────────────────────────────────────────────────
# SearchResultRepository
# ─────────────────────────────────────────────────────────────────────────────

class SearchResultRepository:
    def __init__(self, db: Database) -> None:
        self._col: Collection = db[SEARCH_RESULTS_COLLECTION]

    def insert(self, result: SearchResult) -> InsertOneResult:
        return self._col.insert_one(result.model_dump(exclude_none=True))

    def insert_many(self, results: List[SearchResult]) -> None:
        if results:
            self._col.insert_many(
                [r.model_dump(exclude_none=True) for r in results],
                ordered=False,
            )

    def find_by_run_id(self, run_id: str) -> List[Dict[str, Any]]:
        return list(self._col.find({"run_id": run_id}))

    def mark_fetched(self, run_id: str, url: str) -> UpdateResult:
        return self._col.update_one(
            {"run_id": run_id, "url": url},
            {"$set": {"fetched": True}},
        )


# ─────────────────────────────────────────────────────────────────────────────
# SourceFetchRepository
# ─────────────────────────────────────────────────────────────────────────────

class SourceFetchRepository:
    def __init__(self, db: Database) -> None:
        self._col: Collection = db[SOURCE_FETCHES_COLLECTION]

    def insert(self, fetch: SourceFetch) -> InsertOneResult:
        return self._col.insert_one(fetch.model_dump(exclude_none=True))

    def find_by_run_id(self, run_id: str) -> List[Dict[str, Any]]:
        return list(self._col.find({"run_id": run_id}))

    def find_by_url(self, url: str, limit: int = 10) -> List[Dict[str, Any]]:
        return list(
            self._col.find({"url": url}, limit=limit).sort("fetched_at", DESCENDING)
        )


# ─────────────────────────────────────────────────────────────────────────────
# ExtractionEventRepository
# ─────────────────────────────────────────────────────────────────────────────

class ExtractionEventRepository:
    def __init__(self, db: Database) -> None:
        self._col: Collection = db[EXTRACTION_EVENTS_COLLECTION]

    def insert(self, event: ExtractionEvent) -> InsertOneResult:
        return self._col.insert_one(event.model_dump(exclude_none=True))

    def find_by_run_id(self, run_id: str) -> List[Dict[str, Any]]:
        return list(self._col.find({"run_id": run_id}))

    def find_by_canonical_url(self, canonical_url: str) -> List[Dict[str, Any]]:
        return list(
            self._col.find({"canonical_url": canonical_url}).sort(
                "extracted_at", DESCENDING
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# QueryCacheRepository
# ─────────────────────────────────────────────────────────────────────────────

class QueryCacheRepository:
    def __init__(self, db: Database) -> None:
        self._col: Collection = db[QUERY_CACHE_COLLECTION]

    def get(self, query_hash: str) -> Optional[Dict[str, Any]]:
        return self._col.find_one({"query_hash": query_hash})

    def upsert(self, entry: QueryCache) -> UpdateResult:
        doc = entry.model_dump(exclude_none=True)
        return self._col.update_one(
            {"query_hash": entry.query_hash},
            {"$set": doc},
            upsert=True,
        )

    def increment_hit(self, query_hash: str) -> UpdateResult:
        return self._col.update_one(
            {"query_hash": query_hash},
            {"$inc": {"hit_count": 1}},
        )

    def delete_expired(self) -> int:
        result = self._col.delete_many({"expires_at": {"$lt": datetime.now(timezone.utc)}})
        return result.deleted_count


# ─────────────────────────────────────────────────────────────────────────────
# SourceHealthRepository
# ─────────────────────────────────────────────────────────────────────────────

class SourceHealthRepository:
    def __init__(self, db: Database) -> None:
        self._col: Collection = db[SOURCE_HEALTH_COLLECTION]

    def upsert(self, health: SourceHealth) -> UpdateResult:
        doc = health.model_dump(exclude_none=True)
        doc["updated_at"] = datetime.now(timezone.utc)
        return self._col.update_one(
            {"source_name": health.source_name},
            {"$set": doc},
            upsert=True,
        )

    def find_by_source(self, source_name: str) -> Optional[Dict[str, Any]]:
        return self._col.find_one({"source_name": source_name})

    def find_open_circuits(self) -> List[Dict[str, Any]]:
        return list(self._col.find({"circuit_state": "open"}))

    def all(self) -> List[Dict[str, Any]]:
        return list(self._col.find({}))


# ─────────────────────────────────────────────────────────────────────────────
# UrlStateRepository  — crash recovery / incremental processing
# ─────────────────────────────────────────────────────────────────────────────

class UrlStateRepository:
    """Persists per-URL processing state for crash recovery."""

    URL_STATES_COLLECTION = "url_states"

    def __init__(self, db: Database) -> None:
        self._col: Collection = db[self.URL_STATES_COLLECTION]

    def upsert(self, state: UrlState) -> UpdateResult:
        doc = state.model_dump(exclude_none=True)
        doc["updated_at"] = datetime.now(timezone.utc)
        return self._col.update_one(
            {"run_id": state.run_id, "url": state.url},
            {"$set": doc},
            upsert=True,
        )

    def update_status(self, run_id: str, url: str, status: str, **extra) -> UpdateResult:
        fields = {"status": status, "updated_at": datetime.now(timezone.utc), **extra}
        return self._col.update_one(
            {"run_id": run_id, "url": url},
            {"$set": fields},
        )

    def find_by_run_id(self, run_id: str) -> List[Dict[str, Any]]:
        return list(self._col.find({"run_id": run_id}))

    def find_by_status(self, run_id: str, status: str) -> List[Dict[str, Any]]:
        return list(self._col.find({"run_id": run_id, "status": status}))

    def get(self, run_id: str, url: str) -> Optional[Dict[str, Any]]:
        return self._col.find_one({"run_id": run_id, "url": url})

    def get_processed_urls(self, run_id: str) -> set:
        """Return set of URLs already processed (fetched/stored/skipped)."""
        terminal = {
            UrlProcessingStatus.fetched.value,
            UrlProcessingStatus.stored.value,
            UrlProcessingStatus.duplicate.value,
            UrlProcessingStatus.fetch_skipped.value,
            UrlProcessingStatus.extraction_done.value,
        }
        docs = self._col.find(
            {"run_id": run_id, "status": {"$in": list(terminal)}},
            {"url": 1},
        )
        return {d["url"] for d in docs}


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: initialise all indexes in one call
# ─────────────────────────────────────────────────────────────────────────────

class SearchRepository:
    """
    Facade that holds all search-domain repositories and ensures
    indexes are created once on construction.
    """

    def __init__(self, db: Database) -> None:
        _ensure_indexes(db)
        self.runs = SearchRunRepository(db)
        self.results = SearchResultRepository(db)
        self.fetches = SourceFetchRepository(db)
        self.extractions = ExtractionEventRepository(db)
        self.query_cache = QueryCacheRepository(db)
        self.health = SourceHealthRepository(db)
        self.url_states = UrlStateRepository(db)
