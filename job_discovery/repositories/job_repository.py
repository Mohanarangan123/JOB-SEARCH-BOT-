"""
Repository for the `jobs` and `job_versions` collections.

Uses PyMongo directly — no ODM.
All public methods accept / return plain dicts or Pydantic models.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.results import InsertOneResult, UpdateResult

from job_discovery.models.job import JobRecord, LifecycleStatus


# Collection names
JOBS_COLLECTION = "jobs"
JOB_VERSIONS_COLLECTION = "job_versions"


_indexes_created: bool = False


def _ensure_indexes(db: Database) -> None:
    """Create all indexes once per process lifetime."""
    global _indexes_created
    if _indexes_created:
        return
    _indexes_created = True
    jobs: Collection = db[JOBS_COLLECTION]
    jobs.create_indexes([
        # Primary deduplication keys
        IndexModel([("canonical_url", ASCENDING)], unique=True, sparse=True, name="idx_canonical_url"),
        IndexModel([("external_job_id", ASCENDING)], sparse=True, name="idx_external_job_id"),
        IndexModel([("content_hash", ASCENDING)], sparse=True, name="idx_content_hash"),
        # Lifecycle filtering
        IndexModel([("lifecycle_status", ASCENDING)], name="idx_lifecycle_status"),
        # Time-series queries
        IndexModel([("first_seen_at", DESCENDING)], name="idx_first_seen_at"),
        IndexModel([("last_seen_at", DESCENDING)], name="idx_last_seen_at"),
        IndexModel([("last_verified_at", DESCENDING)], name="idx_last_verified_at"),
        # Source lookups
        IndexModel([("source.source_name", ASCENDING)], name="idx_source_name"),
        # Compound: active jobs by source
        IndexModel(
            [("lifecycle_status", ASCENDING), ("source.source_name", ASCENDING)],
            name="idx_status_source",
        ),
    ])

    versions: Collection = db[JOB_VERSIONS_COLLECTION]
    versions.create_indexes([
        IndexModel([("canonical_url", ASCENDING)], name="idx_ver_canonical_url"),
        IndexModel([("content_hash", ASCENDING)], name="idx_ver_content_hash"),
        IndexModel([("versioned_at", DESCENDING)], name="idx_ver_versioned_at"),
    ])


class JobRepository:
    """CRUD and upsert operations for the jobs collection."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._jobs: Collection = db[JOBS_COLLECTION]
        self._versions: Collection = db[JOB_VERSIONS_COLLECTION]
        _ensure_indexes(db)

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #

    def upsert_by_canonical_url(
        self,
        record: JobRecord,
    ) -> UpdateResult:
        """
        Insert or update a job by canonical_url.
        Always sets updated_at; preserves first_seen_at on update.
        """
        doc = record.model_dump(exclude_none=True)
        doc["updated_at"] = datetime.now(timezone.utc)

        result = self._jobs.update_one(
            {"canonical_url": record.canonical_url},
            {
                "$set": doc,
                "$setOnInsert": {"first_seen_at": doc.get("first_seen_at", datetime.now(timezone.utc))},
            },
            upsert=True,
        )
        return result

    def insert_one(self, record: JobRecord) -> InsertOneResult:
        doc = record.model_dump(exclude_none=True)
        doc.setdefault("created_at", datetime.now(timezone.utc))
        doc.setdefault("updated_at", datetime.now(timezone.utc))
        return self._jobs.insert_one(doc)

    def save_version(self, canonical_url: str, snapshot: Dict[str, Any]) -> None:
        """Append a versioned snapshot of a job document."""
        snapshot["canonical_url"] = canonical_url
        snapshot["versioned_at"] = datetime.now(timezone.utc)
        self._versions.insert_one(snapshot)

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #

    def find_by_canonical_url(self, canonical_url: str) -> Optional[Dict[str, Any]]:
        return self._jobs.find_one({"canonical_url": canonical_url})

    def find_by_content_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        return self._jobs.find_one({"content_hash": content_hash})

    def find_by_external_id(
        self,
        external_job_id: str,
        source_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        query: Dict[str, Any] = {"external_job_id": external_job_id}
        if source_name:
            query["source.source_name"] = source_name
        return self._jobs.find_one(query)

    def find_active(self, limit: int = 100) -> List[Dict[str, Any]]:
        cursor = self._jobs.find(
            {"lifecycle_status": LifecycleStatus.active},
            limit=limit,
        ).sort("last_seen_at", DESCENDING)
        return list(cursor)

    def count(self, filter_: Optional[Dict[str, Any]] = None) -> int:
        return self._jobs.count_documents(filter_ or {})

    # ------------------------------------------------------------------ #
    # Update lifecycle
    # ------------------------------------------------------------------ #

    def mark_removed(self, canonical_url: str) -> UpdateResult:
        return self._jobs.update_one(
            {"canonical_url": canonical_url},
            {
                "$set": {
                    "lifecycle_status": LifecycleStatus.removed,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

    def increment_fetch_failure(self, canonical_url: str) -> UpdateResult:
        return self._jobs.update_one(
            {"canonical_url": canonical_url},
            {
                "$inc": {"retrieval.consecutive_fetch_failures": 1},
                "$set": {"updated_at": datetime.now(timezone.utc)},
            },
        )

    # ------------------------------------------------------------------ #
    # Versions read
    # ------------------------------------------------------------------ #

    def get_versions(
        self,
        canonical_url: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        cursor = self._versions.find(
            {"canonical_url": canonical_url},
            limit=limit,
        ).sort("versioned_at", DESCENDING)
        return list(cursor)
