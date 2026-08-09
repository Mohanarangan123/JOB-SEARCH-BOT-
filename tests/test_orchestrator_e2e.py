"""
Mocked end-to-end orchestration test.
Kept in a separate file so it can be run independently.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

try:
    import mongomock
    _MONGOMOCK = True
except ImportError:
    _MONGOMOCK = False


@pytest.fixture
def db():
    if not _MONGOMOCK:
        pytest.skip("mongomock not installed")
    return mongomock.MongoClient()["test_orch_e2e"]


@pytest.mark.asyncio
async def test_orchestrator_mocked_run(db, tmp_path):
    """
    Full pipeline with mocked search provider and HTTP client.
    Verifies: search → URL dedup → fetch → store → MongoDB upsert.
    """
    from job_discovery.repositories.job_repository import JobRepository
    from job_discovery.repositories.search_repository import SearchRepository
    from job_discovery.orchestrator.search_orchestrator import SearchOrchestrator
    from job_discovery.search.providers.base import ProviderSearchResult
    from job_discovery.search.query_builder import SearchCriteria
    from job_discovery.storage.raw_store import RawStore

    jr = JobRepository(db)
    sr = SearchRepository(db)
    store = RawStore(str(tmp_path))

    fixture_html = (
        b"<html><body>"
        b"<h1>Python Developer</h1>"
        b"<p>Company: TechCorp</p>"
        b"<p>Location: Chennai</p>"
        b"<p>Employment Type: Full-Time</p>"
        b"<p>Salary: 8-14 LPA</p>"
        b"<p>Experience: 3-6 years</p>"
        b"<p>Email: jobs@techcorp.in</p>"
        b"</body></html>"
    )

    result = ProviderSearchResult(
        url="https://www.naukri.com/job-listings-python-dev-12345678",
        source_name="naukri",
        query="Python Developer Chennai",
        query_hash=ProviderSearchResult.make_query_hash("Python Developer Chennai"),
        title="Python Developer",
    )

    mock_provider = MagicMock()
    mock_provider.search.return_value = [result]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = fixture_html
    mock_resp.headers = {"content-type": "text/html"}

    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.get.return_value = mock_resp
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    orchestrator = SearchOrchestrator(
        search_provider=mock_provider,
        job_repo=jr,
        search_repo=sr,
        raw_store=store,
        http_client=mock_http,
        ollama_client=None,
    )

    criteria = SearchCriteria(title="Python Developer", location="Chennai")
    run = await orchestrator.run(criteria)

    # Verify run was tracked
    assert run.run_id.startswith("SR_")
    assert run.urls_discovered >= 1
    assert run.urls_fetched >= 1

    # Verify job stored in MongoDB
    count = jr.count()
    assert count >= 1

    stored = jr.find_by_canonical_url(
        "https://www.naukri.com/job-listings-python-dev-12345678"
    )
    assert stored is not None
    assert stored["lifecycle_status"] == "active"

    print(f"\n  run_id:          {run.run_id}")
    print(f"  urls_discovered: {run.urls_discovered}")
    print(f"  urls_fetched:    {run.urls_fetched}")
    print(f"  unique_jobs:     {run.unique_jobs}")
    print(f"  jobs in MongoDB: {count}")
