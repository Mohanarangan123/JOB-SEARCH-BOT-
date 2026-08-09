"""
FastAPI application entrypoint for the Job Discovery System.
"""
from __future__ import annotations

from fastapi import FastAPI

from job_discovery.api.routes import router

app = FastAPI(
    title="Job Discovery System",
    version="1.0.0",
    description="Automated job discovery, extraction, normalization, and ranking API.",
)

app.include_router(router, prefix="/api")


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    return {"status": "ok"}
