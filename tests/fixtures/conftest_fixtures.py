"""
Shared fixture helpers for extraction tests.
Import from this module in test files.
"""
from __future__ import annotations

from pathlib import Path

FIXTURE_DIR = Path(__file__).parent


def load_fixture(name: str) -> bytes:
    """Load a fixture HTML file as bytes."""
    return (FIXTURE_DIR / name).read_bytes()


def load_fixture_text(name: str) -> str:
    """Load a fixture HTML file as string."""
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")
