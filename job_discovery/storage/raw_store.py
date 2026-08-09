"""
Local filesystem raw content storage.

Layout:
  <base_dir>/
    <source>/
      <job_id_or_hash>/
        raw.html          — original HTTP response body
        raw.txt           — text-extracted version (optional, may be empty)
        metadata.json     — fetch metadata (url, status, content-type, etc.)
        retrieval.json    — retrieval tracking (timestamps, attempts, hash)

The abstraction is designed to be swappable with object storage (S3/GCS)
by replacing RawStore with a compatible implementation.

Content hash:
  SHA-256(normalize_content(raw_bytes))
  Normalisation: strip leading/trailing whitespace; \r\n → \n.
  Identical content → identical hash.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Content hashing
# ─────────────────────────────────────────────────────────────────────────────

def compute_content_hash(raw: bytes | str) -> str:
    """
    SHA-256 of normalised content.
    Normalisation: decode bytes as UTF-8 (errors=replace), strip outer
    whitespace, normalise line endings.
    """
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = raw
    normalised = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# RawStore
# ─────────────────────────────────────────────────────────────────────────────

class RawStore:
    """
    Stores and retrieves raw page content on the local filesystem.

    Args:
        base_dir: Root directory for all raw storage.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base = Path(base_dir)

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #

    def save(
        self,
        *,
        source: str,
        job_key: str,
        raw_html: bytes | str,
        raw_text: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        retrieval: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Persist raw content to <base>/<source>/<job_key>/.

        Returns the directory path where files were written.
        """
        slot = self._slot(source, job_key)
        slot.mkdir(parents=True, exist_ok=True)

        # raw.html
        html_bytes = raw_html if isinstance(raw_html, bytes) else raw_html.encode("utf-8")
        (slot / "raw.html").write_bytes(html_bytes)

        # raw.txt
        (slot / "raw.txt").write_text(raw_text or "", encoding="utf-8")

        # metadata.json
        meta = metadata or {}
        meta.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
        (slot / "metadata.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )

        # retrieval.json
        ret = retrieval or {}
        ret.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
        (slot / "retrieval.json").write_text(
            json.dumps(ret, indent=2, default=str), encoding="utf-8"
        )

        logger.debug("RawStore.save: %s", slot)
        return slot

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #

    def load_html(self, source: str, job_key: str) -> Optional[bytes]:
        """Return raw HTML bytes, or None if not found."""
        path = self._slot(source, job_key) / "raw.html"
        return path.read_bytes() if path.exists() else None

    def load_text(self, source: str, job_key: str) -> Optional[str]:
        """Return raw text, or None if not found."""
        path = self._slot(source, job_key) / "raw.txt"
        return path.read_text(encoding="utf-8") if path.exists() else None

    def load_metadata(self, source: str, job_key: str) -> Optional[Dict[str, Any]]:
        path = self._slot(source, job_key) / "metadata.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def load_retrieval(self, source: str, job_key: str) -> Optional[Dict[str, Any]]:
        path = self._slot(source, job_key) / "retrieval.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def slot_path(self, source: str, job_key: str) -> Path:
        """Return the directory path for a slot (may not exist yet)."""
        return self._slot(source, job_key)

    def exists(self, source: str, job_key: str) -> bool:
        return (self._slot(source, job_key) / "raw.html").exists()

    def _slot(self, source: str, job_key: str) -> Path:
        # Sanitise both components to avoid path traversal
        safe_source = _safe_name(source)
        safe_key = _safe_name(job_key)
        return self._base / safe_source / safe_key


def _safe_name(name: str) -> str:
    """Replace characters that are unsafe in directory names."""
    import re
    return re.sub(r"[^\w\-.]", "_", name)[:128]
