"""
Tests for RawStore and content hashing.
Uses pytest's tmp_path fixture — no real job sites, no persistent files.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from job_discovery.storage.raw_store import RawStore, compute_content_hash


# ─────────────────────────────────────────────────────────────────────────────
# Content hashing
# ─────────────────────────────────────────────────────────────────────────────

class TestContentHash:
    def test_deterministic_bytes(self):
        h1 = compute_content_hash(b"<html>hello</html>")
        h2 = compute_content_hash(b"<html>hello</html>")
        assert h1 == h2

    def test_deterministic_str(self):
        h1 = compute_content_hash("<html>hello</html>")
        h2 = compute_content_hash("<html>hello</html>")
        assert h1 == h2

    def test_bytes_and_str_equal(self):
        content = "<html>hello</html>"
        assert compute_content_hash(content) == compute_content_hash(content.encode())

    def test_different_content_different_hash(self):
        h1 = compute_content_hash("<html>job A</html>")
        h2 = compute_content_hash("<html>job B</html>")
        assert h1 != h2

    def test_whitespace_normalised(self):
        """Leading/trailing whitespace and CRLF differences produce same hash."""
        h1 = compute_content_hash("  <html>hello</html>  ")
        h2 = compute_content_hash("<html>hello</html>")
        assert h1 == h2

    def test_crlf_normalised(self):
        h1 = compute_content_hash("<html>line1\r\nline2</html>")
        h2 = compute_content_hash("<html>line1\nline2</html>")
        assert h1 == h2

    def test_hash_is_64_hex_chars(self):
        h = compute_content_hash("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_empty_content(self):
        h = compute_content_hash("")
        assert len(h) == 64

    def test_binary_bytes(self):
        h = compute_content_hash(b"\x00\x01\x02")
        assert len(h) == 64


# ─────────────────────────────────────────────────────────────────────────────
# RawStore — save and load
# ─────────────────────────────────────────────────────────────────────────────

class TestRawStore:
    def test_save_creates_directory(self, tmp_path):
        store = RawStore(tmp_path)
        store.save(
            source="linkedin",
            job_key="abc123",
            raw_html=b"<html>test</html>",
        )
        slot = tmp_path / "linkedin" / "abc123"
        assert slot.is_dir()

    def test_save_creates_raw_html(self, tmp_path):
        store = RawStore(tmp_path)
        store.save(source="indeed", job_key="xyz", raw_html=b"<html>job</html>")
        assert (tmp_path / "indeed" / "xyz" / "raw.html").exists()

    def test_save_creates_raw_txt(self, tmp_path):
        store = RawStore(tmp_path)
        store.save(source="naukri", job_key="k1", raw_html=b"<html/>", raw_text="Job text")
        txt = (tmp_path / "naukri" / "k1" / "raw.txt").read_text()
        assert txt == "Job text"

    def test_save_creates_metadata_json(self, tmp_path):
        store = RawStore(tmp_path)
        store.save(
            source="linkedin",
            job_key="m1",
            raw_html=b"<html/>",
            metadata={"url": "https://linkedin.com/jobs/view/1"},
        )
        meta_path = tmp_path / "linkedin" / "m1" / "metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["url"] == "https://linkedin.com/jobs/view/1"

    def test_save_creates_retrieval_json(self, tmp_path):
        store = RawStore(tmp_path)
        store.save(
            source="linkedin",
            job_key="r1",
            raw_html=b"<html/>",
            retrieval={"retry_count": 0},
        )
        ret_path = tmp_path / "linkedin" / "r1" / "retrieval.json"
        assert ret_path.exists()
        ret = json.loads(ret_path.read_text())
        assert ret["retry_count"] == 0

    def test_load_html(self, tmp_path):
        store = RawStore(tmp_path)
        content = b"<html>loaded content</html>"
        store.save(source="indeed", job_key="load1", raw_html=content)
        loaded = store.load_html("indeed", "load1")
        assert loaded == content

    def test_load_html_missing_returns_none(self, tmp_path):
        store = RawStore(tmp_path)
        assert store.load_html("linkedin", "nonexistent") is None

    def test_load_text(self, tmp_path):
        store = RawStore(tmp_path)
        store.save(source="naukri", job_key="t1", raw_html=b"<html/>", raw_text="hello")
        assert store.load_text("naukri", "t1") == "hello"

    def test_load_text_missing_returns_none(self, tmp_path):
        store = RawStore(tmp_path)
        assert store.load_text("naukri", "missing") is None

    def test_load_metadata(self, tmp_path):
        store = RawStore(tmp_path)
        store.save(source="s", job_key="k", raw_html=b"x", metadata={"key": "val"})
        meta = store.load_metadata("s", "k")
        assert meta["key"] == "val"

    def test_exists_false_before_save(self, tmp_path):
        store = RawStore(tmp_path)
        assert not store.exists("linkedin", "notyet")

    def test_exists_true_after_save(self, tmp_path):
        store = RawStore(tmp_path)
        store.save(source="linkedin", job_key="e1", raw_html=b"<html/>")
        assert store.exists("linkedin", "e1")

    def test_slot_path(self, tmp_path):
        store = RawStore(tmp_path)
        p = store.slot_path("indeed", "j1")
        assert str(p) == str(tmp_path / "indeed" / "j1")

    def test_save_string_html(self, tmp_path):
        """String HTML should be accepted and stored as UTF-8 bytes."""
        store = RawStore(tmp_path)
        store.save(source="hirist", job_key="s1", raw_html="<html>string</html>")
        loaded = store.load_html("hirist", "s1")
        assert loaded == b"<html>string</html>"

    def test_save_returns_path(self, tmp_path):
        store = RawStore(tmp_path)
        path = store.save(source="cutshort", job_key="p1", raw_html=b"x")
        assert isinstance(path, Path)
        assert path.is_dir()

    def test_path_sanitisation(self, tmp_path):
        """Source/job_key with unsafe chars should be sanitised."""
        store = RawStore(tmp_path)
        store.save(source="my/source", job_key="job/../hack", raw_html=b"x")
        # Should not create directories outside base
        assert not (tmp_path / "hack").exists()

    def test_metadata_has_saved_at(self, tmp_path):
        store = RawStore(tmp_path)
        store.save(source="wellfound", job_key="ts1", raw_html=b"x")
        meta = store.load_metadata("wellfound", "ts1")
        assert "saved_at" in meta
