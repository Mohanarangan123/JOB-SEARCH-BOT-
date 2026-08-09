"""
run_export.py — CLI entrypoint for Excel export.

Usage:
    python -m job_discovery.export.run_export [options]

Options:
    --source SOURCE        Filter by source name (e.g. naukri, linkedin)
    --location LOCATION    Filter by location substring
    --posted-within DAYS   Only jobs seen within N days
    --exp-min YEARS        Minimum experience years
    --exp-max YEARS        Maximum experience years
    --output PATH          Override output path
    --overwrite            Allow overwriting (backs up existing file first)

Environment variables:
    EXPORT_OUTPUT_PATH     Default output directory (default: ./exports)
    MONGODB_URI            MongoDB connection string
    MONGODB_DATABASE       MongoDB database name
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Filename helpers
# ─────────────────────────────────────────────────────────────────────────────

def _timestamped_filename() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    return f"job_listings_{ts}.xlsx"


def _backup_path(existing: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    return existing.parent / f"{existing.stem}.bak-{ts}{existing.suffix}"


def _resolve_output_path(output_override: Optional[str] = None) -> Path:
    from config import get_settings
    settings = get_settings()
    base = Path(output_override or settings.export_output_path)
    return base / _timestamped_filename()


# ─────────────────────────────────────────────────────────────────────────────
# Mongo filter builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_filter(
    source: Optional[str] = None,
    location: Optional[str] = None,
    posted_within_days: Optional[int] = None,
    exp_min: Optional[int] = None,
    exp_max: Optional[int] = None,
) -> Dict[str, Any]:
    f: Dict[str, Any] = {"lifecycle_status": {"$in": ["active", "unknown"]}}

    if source:
        f["source.source_name"] = source.lower().strip()

    if location:
        f["details.location"] = {"$regex": location, "$options": "i"}

    if posted_within_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=posted_within_days)
        f["last_seen_at"] = {"$gte": cutoff}

    if exp_min is not None:
        f["details.requirements.experience_years_max"] = {"$gte": exp_min}

    if exp_max is not None:
        f["details.requirements.experience_years_min"] = {"$lte": exp_max}

    return f


# ─────────────────────────────────────────────────────────────────────────────
# Main export function (importable + CLI)
# ─────────────────────────────────────────────────────────────────────────────

def export_jobs(
    *,
    source:             Optional[str]  = None,
    location:           Optional[str]  = None,
    posted_within_days: Optional[int]  = None,
    exp_min:            Optional[int]  = None,
    exp_max:            Optional[int]  = None,
    output_path:        Optional[str]  = None,
    overwrite:          bool           = False,
    run_id:             Optional[str]  = None,
    db=None,    # Optional injected DB (for tests / API use)
) -> Path:
    """
    Export jobs to an XLSX file.

    Returns the path of the written file.
    Raises FileExistsError if output already exists and overwrite=False.
    """
    from job_discovery.export.xlsx_writer import XlsxWriter

    # Resolve output path
    out = _resolve_output_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Guard against silent overwrite
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"Export file already exists: {out}. "
            "Use overwrite=True to replace it."
        )
    if out.exists() and overwrite:
        bak = _backup_path(out)
        shutil.copy2(str(out), str(bak))
        logger.info("Backed up existing export to %s", bak)

    # Connect to MongoDB
    if db is None:
        from job_discovery.db import get_db
        db = get_db()

    # Build filter and fetch
    mongo_filter = _build_filter(
        source=source,
        location=location,
        posted_within_days=posted_within_days,
        exp_min=exp_min,
        exp_max=exp_max,
    )

    # Log active filters
    active_filters = {k: v for k, v in {
        "source": source,
        "location": location,
        "posted_within_days": posted_within_days,
        "exp_min": exp_min,
        "exp_max": exp_max,
    }.items() if v is not None}
    if active_filters:
        logger.info("Export filters: %s", active_filters)

    jobs = list(db["jobs"].find(mongo_filter, {"retrieval.raw_content_path": 0}))
    row_count = len(jobs)

    if row_count == 0:
        logger.warning(
            "Export produced ZERO results. Filters: %s. Writing empty file.",
            active_filters or "none",
        )
    else:
        logger.info("Exporting %d jobs to %s", row_count, out)

    writer = XlsxWriter()
    writer.write(jobs, out, run_id=run_id)

    logger.info("Export complete: %d rows → %s", row_count, out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export job listings to Excel (.xlsx)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source",          default=None, help="Filter by source name")
    p.add_argument("--location",        default=None, help="Filter by location (regex)")
    p.add_argument("--posted-within",   type=int, default=None, dest="posted_within",
                   metavar="DAYS",      help="Only jobs seen within N days")
    p.add_argument("--exp-min",         type=int, default=None, dest="exp_min",
                   metavar="YEARS",     help="Minimum experience years")
    p.add_argument("--exp-max",         type=int, default=None, dest="exp_max",
                   metavar="YEARS",     help="Maximum experience years")
    p.add_argument("--output",          default=None, help="Override output directory")
    p.add_argument("--overwrite",       action="store_true",
                   help="Overwrite existing export (backs up first)")
    p.add_argument("--run-id",          default=None, dest="run_id",
                   help="Embed a search run ID in the export")
    p.add_argument("-v", "--verbose",   action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        path = export_jobs(
            source=args.source,
            location=args.location,
            posted_within_days=args.posted_within,
            exp_min=args.exp_min,
            exp_max=args.exp_max,
            output_path=args.output,
            overwrite=args.overwrite,
            run_id=args.run_id,
        )
        print(f"Exported to: {path}")
        return 0
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Export failed: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
