"""Recursive, concurrent R2 -> local directory download logic.

Used by scripts/r2-download.py. Never deletes anything from R2.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, List, Optional

from botocore.exceptions import BotoCoreError, ClientError

from .r2 import R2Client

TRANSIENT_EXCEPTIONS = (BotoCoreError, ClientError, TimeoutError, ConnectionError, OSError)


class DateRangeError(ValueError):
    """Raised for an invalid --since/--until value or an inverted range."""


@dataclass
class DownloadResult:
    source: str
    destination: str
    status: str  # downloaded | skipped | failed
    size: int = 0
    duration_ms: int = 0
    error: str = ""
    last_modified: str = ""


def _parse_date_boundary(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise DateRangeError(f"Invalid date '{value}': expected YYYY-MM-DD") from exc


def resolve_date_range(
    since: Optional[str], until: Optional[str]
) -> "tuple[Optional[datetime], Optional[datetime]]":
    """Parse --since/--until into a timezone-aware, half-open UTC range.

    Returns (since_dt, until_dt):
      - since_dt is an inclusive lower bound: that day's UTC midnight.
      - until_dt is an exclusive upper bound: the UTC midnight starting the
        day *after* `until`, so `--until 2026-08-10` includes the entirety
        of 2026-08-10.

    Raises DateRangeError if either value is not YYYY-MM-DD, or if `since`
    falls after `until`.
    """
    since_dt: Optional[datetime] = None
    until_dt: Optional[datetime] = None

    if since:
        since_day = _parse_date_boundary(since)
        since_dt = datetime(since_day.year, since_day.month, since_day.day, tzinfo=timezone.utc)

    if until:
        until_day = _parse_date_boundary(until)
        until_dt = datetime(
            until_day.year, until_day.month, until_day.day, tzinfo=timezone.utc
        ) + timedelta(days=1)

    if since_dt is not None and until_dt is not None and since_dt >= until_dt:
        raise DateRangeError(f"--since ({since}) must not be after --until ({until})")

    return since_dt, until_dt


def object_in_date_range(
    last_modified: Optional[datetime], since: Optional[datetime], until: Optional[datetime]
) -> bool:
    """True if `last_modified` falls within [since, until) (UTC, half-open).

    A missing `since`/`until` bound imposes no constraint on that side. An
    object with no `LastModified` at all is always considered in range when
    no bounds are given, but excluded once any bound is set (there is no
    timestamp to compare against).
    """
    if since is None and until is None:
        return True
    if last_modified is None:
        return False
    if last_modified.tzinfo is None:
        last_modified = last_modified.replace(tzinfo=timezone.utc)
    if since is not None and last_modified < since:
        return False
    if until is not None and last_modified >= until:
        return False
    return True


def key_to_local_path(key: str, remote_prefix: str, local_dir: Path) -> Path:
    """Map an R2 key back to a local path, preserving sub-directories."""
    relative = key[len(remote_prefix):] if key.startswith(remote_prefix) else key
    relative = relative.lstrip("/")
    return local_dir / Path(*relative.split("/")) if relative else local_dir


def _download_one(
    client: R2Client,
    key: str,
    size: int,
    last_modified: Optional[datetime],
    remote_prefix: str,
    local_dir: Path,
    force: bool,
    dry_run: bool,
    max_attempts: int,
) -> DownloadResult:
    started = time.monotonic()
    local_path = key_to_local_path(key, remote_prefix, local_dir)
    last_modified_str = last_modified.isoformat() if last_modified else ""

    if key.endswith("/"):
        return DownloadResult(
            key, str(local_path), "skipped", size, 0, "directory marker", last_modified_str
        )

    if local_path.exists() and not force:
        return DownloadResult(
            key, str(local_path), "skipped", size, 0, "already exists", last_modified_str
        )

    if dry_run:
        return DownloadResult(
            key, str(local_path), "skipped", size, 0, "dry-run", last_modified_str
        )

    attempt = 0
    while True:
        attempt += 1
        try:
            client.download_file(key, local_path)
            duration_ms = int((time.monotonic() - started) * 1000)
            return DownloadResult(
                key, str(local_path), "downloaded", size, duration_ms, "", last_modified_str
            )
        except TRANSIENT_EXCEPTIONS as exc:
            if attempt >= max_attempts:
                duration_ms = int((time.monotonic() - started) * 1000)
                return DownloadResult(
                    key,
                    str(local_path),
                    "failed",
                    size,
                    duration_ms,
                    str(exc),
                    last_modified_str,
                )
            time.sleep(min(0.25 * (2 ** attempt), 5))


def run_download(
    client: R2Client,
    remote_prefix: str,
    local_dir: Path,
    workers: int = 8,
    force: bool = False,
    dry_run: bool = False,
    max_attempts: int = 3,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    progress_cb: Optional[Callable[[DownloadResult], None]] = None,
) -> List[DownloadResult]:
    local_dir = Path(local_dir)
    objects = [
        obj
        for obj in client.list_objects(remote_prefix)
        if object_in_date_range(obj.get("LastModified"), since, until)
    ]
    results: List[DownloadResult] = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _download_one,
                client,
                obj["Key"],
                obj.get("Size", 0),
                obj.get("LastModified"),
                remote_prefix,
                local_dir,
                force,
                dry_run,
                max_attempts,
            ): obj["Key"]
            for obj in objects
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if progress_cb:
                progress_cb(result)

    return results
