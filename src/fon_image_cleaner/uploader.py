"""Recursive, concurrent local directory -> R2 upload logic.

Used by scripts/r2-upload.py. Never deletes remote objects and never uses
sync/delete semantics -- this is a pure additive upload.
"""

from __future__ import annotations

import mimetypes
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from botocore.exceptions import BotoCoreError, ClientError

from .r2 import R2Client

TRANSIENT_EXCEPTIONS = (BotoCoreError, ClientError, TimeoutError, ConnectionError, OSError)

CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


@dataclass
class UploadResult:
    source: str
    destination: str
    status: str  # uploaded | skipped | failed
    size: int = 0
    duration_ms: int = 0
    error: str = ""


def guess_content_type(path: Path) -> Optional[str]:
    known = CONTENT_TYPES.get(path.suffix.lower())
    if known:
        return known
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed


def local_path_to_key(local_path: Path, local_dir: Path, remote_prefix: str) -> str:
    relative = local_path.relative_to(local_dir)
    relative_posix = "/".join(relative.parts)
    prefix = remote_prefix
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return f"{prefix}{relative_posix}"


def _upload_one(
    client: R2Client,
    local_path: Path,
    local_dir: Path,
    remote_prefix: str,
    force: bool,
    dry_run: bool,
    max_attempts: int,
) -> UploadResult:
    started = time.monotonic()
    key = local_path_to_key(local_path, local_dir, remote_prefix)
    size = local_path.stat().st_size

    if not force and client.object_exists(key):
        return UploadResult(str(local_path), key, "skipped", size, 0, "already exists remotely")

    if dry_run:
        return UploadResult(str(local_path), key, "skipped", size, 0, "dry-run")

    content_type = guess_content_type(local_path)
    attempt = 0
    while True:
        attempt += 1
        try:
            client.upload_file(local_path, key, content_type)
            duration_ms = int((time.monotonic() - started) * 1000)
            return UploadResult(str(local_path), key, "uploaded", size, duration_ms, "")
        except TRANSIENT_EXCEPTIONS as exc:
            if attempt >= max_attempts:
                duration_ms = int((time.monotonic() - started) * 1000)
                return UploadResult(str(local_path), key, "failed", size, duration_ms, str(exc))
            time.sleep(min(0.25 * (2 ** attempt), 5))


def run_upload(
    client: R2Client,
    local_dir: Path,
    remote_prefix: str,
    workers: int = 8,
    force: bool = False,
    dry_run: bool = False,
    max_attempts: int = 3,
    progress_cb: Optional[Callable[[UploadResult], None]] = None,
) -> List[UploadResult]:
    local_dir = Path(local_dir)
    files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    results: List[UploadResult] = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _upload_one, client, path, local_dir, remote_prefix, force, dry_run, max_attempts
            ): path
            for path in files
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if progress_cb:
                progress_cb(result)

    return results
