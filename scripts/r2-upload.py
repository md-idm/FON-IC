#!/usr/bin/env python3
"""Recursively upload every file from a local directory into an R2 prefix.

Never deletes remote objects and never uses sync/delete semantics -- this is
a pure additive upload. See README.md for usage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fon_image_cleaner.r2 import R2Client  # noqa: E402
from fon_image_cleaner.reporting import write_report  # noqa: E402
from fon_image_cleaner.uploader import run_upload  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a local directory tree into Cloudflare R2")
    parser.add_argument("--local-dir", required=True, help="Local source directory")
    parser.add_argument("--remote-prefix", required=True, help='R2 key prefix to upload into, e.g. "processed/"')
    parser.add_argument("--workers", type=int, default=8, help="Bounded concurrency (default: 8)")
    parser.add_argument("--force", action="store_true", help="Overwrite remote objects that already exist")
    parser.add_argument("--dry-run", action="store_true", help="List what would be uploaded without uploading")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    local_dir = Path(args.local_dir)

    if not local_dir.is_dir():
        print(f"error: local directory not found: {local_dir}", file=sys.stderr)
        return 2

    client = R2Client()

    counts = {"uploaded": 0, "skipped": 0, "failed": 0}
    total_bytes = 0

    def on_progress(result):
        nonlocal total_bytes
        counts[result.status] = counts.get(result.status, 0) + 1
        if result.status == "uploaded":
            total_bytes += result.size
        print(f"[{result.status}] {result.source} -> {result.destination}")

    results = run_upload(
        client=client,
        local_dir=local_dir,
        remote_prefix=args.remote_prefix,
        workers=args.workers,
        force=args.force,
        dry_run=args.dry_run,
        progress_cb=on_progress,
    )

    report_path = write_report("upload", results)

    print()
    print(f"Files found:   {len(results)}")
    print(f"Uploaded:      {counts.get('uploaded', 0)}")
    print(f"Skipped:       {counts.get('skipped', 0)}")
    print(f"Failed:        {counts.get('failed', 0)}")
    print(f"Total bytes:   {total_bytes}")
    print(f"Report:        {report_path}")

    return 1 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
