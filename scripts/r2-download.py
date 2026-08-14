#!/usr/bin/env python3
"""Recursively download every object under an R2 prefix into a local directory.

Never deletes anything from R2. See README.md for usage. This script is
independent of remove-background.py and r2-upload.py -- it never invokes
either automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fon_image_cleaner.downloader import DateRangeError, resolve_date_range, run_download  # noqa: E402
from fon_image_cleaner.r2 import R2Client  # noqa: E402
from fon_image_cleaner.reporting import write_report  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download files from Cloudflare R2 into a local directory")
    parser.add_argument(
        "--remote-prefix",
        default="",
        help='R2 key prefix to copy, e.g. "upload/" (default: "" -- the entire bucket)',
    )
    parser.add_argument("--local-dir", required=True, help="Local destination directory")
    parser.add_argument("--workers", type=int, default=8, help="Bounded concurrency (default: 8)")
    parser.add_argument("--force", action="store_true", help="Overwrite local files that already exist")
    parser.add_argument("--dry-run", action="store_true", help="List what would be downloaded without downloading")
    parser.add_argument(
        "--since", default=None, help="Only include objects with LastModified >= this UTC day (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--until", default=None, help="Only include objects with LastModified within this UTC day (YYYY-MM-DD)"
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    local_dir = Path(args.local_dir)

    try:
        since_dt, until_dt = resolve_date_range(args.since, args.until)
    except DateRangeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    client = R2Client()

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    total_bytes = 0

    def on_progress(result):
        nonlocal total_bytes
        counts[result.status] = counts.get(result.status, 0) + 1
        if result.status == "downloaded":
            total_bytes += result.size
        print(f"[{result.status}] {result.source} -> {result.destination}")

    prefix_label = f"'{args.remote_prefix}'" if args.remote_prefix else "the entire bucket"
    print(f"Listing objects under {prefix_label} ...")
    results = run_download(
        client=client,
        remote_prefix=args.remote_prefix,
        local_dir=local_dir,
        workers=args.workers,
        force=args.force,
        dry_run=args.dry_run,
        since=since_dt,
        until=until_dt,
        progress_cb=on_progress,
    )

    report_path = write_report("download", results)

    print()
    print(f"Files found:   {len(results)}")
    print(f"Downloaded:    {counts.get('downloaded', 0)}")
    print(f"Skipped:       {counts.get('skipped', 0)}")
    print(f"Failed:        {counts.get('failed', 0)}")
    print(f"Total bytes:   {total_bytes}")
    print(f"Report:        {report_path}")

    return 1 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
