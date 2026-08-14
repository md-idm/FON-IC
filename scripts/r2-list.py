#!/usr/bin/env python3
"""Read-only diagnostic: list objects in the configured R2 bucket.

Uses the existing shared R2 client/configuration (fon_image_cleaner.r2).
Strictly read-only -- never uploads, downloads, or deletes anything, and
never prints R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, or any other secret.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fon_image_cleaner.listing import gather_listing, render_report  # noqa: E402
from fon_image_cleaner.r2 import R2Client  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only diagnostic listing of objects in the configured R2 bucket"
    )
    parser.add_argument(
        "--prefix",
        default="",
        help='R2 key prefix to inspect, e.g. "upload/" (default: "" -- the entire bucket)',
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="Number of sample objects to print (default: 20)"
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    client = R2Client()
    result = gather_listing(client, args.prefix, args.limit)
    print(render_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
