"""CSV run-report writer shared by all three CLI scripts."""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"

FIELDNAMES = ["source", "destination", "last_modified", "size", "status", "duration_ms", "error"]


def write_report(kind: str, rows: Sequence, reports_dir: Path = REPORTS_DIR) -> Path:
    """Write ``rows`` (dataclass instances or dicts with FIELDNAMES keys) to
    reports/run-<kind>-YYYYMMDD-HHMMSS.csv and return the path written."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = reports_dir / f"run-{kind}-{timestamp}.csv"
    with report_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            data = asdict(row) if is_dataclass(row) else dict(row)
            writer.writerow({key: data.get(key, "") for key in FIELDNAMES})
    return report_path
