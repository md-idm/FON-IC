"""Read-only R2 listing/diagnostics engine used by scripts/r2-list.py.

Strictly read-only: only ever calls `R2Client.list_objects`, never
`download_file`, `upload_file`, or anything that mutates R2 state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, List
from urllib.parse import urlparse

from .r2 import R2Client


@dataclass
class ObjectSummary:
    key: str
    last_modified: str
    size: int


@dataclass
class ListingResult:
    bucket: str
    endpoint_host: str
    prefix: str
    total: int
    top_level_prefixes: List[str] = field(default_factory=list)
    sample: List[ObjectSummary] = field(default_factory=list)


def endpoint_host(endpoint: str) -> str:
    """Return just the host portion of an R2 endpoint URL, e.g.
    'https://acct.r2.cloudflarestorage.com' -> 'acct.r2.cloudflarestorage.com'.

    Never includes credentials, query strings, or path -- callers must not
    print the raw endpoint/config, only this.
    """
    parsed = urlparse(endpoint)
    return parsed.netloc or endpoint


def compute_top_level_prefixes(keys: Iterable[str], prefix: str) -> List[str]:
    """Immediate subdirectories under `prefix`, S3 "common prefix" style.

    A key directly under `prefix` (no further '/') does not contribute a
    top-level prefix -- only keys with at least one more path segment do.
    Callers must only pass keys that already start with `prefix` (which is
    what `client.list_objects(prefix)` guarantees server-side).
    """
    seen = set()
    for key in keys:
        relative = key[len(prefix) :] if key.startswith(prefix) else key
        if "/" in relative:
            segment = relative.split("/", 1)[0]
            seen.add(f"{prefix}{segment}/")
    return sorted(seen)


def _format_last_modified(value) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value is not None else ""


def gather_listing(client: R2Client, prefix: str, limit: int) -> ListingResult:
    """List every object under `prefix` (paginated transparently by
    `client.list_objects`) and summarize it. Read-only: never downloads,
    uploads, or deletes anything.
    """
    objects = list(client.list_objects(prefix))
    keys = [obj.get("Key", "") for obj in objects]
    sample = [
        ObjectSummary(
            key=obj.get("Key", ""),
            last_modified=_format_last_modified(obj.get("LastModified")),
            size=obj.get("Size", 0),
        )
        for obj in objects[: max(0, limit)]
    ]
    return ListingResult(
        bucket=client.bucket,
        endpoint_host=endpoint_host(client.config.endpoint),
        prefix=prefix,
        total=len(objects),
        top_level_prefixes=compute_top_level_prefixes(keys, prefix),
        sample=sample,
    )


def render_report(result: ListingResult) -> str:
    """Build the human-readable diagnostic report.

    Only ever reads fields already scrubbed down to bucket name / endpoint
    host / object metadata -- this function never receives an R2Config or
    credentials, so it is structurally incapable of printing secrets.
    """
    lines = [
        f"Bucket:         {result.bucket}",
        f"Endpoint host:  {result.endpoint_host}",
        f"Prefix:         {result.prefix or '(entire bucket)'}",
        "",
        f"Total objects:  {result.total}",
        "",
        f"Top-level prefixes ({len(result.top_level_prefixes)}):",
    ]
    if result.top_level_prefixes:
        lines.extend(f"  {p}" for p in result.top_level_prefixes)
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append(f"First {len(result.sample)} object(s):")
    if result.sample:
        for obj in result.sample:
            lines.append(f"  {obj.key}\t{obj.last_modified}\t{obj.size}")
    else:
        lines.append("  (none)")
    return "\n".join(lines)
