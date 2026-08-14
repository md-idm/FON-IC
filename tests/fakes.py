"""In-memory fake standing in for R2Client so tests never touch real R2/S3."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Optional, Set

from botocore.exceptions import ClientError


class FakeR2Client:
    def __init__(
        self,
        objects: Optional[Dict[str, bytes]] = None,
        last_modified: Optional[Dict[str, datetime]] = None,
    ):
        self.objects: Dict[str, bytes] = dict(objects or {})
        self.last_modified: Dict[str, datetime] = dict(last_modified or {})
        self.uploaded: Dict[str, bytes] = {}
        self.upload_content_types: Dict[str, str] = {}
        self.fail_keys: Set[str] = set()

    def list_objects(self, prefix: str) -> Iterator[dict]:
        for key, data in self.objects.items():
            if key.startswith(prefix):
                entry = {"Key": key, "Size": len(data)}
                if key in self.last_modified:
                    entry["LastModified"] = self.last_modified[key]
                yield entry

    def object_exists(self, key: str) -> bool:
        return key in self.objects or key in self.uploaded

    def download_file(self, key: str, local_path) -> None:
        if key in self.fail_keys:
            raise ClientError({"Error": {"Code": "500", "Message": "simulated failure"}}, "GetObject")
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.objects[key])

    def upload_file(self, local_path, key: str, content_type: Optional[str] = None) -> None:
        if key in self.fail_keys:
            raise ClientError({"Error": {"Code": "500", "Message": "simulated failure"}}, "PutObject")
        self.uploaded[key] = Path(local_path).read_bytes()
        if content_type:
            self.upload_content_types[key] = content_type
