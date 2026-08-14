"""Shared Cloudflare R2 (S3-compatible) client.

All three CLI scripts route their storage operations through this module so
that connection setup, listing, existence checks, download, and upload logic
are implemented exactly once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

_NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound"}


def _load_dotenv(path: str = ".env") -> None:
    """Populate os.environ from a .env file if present, without overriding
    variables that are already set. Avoids adding a python-dotenv dependency
    for a handful of KEY=VALUE lines."""
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    return "***" if len(value) <= 4 else f"***{value[-4:]}"


@dataclass(frozen=True)
class R2Config:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket_name: str
    endpoint: str

    def __repr__(self) -> str:
        # Never expose credentials via repr/print/logging/tracebacks -- only
        # a masked suffix, matching the "never log secrets" requirement.
        return (
            "R2Config("
            f"account_id={self.account_id!r}, "
            f"access_key_id={_mask_secret(self.access_key_id)!r}, "
            f"secret_access_key={_mask_secret(self.secret_access_key)!r}, "
            f"bucket_name={self.bucket_name!r}, "
            f"endpoint={self.endpoint!r})"
        )

    @classmethod
    def from_env(cls) -> "R2Config":
        _load_dotenv()
        account_id = os.environ.get("R2_ACCOUNT_ID", "").strip()
        access_key_id = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
        secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
        bucket_name = os.environ.get("R2_BUCKET_NAME", "").strip()
        endpoint = os.environ.get("R2_ENDPOINT", "").strip()
        if not endpoint and account_id:
            endpoint = f"https://{account_id}.r2.cloudflarestorage.com"

        missing = [
            name
            for name, value in (
                ("R2_ACCESS_KEY_ID", access_key_id),
                ("R2_SECRET_ACCESS_KEY", secret_access_key),
                ("R2_BUCKET_NAME", bucket_name),
                ("R2_ENDPOINT", endpoint),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Missing required R2 configuration: "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill in the values, "
                "or set these as environment variables."
            )
        return cls(
            account_id=account_id,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            bucket_name=bucket_name,
            endpoint=endpoint,
        )


class R2Client:
    """Thin wrapper around a boto3 S3 client pointed at a Cloudflare R2 bucket.

    A pre-built ``boto_client`` may be injected (used by tests to mock all
    network calls); otherwise a real boto3 client is created from ``config``.
    """

    def __init__(self, config: Optional[R2Config] = None, boto_client=None):
        self.config = config or R2Config.from_env()
        self.bucket = self.config.bucket_name
        self._client = boto_client or boto3.client(
            "s3",
            endpoint_url=self.config.endpoint,
            aws_access_key_id=self.config.access_key_id,
            aws_secret_access_key=self.config.secret_access_key,
            config=BotoConfig(signature_version="s3v4", retries={"max_attempts": 0}),
            region_name="auto",
        )

    def list_objects(self, prefix: str) -> Iterator[dict]:
        """Yield every object (dict with at least ``Key`` and ``Size``) under
        ``prefix``, transparently paginating."""
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj

    def object_exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in _NOT_FOUND_CODES or status == 404:
                return False
            raise

    def download_file(self, key: str, local_path) -> None:
        """Download ``key`` to ``local_path`` atomically (via a .part temp
        file) so a crash mid-download never leaves a corrupt final file."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = local_path.with_name(local_path.name + ".part")
        self._client.download_file(self.bucket, key, str(tmp_path))
        tmp_path.replace(local_path)

    def upload_file(self, local_path, key: str, content_type: Optional[str] = None) -> None:
        extra_args = {"ContentType": content_type} if content_type else None
        kwargs = {"ExtraArgs": extra_args} if extra_args else {}
        self._client.upload_file(str(local_path), self.bucket, key, **kwargs)
