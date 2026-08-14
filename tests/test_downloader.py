from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fakes import FakeR2Client

from fon_image_cleaner.downloader import (
    DateRangeError,
    object_in_date_range,
    resolve_date_range,
    run_download,
)
from fon_image_cleaner.r2 import R2Client, R2Config

UTC = timezone.utc


def test_recursive_prefix_mapping_and_directory_preservation(tmp_path):
    client = FakeR2Client(
        {
            "upload/a.jpg": b"AAA",
            "upload/b.jpg": b"BBB",
            "upload/reels/c.jpg": b"CCC",
        }
    )
    results = run_download(client, "upload/", tmp_path, workers=2)

    assert (tmp_path / "a.jpg").read_bytes() == b"AAA"
    assert (tmp_path / "b.jpg").read_bytes() == b"BBB"
    assert (tmp_path / "reels" / "c.jpg").read_bytes() == b"CCC"
    assert {r.status for r in results} == {"downloaded"}
    assert len(results) == 3


def test_skip_existing_by_default(tmp_path):
    client = FakeR2Client({"upload/a.jpg": b"NEW"})
    (tmp_path / "a.jpg").write_bytes(b"OLD")

    results = run_download(client, "upload/", tmp_path, workers=1)

    assert (tmp_path / "a.jpg").read_bytes() == b"OLD"
    assert results[0].status == "skipped"


def test_force_overwrites_local_files(tmp_path):
    client = FakeR2Client({"upload/a.jpg": b"NEW"})
    (tmp_path / "a.jpg").write_bytes(b"OLD")

    results = run_download(client, "upload/", tmp_path, workers=1, force=True)

    assert (tmp_path / "a.jpg").read_bytes() == b"NEW"
    assert results[0].status == "downloaded"


def test_dry_run_writes_nothing(tmp_path):
    client = FakeR2Client({"upload/a.jpg": b"NEW", "upload/reels/c.jpg": b"CCC"})

    results = run_download(client, "upload/", tmp_path, workers=2, dry_run=True)

    assert not (tmp_path / "a.jpg").exists()
    assert not (tmp_path / "reels" / "c.jpg").exists()
    assert {r.status for r in results} == {"skipped"}


def test_failed_object_does_not_stop_batch(tmp_path):
    client = FakeR2Client(
        {
            "upload/a.jpg": b"AAA",
            "upload/bad.jpg": b"BAD",
        }
    )
    client.fail_keys.add("upload/bad.jpg")

    results = run_download(client, "upload/", tmp_path, workers=2, max_attempts=1)

    statuses = {r.source: r.status for r in results}
    assert statuses["upload/a.jpg"] == "downloaded"
    assert statuses["upload/bad.jpg"] == "failed"
    assert (tmp_path / "a.jpg").exists()
    assert not (tmp_path / "bad.jpg").exists()


def test_never_deletes_from_r2(tmp_path):
    client = FakeR2Client({"upload/a.jpg": b"AAA"})
    run_download(client, "upload/", tmp_path, workers=1)
    assert "upload/a.jpg" in client.objects


# --- --since / --until parsing and validation -------------------------------


def test_no_dates_means_all_objects_eligible():
    assert object_in_date_range(datetime(2020, 1, 1, tzinfo=UTC), None, None) is True
    assert object_in_date_range(None, None, None) is True


def test_since_excludes_older_objects():
    since_dt, _ = resolve_date_range("2026-08-01", None)
    older = datetime(2026, 7, 31, 23, 59, 59, tzinfo=UTC)
    assert object_in_date_range(older, since_dt, None) is False


def test_since_midnight_boundary_is_included():
    since_dt, _ = resolve_date_range("2026-08-01", None)
    exactly_midnight = datetime(2026, 8, 1, 0, 0, 0, tzinfo=UTC)
    assert object_in_date_range(exactly_midnight, since_dt, None) is True


def test_until_includes_entire_day():
    _, until_dt = resolve_date_range(None, "2026-08-10")
    end_of_day = datetime(2026, 8, 10, 23, 59, 59, tzinfo=UTC)
    assert object_in_date_range(end_of_day, None, until_dt) is True


def test_until_next_day_midnight_is_excluded():
    _, until_dt = resolve_date_range(None, "2026-08-10")
    next_day_midnight = datetime(2026, 8, 11, 0, 0, 0, tzinfo=UTC)
    assert object_in_date_range(next_day_midnight, None, until_dt) is False


def test_since_and_until_range():
    since_dt, until_dt = resolve_date_range("2026-08-01", "2026-08-10")
    inside = datetime(2026, 8, 5, 12, 0, 0, tzinfo=UTC)
    before = datetime(2026, 7, 31, tzinfo=UTC)
    after = datetime(2026, 8, 11, tzinfo=UTC)
    assert object_in_date_range(inside, since_dt, until_dt) is True
    assert object_in_date_range(before, since_dt, until_dt) is False
    assert object_in_date_range(after, since_dt, until_dt) is False


def test_invalid_date_is_rejected():
    with pytest.raises(DateRangeError):
        resolve_date_range("2026/08/01", None)
    with pytest.raises(DateRangeError):
        resolve_date_range(None, "not-a-date")


def test_since_after_until_is_rejected():
    with pytest.raises(DateRangeError):
        resolve_date_range("2026-08-10", "2026-08-01")


def test_since_equal_until_is_a_valid_one_day_range():
    since_dt, until_dt = resolve_date_range("2026-08-05", "2026-08-05")
    assert object_in_date_range(datetime(2026, 8, 5, 6, tzinfo=UTC), since_dt, until_dt) is True
    assert object_in_date_range(datetime(2026, 8, 6, 0, tzinfo=UTC), since_dt, until_dt) is False


def test_naive_last_modified_is_treated_as_utc():
    """Real S3/R2 LastModified values are always tz-aware, but the filter
    must still compare correctly if a naive datetime ever reaches it."""
    since_dt, _ = resolve_date_range("2026-08-01", None)
    naive = datetime(2026, 8, 1, 12, 0, 0)  # no tzinfo
    assert object_in_date_range(naive, since_dt, None) is True


# --- date filtering wired through run_download ------------------------------


def test_run_download_since_filters_objects(tmp_path):
    client = FakeR2Client(
        {"upload/old.jpg": b"OLD", "upload/new.jpg": b"NEW"},
        last_modified={
            "upload/old.jpg": datetime(2026, 7, 1, tzinfo=UTC),
            "upload/new.jpg": datetime(2026, 8, 5, tzinfo=UTC),
        },
    )
    since_dt, _ = resolve_date_range("2026-08-01", None)

    results = run_download(client, "upload/", tmp_path, workers=2, since=since_dt)

    sources = {r.source for r in results}
    assert sources == {"upload/new.jpg"}
    assert (tmp_path / "new.jpg").exists()
    assert not (tmp_path / "old.jpg").exists()


def test_run_download_dry_run_with_date_filter_downloads_nothing(tmp_path):
    client = FakeR2Client(
        {"upload/old.jpg": b"OLD", "upload/new.jpg": b"NEW"},
        last_modified={
            "upload/old.jpg": datetime(2026, 7, 1, tzinfo=UTC),
            "upload/new.jpg": datetime(2026, 8, 5, tzinfo=UTC),
        },
    )
    since_dt, until_dt = resolve_date_range("2026-08-01", "2026-08-10")

    results = run_download(
        client, "upload/", tmp_path, workers=2, since=since_dt, until=until_dt, dry_run=True
    )

    assert [r.source for r in results] == ["upload/new.jpg"]
    assert results[0].status == "skipped"
    assert not (tmp_path / "new.jpg").exists()
    assert not (tmp_path / "old.jpg").exists()


def test_run_download_reports_last_modified(tmp_path):
    when = datetime(2026, 8, 5, 12, 0, 0, tzinfo=UTC)
    client = FakeR2Client({"upload/a.jpg": b"AAA"}, last_modified={"upload/a.jpg": when})

    results = run_download(client, "upload/", tmp_path, workers=1)

    assert results[0].last_modified == when.isoformat()


def test_pagination_and_date_filtering_via_real_r2_client(tmp_path):
    """Exercises R2Client's paginator-flattening together with date
    filtering, mirroring how boto3 actually returns multi-page listings."""
    mock_boto = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": "upload/page1-old.jpg", "Size": 1, "LastModified": datetime(2026, 7, 1, tzinfo=UTC)},
                {"Key": "upload/page1-new.jpg", "Size": 2, "LastModified": datetime(2026, 8, 3, tzinfo=UTC)},
            ]
        },
        {
            "Contents": [
                {"Key": "upload/page2-new.jpg", "Size": 3, "LastModified": datetime(2026, 8, 7, tzinfo=UTC)},
                {"Key": "upload/page2-old.jpg", "Size": 4, "LastModified": datetime(2026, 6, 1, tzinfo=UTC)},
            ]
        },
    ]
    mock_boto.get_paginator.return_value = paginator

    def fake_download_file(bucket, key, filename):
        Path(filename).write_bytes(b"data")

    mock_boto.download_file.side_effect = fake_download_file

    config = R2Config(
        account_id="acct",
        access_key_id="key",
        secret_access_key="secret",
        bucket_name="bucket",
        endpoint="https://acct.r2.cloudflarestorage.com",
    )
    client = R2Client(config, boto_client=mock_boto)
    since_dt, _ = resolve_date_range("2026-08-01", None)

    results = run_download(client, "upload/", tmp_path, workers=2, since=since_dt)

    sources = {r.source for r in results}
    assert sources == {"upload/page1-new.jpg", "upload/page2-new.jpg"}
    assert (tmp_path / "page1-new.jpg").exists()
    assert (tmp_path / "page2-new.jpg").exists()
    assert not (tmp_path / "page1-old.jpg").exists()
    assert not (tmp_path / "page2-old.jpg").exists()
