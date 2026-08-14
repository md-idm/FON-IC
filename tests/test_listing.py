"""Unit tests for fon_image_cleaner.listing (the engine behind scripts/r2-list.py).

R2 access is always mocked -- these tests never contact real Cloudflare.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from fon_image_cleaner.listing import (
    compute_top_level_prefixes,
    endpoint_host,
    gather_listing,
    render_report,
)
from fon_image_cleaner.r2 import R2Client, R2Config

UTC = timezone.utc

FAKE_SECRET = "SuperSecretAccessKeyValue123456"
FAKE_ACCESS_KEY_ID = "AKIAFAKEACCESSKEYID0001"


def _make_mock_client(pages):
    mock_boto = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = pages
    mock_boto.get_paginator.return_value = paginator

    config = R2Config(
        account_id="acct123",
        access_key_id=FAKE_ACCESS_KEY_ID,
        secret_access_key=FAKE_SECRET,
        bucket_name="my-bucket",
        endpoint="https://acct123.r2.cloudflarestorage.com",
    )
    client = R2Client(config, boto_client=mock_boto)
    return client, mock_boto


def test_empty_prefix_lists_entire_bucket():
    client, _ = _make_mock_client(
        [
            {
                "Contents": [
                    {"Key": "a.jpg", "Size": 1, "LastModified": datetime(2026, 1, 1, tzinfo=UTC)},
                    {"Key": "upload/b.jpg", "Size": 2, "LastModified": datetime(2026, 1, 2, tzinfo=UTC)},
                ]
            }
        ]
    )
    result = gather_listing(client, "", limit=20)
    assert result.total == 2
    assert {obj.key for obj in result.sample} == {"a.jpg", "upload/b.jpg"}


def test_prefix_filtering():
    """client.list_objects already filters server-side by Prefix --
    gather_listing must pass the prefix through unchanged."""
    client, mock_boto = _make_mock_client(
        [{"Contents": [{"Key": "upload/a.jpg", "Size": 1, "LastModified": datetime(2026, 1, 1, tzinfo=UTC)}]}]
    )
    result = gather_listing(client, "upload/", limit=20)
    assert result.total == 1
    assert result.sample[0].key == "upload/a.jpg"
    mock_boto.get_paginator.return_value.paginate.assert_called_once_with(
        Bucket="my-bucket", Prefix="upload/"
    )


def test_pagination_combines_all_pages():
    client, _ = _make_mock_client(
        [
            {
                "Contents": [
                    {"Key": f"upload/page1-{i}.jpg", "Size": i, "LastModified": datetime(2026, 1, 1, tzinfo=UTC)}
                    for i in range(3)
                ]
            },
            {
                "Contents": [
                    {"Key": f"upload/page2-{i}.jpg", "Size": i, "LastModified": datetime(2026, 1, 2, tzinfo=UTC)}
                    for i in range(3)
                ]
            },
        ]
    )
    result = gather_listing(client, "upload/", limit=100)
    assert result.total == 6
    assert len(result.sample) == 6


def test_top_level_prefix_detection():
    all_keys = ["upload/a.jpg", "upload/reels/b.jpg", "upload/reels/c.jpg", "processed/d.jpg", "root.jpg"]
    assert compute_top_level_prefixes(all_keys, "") == ["processed/", "upload/"]

    upload_keys = [k for k in all_keys if k.startswith("upload/")]
    assert compute_top_level_prefixes(upload_keys, "upload/") == ["upload/reels/"]


def test_top_level_prefix_detection_via_gather_listing():
    client, _ = _make_mock_client(
        [
            {
                "Contents": [
                    {"Key": "upload/a.jpg", "Size": 1, "LastModified": datetime(2026, 1, 1, tzinfo=UTC)},
                    {"Key": "upload/reels/b.jpg", "Size": 2, "LastModified": datetime(2026, 1, 1, tzinfo=UTC)},
                ]
            }
        ]
    )
    result = gather_listing(client, "upload/", limit=20)
    assert result.top_level_prefixes == ["upload/reels/"]


def test_last_modified_is_included_per_object():
    when = datetime(2026, 8, 5, 12, 30, 0, tzinfo=UTC)
    client, _ = _make_mock_client([{"Contents": [{"Key": "a.jpg", "Size": 10, "LastModified": when}]}])
    result = gather_listing(client, "", limit=10)
    assert result.sample[0].last_modified == when.isoformat()
    assert when.isoformat() in render_report(result)


def test_limit_caps_sample_but_not_total():
    pages = [
        {
            "Contents": [
                {"Key": f"a{i}.jpg", "Size": i, "LastModified": datetime(2026, 1, 1, tzinfo=UTC)}
                for i in range(10)
            ]
        }
    ]
    client, _ = _make_mock_client(pages)
    result = gather_listing(client, "", limit=3)
    assert result.total == 10
    assert len(result.sample) == 3


def test_no_secrets_in_rendered_output():
    client, _ = _make_mock_client(
        [{"Contents": [{"Key": "a.jpg", "Size": 1, "LastModified": datetime(2026, 1, 1, tzinfo=UTC)}]}]
    )
    result = gather_listing(client, "", limit=10)
    report = render_report(result)

    assert FAKE_SECRET not in report
    assert FAKE_ACCESS_KEY_ID not in report
    assert "Authorization" not in report
    assert "authorization" not in report.lower()
    # only bucket name + endpoint host should appear, never the raw endpoint URL
    assert "https://" not in report
    assert "my-bucket" in report
    assert "acct123.r2.cloudflarestorage.com" in report


def test_no_secrets_leak_via_config_repr():
    """Defense in depth: even an accidental print(client.config) must not
    leak credentials."""
    client, _ = _make_mock_client([{"Contents": []}])
    dumped = repr(client.config)
    assert FAKE_SECRET not in dumped
    assert FAKE_ACCESS_KEY_ID not in dumped


def test_endpoint_host_strips_scheme_and_path():
    assert endpoint_host("https://acct123.r2.cloudflarestorage.com") == "acct123.r2.cloudflarestorage.com"
    assert (
        endpoint_host("https://acct123.r2.cloudflarestorage.com/some/path")
        == "acct123.r2.cloudflarestorage.com"
    )
