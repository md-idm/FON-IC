"""Unit tests for R2Client. All boto3 calls are mocked -- no real network I/O."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from fon_image_cleaner.r2 import R2Client, R2Config


def _config():
    return R2Config(
        account_id="acct",
        access_key_id="key",
        secret_access_key="secret",
        bucket_name="bucket",
        endpoint="https://acct.r2.cloudflarestorage.com",
    )


def test_list_objects_paginates_and_flattens():
    mock_boto = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Contents": [{"Key": "a.jpg", "Size": 1}]},
        {"Contents": [{"Key": "b.jpg", "Size": 2}]},
    ]
    mock_boto.get_paginator.return_value = paginator

    client = R2Client(_config(), boto_client=mock_boto)
    keys = [obj["Key"] for obj in client.list_objects("prefix/")]

    assert keys == ["a.jpg", "b.jpg"]
    paginator.paginate.assert_called_once_with(Bucket="bucket", Prefix="prefix/")


def test_list_objects_handles_empty_page():
    mock_boto = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [{}]
    mock_boto.get_paginator.return_value = paginator

    client = R2Client(_config(), boto_client=mock_boto)
    assert list(client.list_objects("prefix/")) == []


def test_object_exists_true():
    mock_boto = MagicMock()
    mock_boto.head_object.return_value = {}
    client = R2Client(_config(), boto_client=mock_boto)
    assert client.object_exists("a.jpg") is True


def test_object_exists_false_on_404():
    mock_boto = MagicMock()
    mock_boto.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadObject")
    client = R2Client(_config(), boto_client=mock_boto)
    assert client.object_exists("missing.jpg") is False


def test_object_exists_reraises_other_errors():
    mock_boto = MagicMock()
    mock_boto.head_object.side_effect = ClientError({"Error": {"Code": "403"}}, "HeadObject")
    client = R2Client(_config(), boto_client=mock_boto)
    with pytest.raises(ClientError):
        client.object_exists("forbidden.jpg")


def test_upload_file_sets_content_type():
    mock_boto = MagicMock()
    client = R2Client(_config(), boto_client=mock_boto)
    client.upload_file("local.jpg", "remote.jpg", content_type="image/jpeg")
    mock_boto.upload_file.assert_called_once_with(
        "local.jpg", "bucket", "remote.jpg", ExtraArgs={"ContentType": "image/jpeg"}
    )


def test_upload_file_without_content_type_omits_extra_args():
    mock_boto = MagicMock()
    client = R2Client(_config(), boto_client=mock_boto)
    client.upload_file("local.bin", "remote.bin")
    mock_boto.upload_file.assert_called_once_with("local.bin", "bucket", "remote.bin")


def test_download_file_uses_temp_file_and_replaces(tmp_path):
    mock_boto = MagicMock()

    def fake_download_file(bucket, key, filename):
        Path(filename).write_bytes(b"data")

    mock_boto.download_file.side_effect = fake_download_file
    client = R2Client(_config(), boto_client=mock_boto)

    dest = tmp_path / "nested" / "a.jpg"
    client.download_file("a.jpg", dest)

    assert dest.read_bytes() == b"data"
    assert not dest.with_name(dest.name + ".part").exists()


def test_config_from_env_raises_on_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME", "R2_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError):
        R2Config.from_env()


def test_config_from_env_derives_endpoint_from_account_id(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("R2_ACCOUNT_ID", "abc123")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("R2_BUCKET_NAME", "bucket")
    monkeypatch.delenv("R2_ENDPOINT", raising=False)

    config = R2Config.from_env()
    assert config.endpoint == "https://abc123.r2.cloudflarestorage.com"
