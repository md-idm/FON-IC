from pathlib import Path

from fakes import FakeR2Client

from fon_image_cleaner.uploader import run_upload


def _make_tree(base: Path) -> None:
    (base / "reels").mkdir(parents=True)
    (base / "a.webp").write_bytes(b"AAA")
    (base / "b.webp").write_bytes(b"BBB")
    (base / "reels" / "c.webp").write_bytes(b"CCC")


def test_recursive_upload_preserves_relative_paths(tmp_path):
    _make_tree(tmp_path)
    client = FakeR2Client()

    results = run_upload(client, tmp_path, "processed/", workers=2)

    assert client.uploaded["processed/a.webp"] == b"AAA"
    assert client.uploaded["processed/b.webp"] == b"BBB"
    assert client.uploaded["processed/reels/c.webp"] == b"CCC"
    assert {r.status for r in results} == {"uploaded"}


def test_content_type_detection(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "c.webp").write_bytes(b"x")
    client = FakeR2Client()

    run_upload(client, tmp_path, "processed/", workers=1)

    assert client.upload_content_types["processed/a.jpg"] == "image/jpeg"
    assert client.upload_content_types["processed/b.png"] == "image/png"
    assert client.upload_content_types["processed/c.webp"] == "image/webp"


def test_skip_existing_remote_object(tmp_path):
    (tmp_path / "a.webp").write_bytes(b"NEW")
    client = FakeR2Client()
    client.objects["processed/a.webp"] = b"OLD"

    results = run_upload(client, tmp_path, "processed/", workers=1)

    assert "processed/a.webp" not in client.uploaded
    assert results[0].status == "skipped"


def test_force_overwrites_remote_object(tmp_path):
    (tmp_path / "a.webp").write_bytes(b"NEW")
    client = FakeR2Client()
    client.objects["processed/a.webp"] = b"OLD"

    results = run_upload(client, tmp_path, "processed/", workers=1, force=True)

    assert client.uploaded["processed/a.webp"] == b"NEW"
    assert results[0].status == "uploaded"


def test_dry_run_uploads_nothing(tmp_path):
    (tmp_path / "a.webp").write_bytes(b"NEW")
    client = FakeR2Client()

    results = run_upload(client, tmp_path, "processed/", workers=1, dry_run=True)

    assert client.uploaded == {}
    assert results[0].status == "skipped"


def test_failed_upload_does_not_stop_batch(tmp_path):
    (tmp_path / "a.webp").write_bytes(b"AAA")
    (tmp_path / "bad.webp").write_bytes(b"BAD")
    client = FakeR2Client()
    client.fail_keys.add("processed/bad.webp")

    results = run_upload(client, tmp_path, "processed/", workers=2, max_attempts=1)

    statuses = {Path(r.source).name: r.status for r in results}
    assert statuses["a.webp"] == "uploaded"
    assert statuses["bad.webp"] == "failed"
    assert client.uploaded["processed/a.webp"] == b"AAA"
