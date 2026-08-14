"""Unit tests for recursive directory traversal in fon_image_cleaner.background.

Covers discovery at arbitrary nesting depth, case-insensitive extension
matching, relative path preservation into --output (including with --format
and --save-mask), duplicate filenames across sibling folders, resume/skip at
any depth, deterministic --limit ordering, and the --output-inside---input
safety guard. All I/O happens in pytest tmp_path -- nothing here touches
real images or Cloudflare.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from fon_image_cleaner.background import discover_images, run_process


def _make_source_image(path: Path, size=(20, 20), color=(0, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def fake_remove_background(image: Image.Image) -> Image.Image:
    """Trivial no-op segmentation -- the whole image stays opaque
    foreground. These tests are about *which files* get discovered and
    *where* output lands, not segmentation quality (that's covered
    elsewhere), so a real/fake AI distinction doesn't matter here."""
    return image.convert("RGBA")


def _counting_fn(calls):
    def _fn(image):
        calls.append(1)
        return fake_remove_background(image)

    return _fn


# --- 1-3. discovery at increasing nesting depth -----------------------


def test_image_directly_under_input_root(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    _make_source_image(input_dir / "a.jpg")

    results = run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=1)

    assert results[0].status == "processed"
    assert (output_dir / "a.jpg").exists()


def test_image_one_directory_deep(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    _make_source_image(input_dir / "hooks" / "a.jpg")

    results = run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=1)

    assert results[0].status == "processed"
    assert (output_dir / "hooks" / "a.jpg").exists()


def test_image_three_or_more_directories_deep(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    _make_source_image(input_dir / "hooks" / "treble" / "small" / "b.png")

    results = run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=1)

    assert results[0].status == "processed"
    assert (output_dir / "hooks" / "treble" / "small" / "b.png").exists()


# --- 4. mixed extensions recursively ------------------------------------


def test_mixed_extensions_recursively(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    _make_source_image(input_dir / "hooks" / "a.jpg")
    _make_source_image(input_dir / "hooks" / "treble" / "b.png")
    _make_source_image(input_dir / "alarms" / "set-1" / "c.webp")
    _make_source_image(input_dir / "floats" / "d.jpeg")

    results = run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=2)

    assert {r.status for r in results} == {"processed"}
    assert len(results) == 4
    assert (output_dir / "hooks" / "a.jpg").exists()
    assert (output_dir / "hooks" / "treble" / "b.png").exists()
    assert (output_dir / "alarms" / "set-1" / "c.webp").exists()
    assert (output_dir / "floats" / "d.jpeg").exists()


# --- 5. uppercase extensions --------------------------------------------


def test_uppercase_extensions_are_discovered(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    _make_source_image(input_dir / "IMAGE.JPG")
    _make_source_image(input_dir / "sub" / "photo.JPEG")
    _make_source_image(input_dir / "sub" / "item.PNG")
    _make_source_image(input_dir / "sub" / "product.WebP")

    results = run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=2)

    assert len(results) == 4
    assert {r.status for r in results} == {"processed"}
    # preserve mode keeps the original extension text (case included)
    assert (output_dir / "IMAGE.JPG").exists()
    assert (output_dir / "sub" / "photo.JPEG").exists()
    assert (output_dir / "sub" / "item.PNG").exists()
    assert (output_dir / "sub" / "product.WebP").exists()


# --- 6. unsupported nested files ignored --------------------------------


def test_unsupported_nested_files_are_ignored(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    _make_source_image(input_dir / "hooks" / "a.jpg")
    (input_dir / "hooks" / "notes.txt").parent.mkdir(parents=True, exist_ok=True)
    (input_dir / "hooks" / "notes.txt").write_text("not an image")
    (input_dir / "data.csv").write_text("a,b,c")
    (input_dir / "meta.json").write_text("{}")
    (input_dir / "archive.db").write_bytes(b"\x00\x01")
    (input_dir / "doc.pdf").write_bytes(b"%PDF-1.4")

    results = run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=1)

    assert len(results) == 1
    assert results[0].source.endswith("a.jpg")
    assert not (output_dir / "hooks" / "notes.txt").exists()
    assert not (output_dir / "data.csv").exists()
    assert not (output_dir / "meta.json").exists()
    assert not (output_dir / "archive.db").exists()
    assert not (output_dir / "doc.pdf").exists()


# --- 7. relative output structure preserved (full example from spec) --


def test_relative_output_structure_matches_full_example(tmp_path: Path):
    input_dir = tmp_path / "original"
    output_dir = tmp_path / "processed"
    _make_source_image(input_dir / "hooks" / "a.jpg")
    _make_source_image(input_dir / "hooks" / "treble" / "b.png")
    _make_source_image(input_dir / "alarms" / "set-1" / "c.webp")
    _make_source_image(input_dir / "floats" / "d.jpeg")

    run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=2)

    assert (output_dir / "hooks" / "a.jpg").exists()
    assert (output_dir / "hooks" / "treble" / "b.png").exists()
    assert (output_dir / "alarms" / "set-1" / "c.webp").exists()
    assert (output_dir / "floats" / "d.jpeg").exists()
    # nothing extra, nothing flattened
    produced = sorted(p.relative_to(output_dir) for p in output_dir.rglob("*") if p.is_file())
    expected = sorted(
        Path(p)
        for p in (
            "hooks/a.jpg",
            "hooks/treble/b.png",
            "alarms/set-1/c.webp",
            "floats/d.jpeg",
        )
    )
    assert produced == expected


# --- 8. duplicate filenames in different folders stay distinct ---------


def test_duplicate_filenames_in_different_folders_stay_distinct(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    _make_source_image(input_dir / "hooks" / "1.jpg", color=(255, 0, 0))
    _make_source_image(input_dir / "reels" / "1.jpg", color=(0, 255, 0))

    results = run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=2)

    assert {r.status for r in results} == {"processed"}
    assert (output_dir / "hooks" / "1.jpg").exists()
    assert (output_dir / "reels" / "1.jpg").exists()
    # Check the canvas *center* -- the corner falls in the default padding
    # margin, which is white for both regardless of source content.
    with Image.open(output_dir / "hooks" / "1.jpg") as img_a:
        center_a = (img_a.width // 2, img_a.height // 2)
        color_a = img_a.convert("RGB").getpixel(center_a)
    with Image.open(output_dir / "reels" / "1.jpg") as img_b:
        center_b = (img_b.width // 2, img_b.height // 2)
        color_b = img_b.convert("RGB").getpixel(center_b)
    assert color_a != color_b  # neither overwrote the other


# --- 9-11. resume/skip/force at depth -----------------------------------


def test_existing_nested_output_is_skipped(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    _make_source_image(input_dir / "hooks" / "treble" / "a.png")
    dest = output_dir / "hooks" / "treble" / "a.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"EXISTING")

    results = run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=1)

    assert results[0].status == "skipped"
    assert results[0].error == "output_exists"
    assert dest.read_bytes() == b"EXISTING"


def test_skipped_nested_output_never_invokes_processing(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    _make_source_image(input_dir / "hooks" / "treble" / "a.png")
    dest = output_dir / "hooks" / "treble" / "a.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"EXISTING")

    calls: list = []
    run_process(input_dir, output_dir, remove_background_fn=_counting_fn(calls), workers=1)

    assert calls == []


def test_force_regenerates_nested_output(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    _make_source_image(input_dir / "hooks" / "treble" / "a.png")
    dest = output_dir / "hooks" / "treble" / "a.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"EXISTING")

    calls: list = []
    results = run_process(
        input_dir, output_dir, remove_background_fn=_counting_fn(calls), workers=1, force=True
    )

    assert results[0].status == "processed"
    assert dest.read_bytes() != b"EXISTING"
    assert len(calls) == 1


# --- 12. --format webp changes extension, keeps directory structure -----


def test_format_webp_changes_extension_preserves_directory(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    _make_source_image(input_dir / "hooks" / "treble" / "a.png")

    run_process(
        input_dir, output_dir, remove_background_fn=fake_remove_background, workers=1, output_format="webp"
    )

    assert (output_dir / "hooks" / "treble" / "a.webp").exists()
    assert not (output_dir / "hooks" / "treble" / "a.png").exists()


# --- 13-14. --limit across recursive results, deterministic ordering ----


def test_limit_applies_across_recursive_results(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    _make_source_image(input_dir / "a.jpg")
    _make_source_image(input_dir / "hooks" / "b.jpg")
    _make_source_image(input_dir / "hooks" / "treble" / "c.jpg")
    _make_source_image(input_dir / "alarms" / "d.jpg")

    results = run_process(
        input_dir, output_dir, remove_background_fn=fake_remove_background, workers=2, limit=2
    )

    assert len(results) == 2


def test_limit_ordering_is_deterministic_and_matches_discover_images(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir_1 = tmp_path / "out1"
    output_dir_2 = tmp_path / "out2"
    # Created deliberately out of alphabetical/nesting order, to prove
    # sorting -- not creation order or directory-walk order -- decides
    # what "the first N" means.
    _make_source_image(input_dir / "zzz.jpg")
    _make_source_image(input_dir / "alarms" / "d.jpg")
    _make_source_image(input_dir / "hooks" / "treble" / "c.jpg")
    _make_source_image(input_dir / "hooks" / "b.jpg")
    _make_source_image(input_dir / "aaa.jpg")

    results_1 = run_process(
        input_dir, output_dir_1, remove_background_fn=fake_remove_background, workers=1, limit=3
    )
    results_2 = run_process(
        input_dir, output_dir_2, remove_background_fn=fake_remove_background, workers=1, limit=3
    )

    assert len(results_1) == 3
    assert sorted(r.source for r in results_1) == sorted(r.source for r in results_2)

    # And matches discover_images()'s own sorted order exactly -- the
    # ordering `run_process` applies --limit against.
    expected_sources = {str(p) for p in discover_images(input_dir)[:3]}
    assert {r.source for r in results_1} == expected_sources


# --- 15. output directories created automatically -----------------------


def test_output_directories_created_automatically(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    _make_source_image(input_dir / "hooks" / "treble" / "small" / "a.jpg")

    assert not output_dir.exists()

    run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=1)

    assert (output_dir / "hooks" / "treble" / "small" / "a.jpg").exists()
    assert (output_dir / "hooks").is_dir()
    assert (output_dir / "hooks" / "treble").is_dir()


# --- 16. input files remain untouched ------------------------------------


def test_input_files_remain_untouched_recursively(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    paths = [
        input_dir / "a.jpg",
        input_dir / "hooks" / "b.png",
        input_dir / "hooks" / "treble" / "c.webp",
    ]
    for p in paths:
        _make_source_image(p)
    original_bytes = {p: p.read_bytes() for p in paths}

    run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=2)

    for p in paths:
        assert p.exists()
        assert p.read_bytes() == original_bytes[p]


# --- 17. output-inside-input / input-equals-output are rejected --------


def test_output_inside_input_is_rejected(tmp_path: Path):
    input_dir = tmp_path / "images"
    output_dir = input_dir / "processed"
    _make_source_image(input_dir / "a.jpg")

    with pytest.raises(ValueError, match="inside"):
        run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=1)

    # rejected before any processing began -- nothing was created or written
    assert not output_dir.exists()


def test_output_nested_deep_inside_input_is_rejected(tmp_path: Path):
    input_dir = tmp_path / "images"
    output_dir = input_dir / "a" / "b" / "processed"
    _make_source_image(input_dir / "a.jpg")

    with pytest.raises(ValueError):
        run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=1)

    assert not output_dir.exists()


def test_output_equal_to_input_is_rejected(tmp_path: Path):
    input_dir = tmp_path / "images"
    _make_source_image(input_dir / "a.jpg")

    with pytest.raises(ValueError, match="same directory"):
        run_process(input_dir, input_dir, remove_background_fn=fake_remove_background, workers=1)


# --- 18. --save-mask preserves nested directory structure ---------------


def test_save_mask_preserves_nested_directory_structure(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    _make_source_image(input_dir / "hooks" / "a.jpg", size=(40, 40), color=(200, 200, 200))

    run_process(input_dir, output_dir, workers=1, mode="edge-background", save_mask=True)

    assert (output_dir / "hooks" / "a.jpg").exists()
    assert (output_dir / "hooks" / "a.mask.png").exists()
