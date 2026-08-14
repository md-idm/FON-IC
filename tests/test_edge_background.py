"""Unit tests for the non-AI --mode edge-background algorithm.

fon_image_cleaner.background.compute_background_mask / apply_background_mask
/ build_edge_background_remove_fn are pure Pillow+numpy+scipy functions --
no rembg, no AI model, and no monkeypatching needed to test them for real.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from fon_image_cleaner.background import (
    DEFAULT_BACKGROUND_TOLERANCE,
    apply_background_mask,
    build_edge_background_remove_fn,
    compute_background_mask,
    run_process,
)

BACKGROUND_GRAY = (200, 200, 200)
PRODUCT_RED = (220, 30, 30)


def _solid(size, color) -> Image.Image:
    return Image.new("RGB", size, color)


def _fill_rect(img: Image.Image, box, color) -> None:
    left, top, right, bottom = box
    for x in range(left, right):
        for y in range(top, bottom):
            img.putpixel((x, y), color)


def _mask_pixel(mask: Image.Image, x: int, y: int) -> int:
    return mask.load()[x, y]


# --- 1. uniform gray border becomes white -----------------------------


def test_uniform_gray_border_becomes_white_end_to_end(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    img = _solid((100, 100), BACKGROUND_GRAY)
    _fill_rect(img, (30, 30, 70, 70), PRODUCT_RED)
    # PNG (lossless) so the exact-white assertion below isn't confounded by
    # JPEG quantization noise near a flat white region.
    img.save(input_dir / "a.png")

    # Default (non-zero) padding, so the canvas has an actual white margin
    # at its corners to check -- a square product at padding=0 fills a
    # square canvas edge-to-edge, leaving no margin at all.
    results = run_process(input_dir, output_dir, canvas_size=200, workers=1, mode="edge-background")

    assert results[0].status == "processed"
    with Image.open(output_dir / "a.png") as out_img:
        out_img = out_img.convert("RGB")
        assert out_img.getpixel((0, 0)) == (255, 255, 255)
        assert out_img.getpixel((out_img.width - 1, out_img.height - 1)) == (255, 255, 255)


def test_uniform_gray_border_becomes_white_in_mask():
    img = _solid((60, 60), BACKGROUND_GRAY)
    _fill_rect(img, (20, 20, 40, 40), PRODUCT_RED)

    mask, _bg_rgb = compute_background_mask(img, tolerance=10)

    assert _mask_pixel(mask, 0, 0) == 255
    assert _mask_pixel(mask, 59, 59) == 255
    assert _mask_pixel(mask, 0, 30) == 255
    assert _mask_pixel(mask, 30, 0) == 255


# --- 2. gray object inside image is preserved --------------------------


def test_gray_object_enclosed_in_product_is_preserved():
    img = _solid((100, 100), BACKGROUND_GRAY)
    _fill_rect(img, (20, 20, 80, 80), PRODUCT_RED)
    # A gray square fully enclosed inside the red product -- same color as
    # the real background, but walled off from it on every side.
    _fill_rect(img, (45, 45, 55, 55), BACKGROUND_GRAY)

    mask, _bg_rgb = compute_background_mask(img, tolerance=10)

    assert _mask_pixel(mask, 50, 50) == 0  # enclosed gray patch -> kept
    assert _mask_pixel(mask, 25, 25) == 0  # surrounding red product -> kept
    assert _mask_pixel(mask, 0, 0) == 255  # real background -> removed


# --- 3. gray handle connected to differently colored product is preserved


def test_gray_handle_attached_to_product_is_preserved():
    img = _solid((120, 120), BACKGROUND_GRAY)
    # A red "head" ...
    _fill_rect(img, (30, 30, 90, 90), PRODUCT_RED)
    # ... with a gray "handle" protruding from it, fully inside the head's
    # footprint on the far side (never reaching the outer background).
    _fill_rect(img, (55, 35, 65, 85), BACKGROUND_GRAY)

    mask, _bg_rgb = compute_background_mask(img, tolerance=10)

    # Every handle pixel must be kept -- walled off by red on both sides.
    for y in range(36, 85):
        assert _mask_pixel(mask, 60, y) == 0, f"handle pixel at y={y} was wrongly marked background"
    assert _mask_pixel(mask, 0, 0) == 255


# --- 4. only regions connected to outer border are removed -------------


def test_only_border_connected_regions_are_removed():
    img = _solid((150, 150), BACKGROUND_GRAY)
    # A large red product covering most of the frame, touching the border
    # on the left and top so real background remains only on the right/bottom.
    _fill_rect(img, (0, 0, 100, 100), PRODUCT_RED)
    # A gray patch inside the red product, NOT connected to the real
    # background at all.
    _fill_rect(img, (30, 30, 50, 50), BACKGROUND_GRAY)

    mask, _bg_rgb = compute_background_mask(img, tolerance=10)

    # Real background (bottom-right, connected to the border) -> removed.
    assert _mask_pixel(mask, 140, 140) == 255
    # Enclosed same-colored patch (not connected to the border) -> kept.
    assert _mask_pixel(mask, 40, 40) == 0
    # Surrounding red product -> kept.
    assert _mask_pixel(mask, 10, 10) == 0


# --- 5. isolated gray interior area remains foreground ------------------


def test_isolated_gray_interior_area_remains_foreground_end_to_end(tmp_path: Path):
    """If the interior gray island were (incorrectly) treated as background,
    it would show up as a WHITE hole punched into the middle of the red
    product (trim only crops the outer bbox, it can't "heal" an interior
    hole) -- so finding the island's own gray color survives at its exact
    expected location is a precise, non-trivial check that the red product
    being present is not enough to satisfy on its own.
    """
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    img = _solid((200, 200), BACKGROUND_GRAY)
    _fill_rect(img, (20, 20, 180, 180), PRODUCT_RED)
    # An isolated gray "island" deep inside the product, far from every edge.
    _fill_rect(img, (90, 90, 110, 110), BACKGROUND_GRAY)
    img.save(input_dir / "a.png")

    # canvas_size=400, padding=0 -> the 160x160 red bbox is scaled by
    # exactly 2.5x to fill the canvas, so the island (local coords
    # (70,70)-(90,90) within the trimmed crop) lands at (175,175)-(225,225);
    # its center (200,200) is far from any edge, so LANCZOS resize should
    # reproduce its flat interior color almost exactly.
    run_process(
        input_dir, output_dir, canvas_size=400, padding_ratio=0.0, workers=1, mode="edge-background"
    )

    with Image.open(output_dir / "a.png") as out_img:
        r, g, b = out_img.convert("RGB").getpixel((200, 200))
        assert (abs(r - 200) <= 5 and abs(g - 200) <= 5 and abs(b - 200) <= 5), (
            f"expected the surviving gray island (~{BACKGROUND_GRAY}) at (200,200), got {(r, g, b)}"
        )


# --- 6. tolerance controls background selection -------------------------


def test_tolerance_controls_background_selection():
    slightly_off_gray = (215, 215, 215)  # 15 away from BACKGROUND_GRAY on every channel
    img = _solid((80, 80), BACKGROUND_GRAY)
    # A patch that touches the border (so it WOULD be reachable by the
    # flood-fill) but is a slightly different shade of gray.
    _fill_rect(img, (0, 30, 20, 50), slightly_off_gray)

    tight_mask, _ = compute_background_mask(img, tolerance=5)
    loose_mask, _ = compute_background_mask(img, tolerance=20)

    # With a tight tolerance the off-shade patch is NOT close enough to the
    # sampled background color, so it is not absorbed into the background.
    assert _mask_pixel(tight_mask, 10, 40) == 0
    # With a looser tolerance covering the 15-unit difference, the same
    # border-connected patch IS absorbed.
    assert _mask_pixel(loose_mask, 10, 40) == 255


# --- 7. foreground RGB pixels are byte-for-byte unchanged ---------------


def test_foreground_rgb_bytes_unchanged_before_final_save():
    img = _solid((60, 60), BACKGROUND_GRAY)
    # A product region with distinctive, non-uniform pixel values so any
    # rounding/blending bug would be caught by an exact comparison.
    for x in range(20, 40):
        for y in range(20, 40):
            img.putpixel((x, y), ((x * 7) % 256, (y * 11) % 256, (x + y) % 256))

    original_pixels = {(x, y): img.getpixel((x, y)) for x in range(20, 40) for y in range(20, 40)}

    mask, _bg_rgb = compute_background_mask(img, tolerance=10)
    result = apply_background_mask(img, mask)

    for (x, y), original_rgb in original_pixels.items():
        r, g, b, a = result.getpixel((x, y))
        assert (r, g, b) == original_rgb, f"foreground pixel at {(x, y)} changed: {original_rgb} -> {(r, g, b)}"
        assert a == 255

    # Background pixels: pure white, fully transparent.
    r, g, b, a = result.getpixel((0, 0))
    assert (r, g, b, a) == (255, 255, 255, 0)


def test_remove_background_fn_output_matches_apply_background_mask():
    """build_edge_background_remove_fn must be exactly compute+apply, not a
    different code path that could diverge in behavior."""
    img = _solid((50, 50), BACKGROUND_GRAY).convert("RGBA")

    fn = build_edge_background_remove_fn(tolerance=DEFAULT_BACKGROUND_TOLERANCE)
    via_fn = fn(img)

    mask, _bg_rgb = compute_background_mask(img)
    via_direct = apply_background_mask(img, mask)

    assert via_fn.tobytes() == via_direct.tobytes()


# --- 8. input is never modified -----------------------------------------


def test_input_never_modified_by_edge_background_mode(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    img = _solid((80, 80), BACKGROUND_GRAY)
    _fill_rect(img, (20, 20, 60, 60), PRODUCT_RED)
    src = input_dir / "a.webp"
    img.save(src)
    original_bytes = src.read_bytes()

    run_process(input_dir, output_dir, workers=1, mode="edge-background")

    assert src.read_bytes() == original_bytes


# --- 9. existing output skip still works with edge-background mode ------


def _counting_fn(inner_fn, calls):
    def _fn(image):
        calls.append(1)
        return inner_fn(image)

    return _fn


def test_existing_output_skip_still_works_with_edge_background_mode(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    img = _solid((60, 60), BACKGROUND_GRAY)
    _fill_rect(img, (10, 10, 50, 50), PRODUCT_RED)
    img.save(input_dir / "a.jpg")
    (output_dir / "a.jpg").write_bytes(b"EXISTING")

    calls: list = []
    real_fn = build_edge_background_remove_fn()
    results = run_process(
        input_dir,
        output_dir,
        workers=1,
        mode="edge-background",
        remove_background_fn=_counting_fn(real_fn, calls),
    )

    assert results[0].status == "skipped"
    assert results[0].error == "output_exists"
    assert (output_dir / "a.jpg").read_bytes() == b"EXISTING"
    assert calls == []  # the edge-background algorithm was never invoked


def test_force_reprocesses_existing_output_in_edge_background_mode(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    img = _solid((60, 60), BACKGROUND_GRAY)
    _fill_rect(img, (10, 10, 50, 50), PRODUCT_RED)
    img.save(input_dir / "a.jpg")
    (output_dir / "a.jpg").write_bytes(b"EXISTING")

    calls: list = []
    real_fn = build_edge_background_remove_fn()
    results = run_process(
        input_dir,
        output_dir,
        workers=1,
        mode="edge-background",
        force=True,
        remove_background_fn=_counting_fn(real_fn, calls),
    )

    assert results[0].status == "processed"
    assert (output_dir / "a.jpg").read_bytes() != b"EXISTING"
    assert len(calls) == 1
