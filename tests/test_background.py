from pathlib import Path

import pytest
from PIL import Image, ImageChops

from fon_image_cleaner.background import (
    DEFAULT_PADDING_RATIO,
    build_remove_background_fn,
    compose_on_white_canvas,
    pil_format_for_suffix,
    resolve_output_suffix,
    run_compare_models,
    run_process,
    trim_transparent_margins,
)

MARGIN = 10


def fake_remove_background(image: Image.Image) -> Image.Image:
    """Stand-in AI segmentation: keeps everything except a fixed-width
    transparent border, simulating a model that found a rectangular subject."""
    width, height = image.size
    result = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    right = max(MARGIN + 1, width - MARGIN)
    bottom = max(MARGIN + 1, height - MARGIN)
    subject = image.convert("RGBA").crop((MARGIN, MARGIN, right, bottom))
    result.paste(subject, (MARGIN, MARGIN))
    return result


def fake_remove_background_empty(image: Image.Image) -> Image.Image:
    """Stand-in for a model that found no subject at all."""
    return Image.new("RGBA", image.size, (0, 0, 0, 0))


def _make_source_image(path: Path, size=(300, 200), color=(0, 0, 0)) -> None:
    Image.new("RGB", size, color).save(path)


def _non_white_bbox(img: Image.Image):
    white = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img.convert("RGB"), white)
    return diff.getbbox()


# --- pure pipeline function tests -----------------------------------------


def test_trim_transparent_margins_crops_to_content():
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    subject = Image.new("RGBA", (40, 20), (255, 0, 0, 255))
    img.paste(subject, (30, 40))

    trimmed = trim_transparent_margins(img)

    assert trimmed.size == (40, 20)


def test_compose_on_white_canvas_size_and_background():
    subject = Image.new("RGBA", (40, 20), (255, 0, 0, 255))

    canvas = compose_on_white_canvas(subject, canvas_size=200, padding_ratio=0.0)

    assert canvas.size == (200, 200)
    assert canvas.getpixel((0, 0)) == (255, 255, 255)
    assert canvas.getpixel((199, 199)) == (255, 255, 255)


# --- end-to-end pipeline tests (rembg mocked out) --------------------------


def test_input_file_is_never_modified(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    src = input_dir / "a.jpg"
    _make_source_image(src)
    original_bytes = src.read_bytes()

    run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=1)

    assert src.read_bytes() == original_bytes


def test_output_is_1200x1200_with_white_background(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _make_source_image(input_dir / "a.jpg", size=(400, 300))

    results = run_process(
        input_dir, output_dir, canvas_size=1200, remove_background_fn=fake_remove_background, workers=1
    )

    assert results[0].status == "processed"
    out_img = Image.open(output_dir / "a.jpg")
    assert out_img.size == (1200, 1200)
    corner = out_img.convert("RGB").getpixel((0, 0))
    assert corner == (255, 255, 255)


def test_aspect_ratio_is_preserved(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    source_size = (400, 200)
    _make_source_image(input_dir / "a.jpg", size=source_size)

    run_process(
        input_dir,
        output_dir,
        canvas_size=1200,
        padding_ratio=0.0,
        remove_background_fn=fake_remove_background,
        workers=1,
    )

    out_img = Image.open(output_dir / "a.jpg")
    bbox = _non_white_bbox(out_img)
    assert bbox is not None
    left, top, right, bottom = bbox
    actual_ratio = (right - left) / (bottom - top)
    expected_ratio = (source_size[0] - 2 * MARGIN) / (source_size[1] - 2 * MARGIN)
    assert abs(actual_ratio - expected_ratio) < 0.05


def test_nested_directory_structure_is_preserved(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    (input_dir / "reels").mkdir(parents=True)
    _make_source_image(input_dir / "reels" / "a.jpg")

    run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=1)

    assert (output_dir / "reels" / "a.jpg").exists()


def test_skip_existing_output_by_default(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    _make_source_image(input_dir / "a.jpg")
    (output_dir / "a.jpg").write_bytes(b"EXISTING")

    results = run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=1)

    assert results[0].status == "skipped"
    assert (output_dir / "a.jpg").read_bytes() == b"EXISTING"


def test_force_overwrites_existing_output(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    _make_source_image(input_dir / "a.jpg")
    (output_dir / "a.jpg").write_bytes(b"EXISTING")

    results = run_process(
        input_dir, output_dir, remove_background_fn=fake_remove_background, workers=1, force=True
    )

    assert results[0].status == "processed"
    assert (output_dir / "a.jpg").read_bytes() != b"EXISTING"


def test_failed_image_is_isolated_from_batch(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _make_source_image(input_dir / "good.jpg")
    (input_dir / "bad.jpg").write_bytes(b"not a real image")

    results = run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=2)

    statuses = {Path(r.source).name: r.status for r in results}
    assert statuses["good.jpg"] == "processed"
    assert statuses["bad.jpg"] == "failed"
    assert (output_dir / "good.jpg").exists()
    assert not (output_dir / "bad.jpg").exists()


def test_limit_caps_number_of_images_processed(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        _make_source_image(input_dir / name)

    results = run_process(
        input_dir, output_dir, remove_background_fn=fake_remove_background, workers=1, limit=2
    )

    assert len(results) == 2


def test_needs_review_when_no_subject_detected(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _make_source_image(input_dir / "a.jpg")

    results = run_process(
        input_dir, output_dir, remove_background_fn=fake_remove_background_empty, workers=1
    )

    assert results[0].status == "needs_review"
    assert (output_dir / "a.jpg").exists()


# --- output format: preserve (default) vs forced --format ------------------


def test_resolve_output_suffix_preserve_keeps_source_extension():
    assert resolve_output_suffix(Path("a.jpg"), "preserve") == ".jpg"
    assert resolve_output_suffix(Path("a.jpeg"), "preserve") == ".jpeg"
    assert resolve_output_suffix(Path("a.png"), "preserve") == ".png"
    assert resolve_output_suffix(Path("a.webp"), "preserve") == ".webp"


def test_resolve_output_suffix_forced_format_ignores_source_extension():
    assert resolve_output_suffix(Path("a.png"), "webp") == ".webp"
    assert resolve_output_suffix(Path("a.webp"), "jpg") == ".jpg"
    assert resolve_output_suffix(Path("a.jpg"), "png") == ".png"


def test_resolve_output_suffix_rejects_invalid_format():
    with pytest.raises(ValueError):
        resolve_output_suffix(Path("a.jpg"), "gif")


def test_pil_format_for_suffix_mapping():
    assert pil_format_for_suffix(".jpg") == "JPEG"
    assert pil_format_for_suffix(".jpeg") == "JPEG"
    assert pil_format_for_suffix(".png") == "PNG"
    assert pil_format_for_suffix(".webp") == "WEBP"


def test_run_process_rejects_invalid_output_format(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _make_source_image(input_dir / "a.jpg")

    with pytest.raises(ValueError):
        run_process(
            input_dir,
            output_dir,
            remove_background_fn=fake_remove_background,
            workers=1,
            output_format="gif",
        )


def test_png_source_remains_png_by_default(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _make_source_image(input_dir / "a.png", size=(300, 200))

    results = run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=1)

    assert results[0].status == "processed"
    dest = output_dir / "a.png"
    assert dest.exists()
    assert not (output_dir / "a.webp").exists()
    with Image.open(dest) as out_img:
        assert out_img.format == "PNG"
        assert out_img.convert("RGB").getpixel((0, 0)) == (255, 255, 255)


def test_jpeg_source_remains_jpeg_by_default(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _make_source_image(input_dir / "a.jpg", size=(300, 200))

    results = run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=1)

    assert results[0].status == "processed"
    dest = output_dir / "a.jpg"
    assert dest.exists()
    assert not (output_dir / "a.webp").exists()
    with Image.open(dest) as out_img:
        assert out_img.format == "JPEG"
        assert out_img.mode == "RGB"
        assert out_img.getpixel((0, 0)) == (255, 255, 255)


def test_jpeg_extension_dot_jpeg_is_preserved_not_renamed_to_jpg(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _make_source_image(input_dir / "a.jpeg", size=(300, 200))

    run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=1)

    dest = output_dir / "a.jpeg"
    assert dest.exists()
    with Image.open(dest) as out_img:
        assert out_img.format == "JPEG"


def test_webp_source_remains_webp_by_default(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _make_source_image(input_dir / "a.webp", size=(300, 200))

    results = run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=1)

    assert results[0].status == "processed"
    dest = output_dir / "a.webp"
    assert dest.exists()
    with Image.open(dest) as out_img:
        assert out_img.format == "WEBP"
        assert out_img.convert("RGB").getpixel((0, 0)) == (255, 255, 255)


def test_mixed_input_formats_each_preserve_their_own_extension(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _make_source_image(input_dir / "a.jpg")
    _make_source_image(input_dir / "b.png")
    _make_source_image(input_dir / "c.webp")

    results = run_process(input_dir, output_dir, remove_background_fn=fake_remove_background, workers=2)

    assert {r.status for r in results} == {"processed"}
    assert (output_dir / "a.jpg").exists()
    assert (output_dir / "b.png").exists()
    assert (output_dir / "c.webp").exists()
    with Image.open(output_dir / "a.jpg") as img:
        assert img.format == "JPEG"
    with Image.open(output_dir / "b.png") as img:
        assert img.format == "PNG"
    with Image.open(output_dir / "c.webp") as img:
        assert img.format == "WEBP"


def test_explicit_format_webp_converts_every_input(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _make_source_image(input_dir / "a.jpg")
    _make_source_image(input_dir / "b.png")

    results = run_process(
        input_dir,
        output_dir,
        remove_background_fn=fake_remove_background,
        workers=2,
        output_format="webp",
    )

    assert {r.status for r in results} == {"processed"}
    assert (output_dir / "a.webp").exists()
    assert (output_dir / "b.webp").exists()
    assert not (output_dir / "a.jpg").exists()
    assert not (output_dir / "b.png").exists()
    with Image.open(output_dir / "a.webp") as img:
        assert img.format == "WEBP"
    with Image.open(output_dir / "b.webp") as img:
        assert img.format == "WEBP"


def test_explicit_format_jpg_converts_every_input(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _make_source_image(input_dir / "a.png")
    _make_source_image(input_dir / "b.webp")

    run_process(
        input_dir, output_dir, remove_background_fn=fake_remove_background, workers=2, output_format="jpg"
    )

    assert (output_dir / "a.jpg").exists()
    assert (output_dir / "b.jpg").exists()
    with Image.open(output_dir / "a.jpg") as img:
        assert img.format == "JPEG"
        assert img.mode == "RGB"


def test_explicit_format_png_converts_every_input(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _make_source_image(input_dir / "a.jpg")
    _make_source_image(input_dir / "b.webp")

    run_process(
        input_dir, output_dir, remove_background_fn=fake_remove_background, workers=2, output_format="png"
    )

    assert (output_dir / "a.png").exists()
    assert (output_dir / "b.png").exists()
    with Image.open(output_dir / "a.png") as img:
        assert img.format == "PNG"
        assert img.convert("RGB").getpixel((0, 0)) == (255, 255, 255)


def test_output_extension_always_matches_actual_encoded_format(tmp_path):
    """For every source/--format combination, whatever extension the file
    was written with must match what Pillow actually encoded it as."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _make_source_image(input_dir / "a.jpg")
    _make_source_image(input_dir / "b.png")
    _make_source_image(input_dir / "c.webp")

    for output_format in ("preserve", "webp", "jpg", "png"):
        run_dir = output_dir / output_format
        run_process(
            input_dir,
            run_dir,
            remove_background_fn=fake_remove_background,
            workers=2,
            output_format=output_format,
        )
        for path in run_dir.rglob("*"):
            if not path.is_file():
                continue
            with Image.open(path) as img:
                expected = pil_format_for_suffix(path.suffix)
                assert img.format == expected, f"{path} has extension {path.suffix} but Pillow read it as {img.format}"


# --- --model / session reuse (rembg mocked at the module level) ------------


def test_model_selection_passed_to_new_session(monkeypatch):
    import rembg

    calls = []

    def fake_new_session(model_name):
        calls.append(model_name)
        return f"session-for-{model_name}"

    def fake_remove(image, session=None, post_process_mask=False, alpha_matting=False):
        return image.convert("RGBA")

    monkeypatch.setattr(rembg, "new_session", fake_new_session)
    monkeypatch.setattr(rembg, "remove", fake_remove)

    fn = build_remove_background_fn("isnet-general-use")
    fn(Image.new("RGB", (10, 10), (255, 0, 0)))

    assert calls == ["isnet-general-use"]


def test_session_is_reused_not_recreated_per_call(monkeypatch):
    import rembg

    new_session_calls = []
    used_sessions = []

    def fake_new_session(model_name):
        new_session_calls.append(model_name)
        return object()  # a distinct sentinel per call

    def fake_remove(image, session=None, post_process_mask=False, alpha_matting=False):
        used_sessions.append(session)
        return image.convert("RGBA")

    monkeypatch.setattr(rembg, "new_session", fake_new_session)
    monkeypatch.setattr(rembg, "remove", fake_remove)

    fn = build_remove_background_fn("u2net")
    img = Image.new("RGB", (10, 10), (0, 0, 0))
    fn(img)
    fn(img)
    fn(img)

    assert new_session_calls == ["u2net"]  # exactly once, not once per call
    assert len(used_sessions) == 3
    assert used_sessions[0] is used_sessions[1] is used_sessions[2]


def test_run_process_creates_session_once_for_whole_batch(monkeypatch, tmp_path):
    """End-to-end: run_process must build the (expensive) rembg session
    exactly once for the whole batch, and every worker thread must reuse
    that same session -- never one session per image."""
    import rembg

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        _make_source_image(input_dir / name)

    new_session_calls = []
    used_sessions = []

    def fake_new_session(model_name):
        new_session_calls.append(model_name)
        return object()

    def fake_remove(image, session=None, post_process_mask=False, alpha_matting=False):
        used_sessions.append(session)
        return fake_remove_background(image)

    monkeypatch.setattr(rembg, "new_session", fake_new_session)
    monkeypatch.setattr(rembg, "remove", fake_remove)

    results = run_process(input_dir, output_dir, workers=3, model="birefnet-general")

    assert new_session_calls == ["birefnet-general"]
    assert len(used_sessions) == 3
    assert len(set(id(s) for s in used_sessions)) == 1
    assert {r.status for r in results} == {"processed"}


def test_post_process_mask_and_alpha_matting_are_passed_through(monkeypatch):
    import rembg

    captured = {}

    def fake_new_session(model_name):
        return "the-session"

    def fake_remove(image, session=None, post_process_mask=False, alpha_matting=False):
        captured["session"] = session
        captured["post_process_mask"] = post_process_mask
        captured["alpha_matting"] = alpha_matting
        return image.convert("RGBA")

    monkeypatch.setattr(rembg, "new_session", fake_new_session)
    monkeypatch.setattr(rembg, "remove", fake_remove)

    fn = build_remove_background_fn("u2net", post_process_mask=True, alpha_matting=True)
    fn(Image.new("RGB", (10, 10)))

    assert captured == {"session": "the-session", "post_process_mask": True, "alpha_matting": True}


def test_post_process_mask_and_alpha_matting_default_off(monkeypatch):
    import rembg

    captured = {}

    def fake_new_session(model_name):
        return "the-session"

    def fake_remove(image, session=None, post_process_mask=False, alpha_matting=False):
        captured["post_process_mask"] = post_process_mask
        captured["alpha_matting"] = alpha_matting
        return image.convert("RGBA")

    monkeypatch.setattr(rembg, "new_session", fake_new_session)
    monkeypatch.setattr(rembg, "remove", fake_remove)

    fn = build_remove_background_fn("u2net")
    fn(Image.new("RGB", (10, 10)))

    assert captured == {"post_process_mask": False, "alpha_matting": False}


# --- padding -----------------------------------------------------------


def test_padding_zero_fills_canvas_without_cropping():
    subject = Image.new("RGBA", (100, 50), (0, 0, 0, 255))  # 2:1, fully opaque

    canvas = compose_on_white_canvas(subject, canvas_size=200, padding_ratio=0.0)

    bbox = _non_white_bbox(canvas)
    assert bbox is not None
    left, top, right, bottom = bbox
    assert canvas.size == (200, 200)
    # the limiting (width) dimension must span the full canvas -- no
    # artificial padding, and the product itself is never cropped since
    # resize_preserving_aspect only ever scales down to fit, never crops.
    assert right - left == 200
    assert bottom - top == 100  # 2:1 aspect preserved exactly


def test_padding_default_0_02_leaves_small_margin(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _make_source_image(input_dir / "a.jpg", size=(400, 200))

    run_process(
        input_dir,
        output_dir,
        canvas_size=1200,
        remove_background_fn=fake_remove_background,
        workers=1,
        # padding_ratio intentionally omitted -- exercises the new default
    )

    with Image.open(output_dir / "a.jpg") as out_img:
        bbox = _non_white_bbox(out_img)
        assert bbox is not None
        left, top, right, bottom = bbox
        width = right - left
        # with 2% padding per side the product should span roughly 96% of
        # the canvas -- comfortably less than full width, but far more than
        # the old 8%-padding default (~84%) would have allowed.
        assert 1100 <= width < 1200


def test_padding_can_be_set_to_0_02_explicitly(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _make_source_image(input_dir / "a.jpg", size=(400, 200))

    run_process(
        input_dir,
        output_dir,
        canvas_size=1200,
        padding_ratio=0.02,
        remove_background_fn=fake_remove_background,
        workers=1,
    )

    with Image.open(output_dir / "a.jpg") as out_img:
        bbox = _non_white_bbox(out_img)
        left, top, right, bottom = bbox
        assert 1100 <= (right - left) < 1200


def test_default_padding_ratio_constant_is_0_02():
    assert DEFAULT_PADDING_RATIO == 0.02


# --- --compare-models --------------------------------------------------


def test_compare_models_isolates_output_directories(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _make_source_image(input_dir / "a.jpg")

    results_by_model = run_compare_models(
        input_dir,
        output_dir,
        models=["u2net", "isnet-general-use", "birefnet-general"],
        limit=5,
        remove_background_fn_factory=lambda model_name: fake_remove_background,
    )

    assert set(results_by_model.keys()) == {"u2net", "isnet-general-use", "birefnet-general"}
    paths = set()
    for model_name in results_by_model:
        dest = output_dir / model_name / "a.jpg"
        assert dest.exists()
        paths.add(dest)
    assert len(paths) == 3  # three genuinely distinct files, none overwritten


def test_compare_models_creates_session_once_per_model_not_per_image(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    for name in ("a.jpg", "b.jpg"):
        _make_source_image(input_dir / name)

    factory_calls = []

    def factory(model_name):
        factory_calls.append(model_name)
        return fake_remove_background

    run_compare_models(
        input_dir,
        output_dir,
        models=["u2net", "isnet-general-use"],
        limit=5,
        remove_background_fn_factory=factory,
    )

    assert factory_calls == ["u2net", "isnet-general-use"]


def test_compare_models_respects_limit(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
        _make_source_image(input_dir / name)

    results_by_model = run_compare_models(
        input_dir,
        output_dir,
        models=["u2net", "isnet-general-use"],
        limit=2,
        remove_background_fn_factory=lambda model_name: fake_remove_background,
    )

    for results in results_by_model.values():
        assert len(results) == 2


def test_compare_models_does_not_modify_input(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    src = input_dir / "a.jpg"
    _make_source_image(src)
    original_bytes = src.read_bytes()

    run_compare_models(
        input_dir,
        output_dir,
        models=["u2net", "isnet-general-use", "birefnet-general"],
        limit=5,
        remove_background_fn_factory=lambda model_name: fake_remove_background,
    )

    assert src.read_bytes() == original_bytes


# --- resume / idempotency: skip existing outputs before touching rembg -----


def _counting_fn(calls):
    """Wraps fake_remove_background so tests can assert exactly how many
    times (if any) the background-removal function was actually invoked."""

    def _fn(image):
        calls.append(1)
        return fake_remove_background(image)

    return _fn


def test_existing_png_skipped_in_preserve_mode(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    _make_source_image(input_dir / "a.png")
    (output_dir / "a.png").write_bytes(b"EXISTING-PNG")

    calls = []
    results = run_process(input_dir, output_dir, remove_background_fn=_counting_fn(calls), workers=1)

    assert results[0].status == "skipped"
    assert results[0].error == "output_exists"
    assert (output_dir / "a.png").read_bytes() == b"EXISTING-PNG"
    assert calls == []


def test_existing_jpg_skipped(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    _make_source_image(input_dir / "a.jpg")
    (output_dir / "a.jpg").write_bytes(b"EXISTING-JPG")

    calls = []
    results = run_process(input_dir, output_dir, remove_background_fn=_counting_fn(calls), workers=1)

    assert results[0].status == "skipped"
    assert results[0].error == "output_exists"
    assert (output_dir / "a.jpg").read_bytes() == b"EXISTING-JPG"
    assert calls == []


def test_existing_webp_skipped(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    _make_source_image(input_dir / "a.webp")
    (output_dir / "a.webp").write_bytes(b"EXISTING-WEBP")

    calls = []
    results = run_process(input_dir, output_dir, remove_background_fn=_counting_fn(calls), workers=1)

    assert results[0].status == "skipped"
    assert results[0].error == "output_exists"
    assert (output_dir / "a.webp").read_bytes() == b"EXISTING-WEBP"
    assert calls == []


def test_nested_existing_output_skipped(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    (input_dir / "sub").mkdir(parents=True)
    (output_dir / "sub").mkdir(parents=True)
    _make_source_image(input_dir / "sub" / "b.png")
    (output_dir / "sub" / "b.png").write_bytes(b"EXISTING")

    calls = []
    results = run_process(input_dir, output_dir, remove_background_fn=_counting_fn(calls), workers=1)

    assert results[0].status == "skipped"
    assert results[0].destination == str(output_dir / "sub" / "b.png")
    assert (output_dir / "sub" / "b.png").read_bytes() == b"EXISTING"
    assert calls == []


def test_force_regenerates_existing_output_and_calls_remove_background(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    _make_source_image(input_dir / "a.jpg")
    (output_dir / "a.jpg").write_bytes(b"EXISTING")

    calls = []
    results = run_process(
        input_dir, output_dir, remove_background_fn=_counting_fn(calls), workers=1, force=True
    )

    assert results[0].status == "processed"
    assert (output_dir / "a.jpg").read_bytes() != b"EXISTING"
    assert len(calls) == 1


def test_skipped_file_never_calls_remove_background_function(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    _make_source_image(input_dir / "a.jpg")
    (output_dir / "a.jpg").write_bytes(b"EXISTING")

    calls = []
    run_process(input_dir, output_dir, remove_background_fn=_counting_fn(calls), workers=1)

    assert calls == []


def test_mixed_batch_existing_skipped_missing_processed(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    _make_source_image(input_dir / "existing.jpg")
    _make_source_image(input_dir / "missing.jpg")
    (output_dir / "existing.jpg").write_bytes(b"EXISTING")

    calls = []
    results = run_process(input_dir, output_dir, remove_background_fn=_counting_fn(calls), workers=2)

    statuses = {Path(r.source).name: r.status for r in results}
    assert statuses["existing.jpg"] == "skipped"
    assert statuses["missing.jpg"] == "processed"
    assert (output_dir / "existing.jpg").read_bytes() == b"EXISTING"
    assert len(calls) == 1  # only the missing image triggered background removal


def test_all_outputs_existing_means_zero_processing_calls_and_no_session(tmp_path, monkeypatch):
    """If every requested image already has an output, run_process must
    return all-skipped results without ever building the rembg session."""
    import rembg

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    for name in ("a.jpg", "b.png", "c.webp"):
        _make_source_image(input_dir / name)
        (output_dir / name).write_bytes(b"EXISTING")

    new_session_calls = []

    def fake_new_session(model_name):
        new_session_calls.append(model_name)
        return object()

    def fake_remove(image, session=None, post_process_mask=False, alpha_matting=False):
        raise AssertionError("rembg.remove must never be called when every output already exists")

    monkeypatch.setattr(rembg, "new_session", fake_new_session)
    monkeypatch.setattr(rembg, "remove", fake_remove)

    # No remove_background_fn injected -- exercises the real lazy
    # session-building path inside run_process.
    results = run_process(input_dir, output_dir, workers=2)

    assert {r.status for r in results} == {"skipped"}
    assert len(results) == 3
    assert new_session_calls == []


def test_output_format_conversion_checks_correct_destination_extension(tmp_path):
    """input a.png + --format webp must check for an existing a.webp -- not
    a.png -- before deciding whether to skip."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    _make_source_image(input_dir / "a.png")
    # An unrelated a.png sitting in the output dir must NOT cause a skip --
    # the real destination for --format webp is a.webp.
    (output_dir / "a.png").write_bytes(b"UNRELATED")

    calls = []
    results = run_process(
        input_dir, output_dir, remove_background_fn=_counting_fn(calls), workers=1, output_format="webp"
    )

    assert results[0].status == "processed"
    assert len(calls) == 1
    assert (output_dir / "a.webp").exists()
    assert (output_dir / "a.png").read_bytes() == b"UNRELATED"  # untouched

    # Now pre-create the REAL destination (a.webp) and confirm it IS skipped.
    calls2 = []
    (output_dir / "a.webp").unlink()
    (output_dir / "a.webp").write_bytes(b"EXISTING-WEBP")
    results2 = run_process(
        input_dir, output_dir, remove_background_fn=_counting_fn(calls2), workers=1, output_format="webp"
    )

    assert results2[0].status == "skipped"
    assert calls2 == []
    assert (output_dir / "a.webp").read_bytes() == b"EXISTING-WEBP"
