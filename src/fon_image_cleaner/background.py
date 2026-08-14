"""Background-removal image pipeline: source image -> clean catalog image.

Pipeline: EXIF orientation -> AI background removal -> trim transparent
margins -> resize preserving aspect ratio -> center on a white canvas -> save.
Input files are only ever opened for reading and are never modified.

By default (`--format preserve`) the output keeps the source file's own
format and extension: .jpg/.jpeg -> JPEG, .png -> PNG, .webp -> WebP. Pass
`--format webp`/`jpg`/`png` to force every output to a single format instead.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image, ImageChops, ImageOps

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

VALID_OUTPUT_FORMATS = {"preserve", "webp", "jpg", "png"}

# Maps a (lowercased) file extension to the Pillow format name used to
# encode it -- the single source of truth so the on-disk extension and the
# actual encoded format can never disagree.
FORMAT_BY_EXTENSION = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
}

# Extension to force when --format explicitly requests a single format
# rather than preserving the source's own.
FORCED_EXTENSION_BY_FORMAT_FLAG = {
    "webp": ".webp",
    "jpg": ".jpg",
    "png": ".png",
}

# rembg/u2net segmentation models supported by --model. birefnet-general is
# the default because it holds onto semi-dark/gray product parts (e.g.
# handles) far better than u2net, which tends to erase them as background.
SUPPORTED_MODELS = ["u2net", "isnet-general-use", "birefnet-general"]
DEFAULT_MODEL = "birefnet-general"

# 8% padding per side left too much artificial white border around the
# product; 2% keeps a small margin without shrinking the product much.
DEFAULT_PADDING_RATIO = 0.02

# --mode: "ai" uses rembg (see build_remove_background_fn); "edge-background"
# is a conservative, non-AI algorithm that only ever removes the connected
# region reachable from the image border that matches the sampled border
# color -- it never segments the product, so it can't mistake a gray product
# part (e.g. a handle) for background just because AI segmentation would.
VALID_MODES = {"ai", "edge-background"}
DEFAULT_MODE = "ai"

# Chebyshev (per-channel max) distance in 0-255 RGB space: how far a pixel's
# color may be from the sampled border color and still count as background.
DEFAULT_BACKGROUND_TOLERANCE = 30.0

# Width (px) of the border band sampled from each edge to estimate the
# background color -- wide enough to get a robust median, narrow enough to
# stay clear of the product even when it sits close to the frame edge.
DEFAULT_EDGE_SAMPLE_WIDTH = 12


@dataclass
class ProcessResult:
    source: str
    destination: str
    status: str  # processed | needs_review | skipped | failed
    size: int = 0
    duration_ms: int = 0
    error: str = ""


RemoveBackgroundFn = Callable[[Image.Image], Image.Image]


def build_remove_background_fn(
    model_name: str = DEFAULT_MODEL,
    post_process_mask: bool = False,
    alpha_matting: bool = False,
) -> RemoveBackgroundFn:
    """Create ONE rembg session for `model_name` and return a closure that
    reuses it for every call.

    `rembg.new_session()` loads the ONNX model from disk/cache, which is
    comparatively expensive -- this must happen once per run (batch), never
    once per image. The returned closure is safe to share across the worker
    threads `run_process` uses: onnxruntime sessions support concurrent
    `Run()` calls from multiple threads.

    Imported lazily so importing this module (e.g. for pure-function unit
    tests, or --help) never requires rembg/onnxruntime to be installed or
    to pay its import/model-load cost.
    """
    from rembg import new_session, remove

    session = new_session(model_name)

    def _remove_background(image: Image.Image) -> Image.Image:
        result = remove(
            image,
            session=session,
            post_process_mask=post_process_mask,
            alpha_matting=alpha_matting,
        )
        if result.mode != "RGBA":
            result = result.convert("RGBA")
        return result

    return _remove_background


def _sample_background_color(rgb_array, edge_sample_width: int) -> "tuple[int, int, int]":
    """Robust background color estimate: the per-channel median over pixels
    sampled from an `edge_sample_width`-pixel band along all four borders
    (which inherently includes every corner), rather than trusting a single
    exact pixel -- a few stray non-background pixels in the sample band
    (e.g. the product touching the frame edge) don't skew a median the way
    they would an average.
    """
    import numpy as np

    height, width, _ = rgb_array.shape
    band = max(1, min(edge_sample_width, height // 2, width // 2))
    samples = np.concatenate(
        [
            rgb_array[:band, :, :].reshape(-1, 3),
            rgb_array[-band:, :, :].reshape(-1, 3),
            rgb_array[:, :band, :].reshape(-1, 3),
            rgb_array[:, -band:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    median = np.median(samples, axis=0)
    return tuple(int(round(v)) for v in median)


def compute_background_mask(
    image: Image.Image,
    tolerance: float = DEFAULT_BACKGROUND_TOLERANCE,
    edge_sample_width: int = DEFAULT_EDGE_SAMPLE_WIDTH,
) -> "tuple[Image.Image, tuple[int, int, int]]":
    """Detect the background region with a conservative, non-AI algorithm.

    1. Sample the background color from border/corner regions (robust
       median, not a single assumed RGB value).
    2. Flood-fill outward, seeded ONLY from the image border, through
       pixels within `tolerance` of that sampled color.
    3. A pixel is only ever marked background if it is BOTH close enough in
       color AND connected (4-connectivity) to the outer boundary through
       other such pixels -- an interior pixel that happens to match the
       background color but is walled off by product pixels (e.g. a gray
       handle) is never reached by the flood-fill and stays foreground,
       no matter how "background-like" its color looks in isolation.

    Never crops or resizes `image` -- the returned mask is exactly the same
    size as the input.

    Returns (mask, background_rgb): mask is a mode "L" image, same size as
    `image`, where 255 = detected background and 0 = kept as foreground.
    """
    import numpy as np
    from scipy import ndimage

    rgb_array = np.asarray(image.convert("RGB"), dtype=np.int16)
    background_rgb = _sample_background_color(rgb_array, edge_sample_width)

    diff = np.abs(rgb_array - np.array(background_rgb, dtype=np.int16))
    close_to_background = diff.max(axis=-1) <= tolerance  # (H, W) bool

    # 4-connectivity (a plain cross, no diagonals) so two background pockets
    # that only touch corner-to-corner across a foreground pixel are never
    # bridged into one region -- the more conservative choice.
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labeled, _ = ndimage.label(close_to_background, structure=structure)

    border_labels = set(labeled[0, :].tolist()) | set(labeled[-1, :].tolist())
    border_labels |= set(labeled[:, 0].tolist()) | set(labeled[:, -1].tolist())
    border_labels.discard(0)  # label 0 = "not part of close_to_background" at all

    background_region = np.isin(labeled, list(border_labels)) if border_labels else np.zeros_like(labeled, dtype=bool)
    mask = Image.fromarray((background_region * 255).astype(np.uint8), mode="L")
    return mask, background_rgb


def apply_background_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Return an RGBA copy of `image` where pixels marked background in
    `mask` (255 = background, per `compute_background_mask`) are painted
    pure white and made transparent.

    Every other (foreground) pixel's RGB bytes are left completely
    untouched -- only alpha is forced to fully opaque for them -- so
    foreground color is never altered, only the background is replaced.
    """
    rgba = image.convert("RGBA")
    r, g, b, _existing_alpha = rgba.split()
    mask_l = mask.convert("L")

    # 255 = background in `mask` -> invert so foreground=255 (opaque),
    # background=0 (transparent).
    new_alpha = ImageChops.invert(mask_l)

    rgb = Image.merge("RGB", (r, g, b))
    white = Image.new("RGB", rgb.size, (255, 255, 255))
    # `mask_l` is strictly 0/255 (no partial values), so this paste is an
    # exact replace-or-keep with no blending: foreground pixels come out of
    # this call byte-for-byte identical to the input.
    rgb.paste(white, (0, 0), mask_l)

    result = rgb.convert("RGBA")
    result.putalpha(new_alpha)
    return result


def build_edge_background_remove_fn(
    tolerance: float = DEFAULT_BACKGROUND_TOLERANCE,
    edge_sample_width: int = DEFAULT_EDGE_SAMPLE_WIDTH,
) -> RemoveBackgroundFn:
    """Non-AI counterpart to build_remove_background_fn: same
    `(image) -> image` contract, so it plugs into the exact same pipeline,
    but never runs a segmentation model -- see `compute_background_mask`.
    """

    def _remove_background(image: Image.Image) -> Image.Image:
        mask, _background_rgb = compute_background_mask(image, tolerance, edge_sample_width)
        return apply_background_mask(image, mask)

    return _remove_background


def build_edge_background_mask_fn(
    tolerance: float = DEFAULT_BACKGROUND_TOLERANCE,
    edge_sample_width: int = DEFAULT_EDGE_SAMPLE_WIDTH,
) -> Callable[[Image.Image], Image.Image]:
    """For --save-mask: a pure function returning just the detected mask
    (255 = background) for debugging/visual inspection. Stateless like
    `build_edge_background_remove_fn`'s closure, so it is safe to share
    across worker threads.
    """

    def _mask_only(image: Image.Image) -> Image.Image:
        mask, _background_rgb = compute_background_mask(image, tolerance, edge_sample_width)
        return mask

    return _mask_only


def apply_exif_orientation(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image) or image


def has_visible_subject(image: Image.Image) -> bool:
    """True if the image has any non-fully-transparent pixel."""
    if image.mode != "RGBA":
        return True
    return image.getchannel("A").getbbox() is not None


def trim_transparent_margins(image: Image.Image) -> Image.Image:
    if image.mode != "RGBA":
        return image
    bbox = image.getchannel("A").getbbox()
    if bbox is None:
        return image
    return image.crop(bbox)


def resize_preserving_aspect(image: Image.Image, max_size: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0 or max_size <= 0:
        return image
    scale = min(max_size / width, max_size / height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.LANCZOS)


def compose_on_white_canvas(image: Image.Image, canvas_size: int, padding_ratio: float) -> Image.Image:
    """Resize `image` (preserving aspect ratio) to fit within the canvas
    minus padding, then center it on an opaque white square canvas.

    `padding_ratio` may be 0 -- the product then fills the canvas edge to
    edge (still fully within bounds; aspect ratio is always preserved, so
    the product is never cropped, only ever letterboxed by the unused
    dimension).

    The returned canvas is always mode "RGB" -- fully opaque with a pure
    white (255, 255, 255) background -- regardless of output format, so it
    is already valid input for JPEG, PNG, or WebP encoding.
    """
    canvas = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    usable = max(1, int(round(canvas_size * (1 - padding_ratio * 2))))
    resized = resize_preserving_aspect(image, usable)

    x = (canvas_size - resized.width) // 2
    y = (canvas_size - resized.height) // 2

    if resized.mode == "RGBA":
        canvas.paste(resized, (x, y), resized.getchannel("A"))
    else:
        canvas.paste(resized, (x, y))
    return canvas


def build_output_image(
    image: Image.Image,
    remove_background_fn: RemoveBackgroundFn,
    canvas_size: int,
    padding_ratio: float,
    mask_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
) -> "tuple[Image.Image, bool, Optional[Image.Image]]":
    """Run the full pipeline on an already-open source image.

    Returns (canvas, subject_found, mask_image). subject_found is False when
    the background-removal step left no visible pixels, signalling the
    caller to flag the result for manual review rather than silently
    emitting a blank white square. mask_image is the raw detected-background
    mask (255 = background) from `mask_fn` if one was supplied (--save-mask
    debugging), else None; it is computed independently of
    `remove_background_fn` so it never shares mutable state across threads.
    """
    oriented = apply_exif_orientation(image)
    if oriented.mode != "RGBA":
        oriented = oriented.convert("RGBA")

    no_bg = remove_background_fn(oriented)
    if no_bg.mode != "RGBA":
        no_bg = no_bg.convert("RGBA")

    mask_image = mask_fn(oriented) if mask_fn is not None else None

    subject_found = has_visible_subject(no_bg)
    trimmed = trim_transparent_margins(no_bg)
    canvas = compose_on_white_canvas(trimmed, canvas_size, padding_ratio)
    return canvas, subject_found, mask_image


def discover_images(input_dir: Path) -> List[Path]:
    """Recursively find every supported image under `input_dir`, at any
    nesting depth, matching extensions case-insensitively (`.JPG`, `.WebP`,
    etc. all count). Non-image files (`.txt`, `.csv`, `.json`, `.db`,
    `.pdf`, ...) are ignored regardless of location.

    Returned in sorted order by full path, which -- since every path shares
    the `input_dir` prefix -- is equivalent to sorting by path relative to
    `input_dir`. This is the ordering `--limit` is applied against in
    `run_process`, so which N files "the first N" means is deterministic
    and stable across runs for an unchanged directory tree.
    """
    return sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _validate_input_output_dirs(input_dir: Path, output_dir: Path) -> None:
    """Refuse configurations where writing to `output_dir` could feed back
    into `input_dir`'s own recursive discovery: the exact same directory,
    or an `--output` nested inside `--input`.

    Rejecting outright (rather than trying to filter the output tree out of
    discovery) is the safest and simplest option: `discover_images` stays a
    single, simple `rglob` with no special-casing, and there is no way for
    a file written by this run -- or left over from a previous run under
    `--output` -- to ever be silently picked back up as a new source image.
    """
    resolved_input = input_dir.resolve()
    resolved_output = output_dir.resolve()

    if resolved_output == resolved_input:
        raise ValueError(
            f"--input and --output resolve to the same directory ({resolved_input}); "
            "refusing to process images in place."
        )
    if resolved_input in resolved_output.parents:
        raise ValueError(
            f"--output ({resolved_output}) is inside --input ({resolved_input}). "
            "Recursive discovery would then also scan whatever --output contains "
            "(including files this run just created, or leftovers from a previous "
            "run) as if they were new source images. Choose an --output directory "
            "outside of --input."
        )


def resolve_output_suffix(source_path: Path, output_format: str) -> str:
    """Return the destination file suffix (with leading dot) for
    `source_path` given the --format selection.

    'preserve' (the default) keeps the source's own extension exactly as-is
    (basename and casing untouched). 'webp'/'jpg'/'png' force every output
    to that single format/extension instead.
    """
    if output_format not in VALID_OUTPUT_FORMATS:
        raise ValueError(
            f"Invalid output format {output_format!r}; expected one of {sorted(VALID_OUTPUT_FORMATS)}"
        )
    if output_format == "preserve":
        return source_path.suffix
    return FORCED_EXTENSION_BY_FORMAT_FLAG[output_format]


def pil_format_for_suffix(suffix: str) -> str:
    """Map a destination file suffix to the Pillow format name to encode
    it with. Falls back to WEBP for an unrecognized suffix, which should
    not happen in practice since `resolve_output_suffix` only ever produces
    suffixes present in FORMAT_BY_EXTENSION."""
    return FORMAT_BY_EXTENSION.get(suffix.lower(), "WEBP")


def _save_kwargs(pil_format: str, quality: int) -> dict:
    if pil_format == "JPEG":
        return {"quality": quality}
    if pil_format == "WEBP":
        return {"quality": quality}
    if pil_format == "PNG":
        # PNG is always lossless in Pillow; `optimize` just spends more time
        # for a smaller file -- this is the "default Pillow optimization".
        return {"optimize": True}
    return {}


def _process_one(
    source_path: Path,
    dest_path: Path,
    remove_background_fn: RemoveBackgroundFn,
    canvas_size: int,
    padding_ratio: float,
    quality: int,
    force: bool,
    mask_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
    save_mask: bool = False,
) -> ProcessResult:
    started = time.monotonic()

    if dest_path.exists() and not force:
        # Safety net for a destination created between planning (in
        # run_process) and this call -- run_process already filters these
        # out up front so remove_background_fn is never invoked for them.
        return ProcessResult(str(source_path), str(dest_path), "skipped", 0, 0, "output_exists")

    try:
        with Image.open(source_path) as img:
            img.load()
            canvas, subject_found, mask_image = build_output_image(
                img, remove_background_fn, canvas_size, padding_ratio, mask_fn=mask_fn
            )

        if save_mask and mask_image is not None:
            mask_path = dest_path.with_name(dest_path.stem + ".mask.png")
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask_image.save(mask_path, format="PNG")

        pil_format = pil_format_for_suffix(dest_path.suffix)
        if pil_format == "JPEG" and canvas.mode != "RGB":
            # compose_on_white_canvas already returns RGB, but JPEG cannot
            # encode anything else -- guard against that explicitly.
            canvas = canvas.convert("RGB")

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_name(dest_path.name + ".part")
        canvas.save(tmp_path, format=pil_format, **_save_kwargs(pil_format, quality))
        tmp_path.replace(dest_path)

        size = dest_path.stat().st_size
        duration_ms = int((time.monotonic() - started) * 1000)
        status = "processed" if subject_found else "needs_review"
        error = "" if subject_found else "no subject detected after background removal"
        return ProcessResult(str(source_path), str(dest_path), status, size, duration_ms, error)
    except Exception as exc:  # noqa: BLE001 - isolate per-file failures
        duration_ms = int((time.monotonic() - started) * 1000)
        return ProcessResult(str(source_path), str(dest_path), "failed", 0, duration_ms, str(exc))


def run_process(
    input_dir: Path,
    output_dir: Path,
    canvas_size: int = 1200,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
    quality: int = 90,
    workers: int = 2,
    limit: Optional[int] = None,
    force: bool = False,
    output_format: str = "preserve",
    mode: str = DEFAULT_MODE,
    model: str = DEFAULT_MODEL,
    post_process_mask: bool = False,
    alpha_matting: bool = False,
    background_tolerance: float = DEFAULT_BACKGROUND_TOLERANCE,
    edge_sample_width: int = DEFAULT_EDGE_SAMPLE_WIDTH,
    save_mask: bool = False,
    remove_background_fn: Optional[RemoveBackgroundFn] = None,
    progress_cb: Optional[Callable[[ProcessResult], None]] = None,
) -> List[ProcessResult]:
    if output_format not in VALID_OUTPUT_FORMATS:
        raise ValueError(
            f"Invalid output format {output_format!r}; expected one of {sorted(VALID_OUTPUT_FORMATS)}"
        )
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode {mode!r}; expected one of {sorted(VALID_MODES)}")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    _validate_input_output_dirs(input_dir, output_dir)

    files = discover_images(input_dir)
    if limit is not None:
        files = files[:limit]

    # Resolve each file's destination path up front, using the already-
    # selected --format logic, and split into images that already have an
    # output (skipped, unless --force) vs. images that actually need to run
    # through the segmentation model. This check happens before touching
    # rembg at all, so a file with an existing destination is never opened,
    # never segmented, and never overwritten.
    results: List[ProcessResult] = []
    pending: List["tuple[Path, Path]"] = []
    for source_path in files:
        relative = source_path.relative_to(input_dir)
        dest_suffix = resolve_output_suffix(source_path, output_format)
        dest_path = output_dir / relative.with_suffix(dest_suffix)
        if dest_path.exists() and not force:
            result = ProcessResult(str(source_path), str(dest_path), "skipped", 0, 0, "output_exists")
            results.append(result)
            if progress_cb:
                progress_cb(result)
        else:
            pending.append((source_path, dest_path))

    if not pending:
        # Nothing left to do -- never create/load a segmentation session
        # (rembg.new_session() is expensive) when there is no work for it.
        return results

    # Built once, here, outside the per-image loop -- shared by every worker
    # thread below so the (expensive) rembg session is created exactly once
    # per run_process() call, never once per image, and only when at least
    # one image actually needs processing. mode="edge-background" never
    # touches rembg at all.
    if remove_background_fn is None:
        if mode == "edge-background":
            remove_background_fn = build_edge_background_remove_fn(background_tolerance, edge_sample_width)
        else:
            remove_background_fn = build_remove_background_fn(model, post_process_mask, alpha_matting)

    mask_fn = None
    if save_mask and mode == "edge-background":
        mask_fn = build_edge_background_mask_fn(background_tolerance, edge_sample_width)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _process_one,
                source_path,
                dest_path,
                remove_background_fn,
                canvas_size,
                padding_ratio,
                quality,
                force,
                mask_fn,
                save_mask,
            ): source_path
            for source_path, dest_path in pending
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if progress_cb:
                progress_cb(result)

    return results


def run_compare_models(
    input_dir: Path,
    output_dir: Path,
    models: Optional[List[str]] = None,
    limit: int = 5,
    canvas_size: int = 1200,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
    quality: int = 90,
    workers: int = 2,
    force: bool = False,
    output_format: str = "preserve",
    post_process_mask: bool = False,
    alpha_matting: bool = False,
    remove_background_fn_factory: Optional[Callable[[str], RemoveBackgroundFn]] = None,
    progress_cb: Optional[Callable[[str, ProcessResult], None]] = None,
) -> "Dict[str, List[ProcessResult]]":
    """Process the same first `limit` images with each model in `models`,
    writing each model's output to its own `output_dir/<model>/` directory
    so results never overwrite each other. For visual comparison only.

    `remove_background_fn_factory(model_name) -> RemoveBackgroundFn` may be
    injected for testing; it defaults to `build_remove_background_fn`, which
    creates one fresh session per model (still only once per model, not
    once per image).
    """
    models = list(models) if models else list(SUPPORTED_MODELS)
    factory = remove_background_fn_factory or (
        lambda model_name: build_remove_background_fn(model_name, post_process_mask, alpha_matting)
    )

    output_dir = Path(output_dir)
    results_by_model: "Dict[str, List[ProcessResult]]" = {}
    for model_name in models:
        model_output_dir = output_dir / model_name

        def _on_progress(result: ProcessResult, _model_name: str = model_name) -> None:
            if progress_cb:
                progress_cb(_model_name, result)

        results_by_model[model_name] = run_process(
            input_dir=input_dir,
            output_dir=model_output_dir,
            canvas_size=canvas_size,
            padding_ratio=padding_ratio,
            quality=quality,
            workers=workers,
            limit=limit,
            force=force,
            output_format=output_format,
            remove_background_fn=factory(model_name),
            progress_cb=_on_progress,
        )
    return results_by_model
