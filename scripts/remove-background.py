#!/usr/bin/env python3
"""Remove backgrounds from local images and build clean catalog images.

Pipeline: EXIF orientation -> AI background removal -> trim transparent
margins -> resize preserving aspect ratio -> center -> white canvas -> save.
By default (--format preserve) the output keeps each source file's own
format/extension (.jpg/.jpeg -> JPEG, .png -> PNG, .webp -> WebP); pass
--format webp/jpg/png to force everything to one format. Input files are
never modified. See README.md for usage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fon_image_cleaner.background import (  # noqa: E402
    DEFAULT_BACKGROUND_TOLERANCE,
    DEFAULT_EDGE_SAMPLE_WIDTH,
    DEFAULT_MODE,
    DEFAULT_MODEL,
    DEFAULT_PADDING_RATIO,
    SUPPORTED_MODELS,
    VALID_MODES,
    run_compare_models,
    run_process,
)
from fon_image_cleaner.reporting import write_report  # noqa: E402

DEFAULT_COMPARE_LIMIT = 5


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove backgrounds and build catalog images")
    parser.add_argument("--input", required=True, help="Local input directory")
    parser.add_argument("--output", required=True, help="Local output directory")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images (for testing)")
    parser.add_argument("--workers", type=int, default=2, help="Bounded concurrency (default: 2)")
    parser.add_argument("--size", type=int, default=1200, help="Square canvas size in px (default: 1200)")
    parser.add_argument(
        "--padding",
        type=float,
        default=DEFAULT_PADDING_RATIO,
        help=f"Padding ratio per side, 0 for no padding (default: {DEFAULT_PADDING_RATIO})",
    )
    parser.add_argument(
        "--quality", type=int, default=90, help="JPEG/WebP quality 0-100 (default: 90; ignored for PNG)"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output files that already exist")
    parser.add_argument(
        "--format",
        choices=["preserve", "webp", "jpg", "png"],
        default="preserve",
        help="Output format: 'preserve' keeps each source's own format/extension (default), "
        "or force everything to 'webp'/'jpg'/'png'",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(VALID_MODES),
        default=DEFAULT_MODE,
        help=(
            f"Background removal mode (default: {DEFAULT_MODE}). 'ai' uses rembg (see --model); "
            "'edge-background' is a conservative non-AI algorithm that only removes the background "
            "region connected to the image border -- it never segments the product, so it can't "
            "mistake a gray product part (e.g. a handle) for background the way AI segmentation can."
        ),
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default=DEFAULT_MODEL,
        help=f"rembg segmentation model, only used with --mode ai (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--post-process-mask",
        action="store_true",
        help="Enable rembg's mask post-processing, only used with --mode ai (default: off)",
    )
    parser.add_argument(
        "--alpha-matting",
        action="store_true",
        help="Enable rembg's alpha matting refinement, only used with --mode ai (default: off)",
    )
    parser.add_argument(
        "--background-tolerance",
        type=float,
        default=DEFAULT_BACKGROUND_TOLERANCE,
        help=(
            "Only used with --mode edge-background: max per-channel color "
            f"distance (0-255) from the sampled border color to still count "
            f"as background (default: {DEFAULT_BACKGROUND_TOLERANCE})"
        ),
    )
    parser.add_argument(
        "--edge-sample-width",
        type=int,
        default=DEFAULT_EDGE_SAMPLE_WIDTH,
        help=(
            "Only used with --mode edge-background: width in pixels of the "
            "border band sampled (median) to estimate the background color "
            f"(default: {DEFAULT_EDGE_SAMPLE_WIDTH})"
        ),
    )
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help=(
            "Only used with --mode edge-background: also save a debug mask "
            "(<name>.mask.png, 255=background/white, 0=foreground/kept) next "
            "to each output, showing exactly which pixels were treated as background."
        ),
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help=(
            "Process the first N images (see --limit, default "
            f"{DEFAULT_COMPARE_LIMIT} in this mode) with every model in "
            f"{SUPPORTED_MODELS}, writing each into its own "
            "<output>/<model>/ subdirectory for visual comparison. --model is ignored."
        ),
    )
    return parser.parse_args(argv)


def _print_summary(results) -> dict:
    counts = {"processed": 0, "needs_review": 0, "skipped": 0, "failed": 0}
    total_bytes = 0
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
        if result.status in ("processed", "needs_review"):
            total_bytes += result.size
    print(f"Images found:  {len(results)}")
    print(f"Processed:     {counts.get('processed', 0)}")
    print(f"Needs review:  {counts.get('needs_review', 0)}")
    print(f"Skipped:       {counts.get('skipped', 0)}")
    print(f"Failed:        {counts.get('failed', 0)}")
    print(f"Total bytes:   {total_bytes}")
    return counts


def _run_compare(args) -> int:
    limit = args.limit if args.limit is not None else DEFAULT_COMPARE_LIMIT
    print(f"Comparing models {SUPPORTED_MODELS} on the first {limit} image(s) ...")
    print(f"Output: {args.output}/<model>/ (one subdirectory per model, never overwriting each other)")
    print()

    def on_progress(model_name, result):
        print(f"[{model_name}] [{result.status}] {result.source} -> {result.destination}")

    try:
        results_by_model = run_compare_models(
            input_dir=Path(args.input),
            output_dir=Path(args.output),
            limit=limit,
            canvas_size=args.size,
            padding_ratio=args.padding,
            quality=args.quality,
            workers=args.workers,
            force=args.force,
            output_format=args.format,
            post_process_mask=args.post_process_mask,
            alpha_matting=args.alpha_matting,
            progress_cb=on_progress,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    any_failed = False
    for model_name, results in results_by_model.items():
        print()
        print(f"== {model_name} ==")
        counts = _print_summary(results)
        report_path = write_report(f"background-compare-{model_name}", results)
        print(f"Report:        {report_path}")
        any_failed = any_failed or bool(counts.get("failed", 0))

    return 1 if any_failed else 0


def main(argv=None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input)

    if not input_dir.is_dir():
        print(f"error: input directory not found: {input_dir}", file=sys.stderr)
        return 2

    if args.compare_models:
        return _run_compare(args)

    output_dir = Path(args.output)

    def on_progress(result):
        print(f"[{result.status}] {result.source} -> {result.destination}")

    try:
        results = run_process(
            input_dir=input_dir,
            output_dir=output_dir,
            canvas_size=args.size,
            padding_ratio=args.padding,
            quality=args.quality,
            workers=args.workers,
            limit=args.limit,
            force=args.force,
            output_format=args.format,
            mode=args.mode,
            model=args.model,
            post_process_mask=args.post_process_mask,
            alpha_matting=args.alpha_matting,
            background_tolerance=args.background_tolerance,
            edge_sample_width=args.edge_sample_width,
            save_mask=args.save_mask,
            progress_cb=on_progress,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report_path = write_report("background", results)

    print()
    counts = _print_summary(results)
    print(f"Report:        {report_path}")

    return 1 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
