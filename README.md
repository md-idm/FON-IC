# fon-image-cleaner

Standalone image tooling for the FishOnNow catalog: download originals from
Cloudflare R2, remove backgrounds with AI to produce clean catalog images,
and upload the results back to R2.

This project is **fully independent of the main FishOnNow application** —
it does not import from, or get imported by, that codebase. It's just a
handful of CLI scripts and a small shared library.

Requires Python 3.12+.

## Project structure

```
scripts/
  r2-download.py        # STEP 1: R2 -> local
  remove-background.py  # STEP 2/3: local -> local (background removal)
  r2-upload.py           # STEP 4/5: local -> R2
  r2-list.py             # read-only diagnostic: inspect what's in the bucket
src/fon_image_cleaner/
  r2.py           # shared R2 client (list / exists / download / upload)
  downloader.py   # download orchestration used by r2-download.py
  uploader.py     # upload orchestration used by r2-upload.py
  background.py   # image pipeline used by remove-background.py
  listing.py      # read-only listing/diagnostics engine used by r2-list.py
  reporting.py    # shared CSV report writer
tests/            # unit tests (R2 calls are mocked, never hit real R2)
reports/          # CSV run reports land here
.env.example
pyproject.toml
```

All Cloudflare R2 logic lives in `src/fon_image_cleaner/r2.py` and is shared
by every script that talks to R2 — it is not duplicated.

## Install

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Environment variables

Copy `.env.example` to `.env` and fill in your Cloudflare R2 credentials:

```
R2_ACCOUNT_ID=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET_NAME=
R2_ENDPOINT=
```

- `R2_ENDPOINT` should be the standard Cloudflare R2 S3-compatible endpoint,
  e.g. `https://<account_id>.r2.cloudflarestorage.com`. If left blank it is
  derived automatically from `R2_ACCOUNT_ID`.
- `.env` is loaded automatically by the scripts (via `src/fon_image_cleaner/r2.py`)
  if present in the current working directory. It is git-ignored — **never commit it**.
- Only the two R2 scripts read these variables. `remove-background.py` is
  purely local and needs no R2 configuration.

## Complete workflow

### STEP 1 — Download originals from R2

```
python scripts/r2-download.py ^
  --remote-prefix "upload/" ^
  --local-dir "D:\FON-images\original"
```

Optionally narrow by last-modified date (UTC), e.g. only images uploaded in
the first ten days of August 2026:

```
python scripts/r2-download.py ^
  --remote-prefix "upload/" ^
  --local-dir "D:\FON-images\original" ^
  --since "2026-08-01" ^
  --until "2026-08-10"
```

`--since` is inclusive of that whole UTC day; `--until` is inclusive of that
whole UTC day too (internally treated as `< the following UTC midnight`).
Both are optional and independent.

### STEP 2 — Test background removal on 10 images

```
python scripts/remove-background.py ^
  --input "D:\FON-images\original" ^
  --output "D:\FON-images\processed" ^
  --limit 10
```

Inspect the output in `D:\FON-images\processed` and the CSV report under
`reports/` before processing everything.

### STEP 3 — Process all images

```
python scripts/remove-background.py ^
  --input "D:\FON-images\original" ^
  --output "D:\FON-images\processed"
```

### STEP 4 — Dry-run the upload

```
python scripts/r2-upload.py ^
  --local-dir "D:\FON-images\processed" ^
  --remote-prefix "processed/" ^
  --dry-run
```

### STEP 5 — Upload processed images

```
python scripts/r2-upload.py ^
  --local-dir "D:\FON-images\processed" ^
  --remote-prefix "processed/"
```

## Download CLI — `scripts/r2-download.py`

Recursively copies every object under an R2 prefix into a local directory,
preserving sub-directory structure.

| Flag | Default | Description |
|---|---|---|
| `--remote-prefix` | `""` (entire bucket) | R2 key prefix to copy, e.g. `"upload/"` |
| `--local-dir` | required | Local destination directory |
| `--workers` | `8` | Bounded concurrency |
| `--force` | off | Overwrite local files that already exist |
| `--dry-run` | off | List what would be downloaded without downloading |
| `--since` | none | Only objects with `LastModified >= YYYY-MM-DD` (UTC) |
| `--until` | none | Only objects with `LastModified` within `YYYY-MM-DD` (UTC), inclusive |

Behavior:
- Recursive, binary-safe, preserves directory structure.
- Omitting `--remote-prefix` (or passing `""`) targets the **entire bucket**.
- `--since`/`--until` filter on the R2 object's `LastModified`, compared as
  timezone-aware UTC; `--since` must not be after `--until`.
- Prints per-file progress plus a final summary (file count, total bytes).
- `--dry-run` respects date filtering — it lists exactly what a real run
  would download, without downloading anything.
- Per-file failures are recorded and do not stop the batch.
- Transient network failures are retried automatically.
- Skips files that already exist locally unless `--force` is given.
- **Never deletes anything from R2.**

If a download unexpectedly returns 0 objects, run `scripts/r2-list.py` first
(below) to check the bucket/prefix/date range actually contain what you expect.

## Diagnostic CLI — `scripts/r2-list.py`

Read-only inspection of what's actually in the configured R2 bucket. Uses the
same shared client/config as the other scripts. Useful for debugging an
unexpectedly empty `r2-download.py` run.

```
python scripts/r2-list.py
python scripts/r2-list.py --prefix "upload/"
python scripts/r2-list.py --limit 100
```

| Flag | Default | Description |
|---|---|---|
| `--prefix` | `""` (entire bucket) | R2 key prefix to inspect, e.g. `"upload/"` |
| `--limit` | `20` | Number of sample objects to print |

Prints: bucket name, endpoint **host only**, total object count, top-level
prefixes, and the first `--limit` objects (key, `LastModified`, size).

- **Strictly read-only** — only lists objects; never downloads, uploads, or
  deletes anything.
- Correctly paginates through the entire prefix before reporting a total.
- **Never prints `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, authorization
  headers, or any other secret** — only the bucket name and the endpoint's
  host (never the full endpoint URL, never the config object itself).

## Upload CLI — `scripts/r2-upload.py`

Recursively uploads a local directory tree into an R2 prefix, preserving
relative directory structure.

| Flag | Default | Description |
|---|---|---|
| `--local-dir` | required | Local source directory |
| `--remote-prefix` | required | R2 key prefix to upload into, e.g. `"processed/"` |
| `--workers` | `8` | Bounded concurrency |
| `--force` | off | Overwrite remote objects that already exist |
| `--dry-run` | off | List what would be uploaded without uploading |

Behavior:
- Recursive, preserves relative paths as the remote key suffix.
- Sets `Content-Type` for `.jpg`/`.jpeg` → `image/jpeg`, `.png` → `image/png`,
  `.webp` → `image/webp`.
- Prints per-file progress plus a final summary (files, total bytes).
- Per-file failures are recorded and do not stop the batch.
- Transient network failures are retried automatically.
- Skips objects that already exist remotely unless `--force` is given.
- **Never deletes remote objects and never uses sync/delete semantics** —
  this is a strictly additive upload.

## Background CLI — `scripts/remove-background.py`

Turns raw product photos into clean, centered catalog images.

Pipeline: `source image → EXIF orientation → background removal (AI or
edge-background) → trim transparent margins → resize preserving aspect
ratio → center → white (#FFFFFF) canvas → save`.

| Flag | Default | Description |
|---|---|---|
| `--input` | required | Local input directory |
| `--output` | required | Local output directory |
| `--limit` | none | Only process the first N images (for test runs) |
| `--workers` | `2` | Bounded concurrency |
| `--size` | `1200` | Square canvas size in px (1200×1200) |
| `--padding` | `0.02` | Padding ratio per side (2%) |
| `--quality` | `90` | JPEG/WebP quality (0–100); ignored for PNG |
| `--force` | off | Overwrite output files that already exist |
| `--format` | `preserve` | `preserve`, `webp`, `jpg`, or `png` — see below |
| `--mode` | `ai` | `ai` (rembg) or `edge-background` (non-AI) — see below |
| `--model` | `birefnet-general` | `u2net`, `isnet-general-use`, or `birefnet-general`; only used with `--mode ai` |
| `--post-process-mask` | off | Enable rembg's mask post-processing; only used with `--mode ai` |
| `--alpha-matting` | off | Enable rembg's alpha matting refinement; only used with `--mode ai` |
| `--background-tolerance` | `30` | Max per-channel color distance (0–255) still counted as background; only used with `--mode edge-background` |
| `--edge-sample-width` | `12` | Border band width (px) sampled to estimate the background color; only used with `--mode edge-background` |
| `--save-mask` | off | Save a debug `<name>.mask.png` showing exactly which pixels were treated as background; only used with `--mode edge-background` |
| `--compare-models` | off | Run all 3 AI models on the first N images into separate dirs — see below |

Behavior:
- Supported input formats: `.jpg`, `.jpeg`, `.png`, `.webp`.
- Preserves relative sub-directories, e.g. `input/reels/a.jpg` →
  `output/reels/a.jpg` (with the default `--format preserve`).
- **Output format defaults to `preserve`**: each file keeps its own source
  format and extension —`.jpg`/`.jpeg` → JPEG, `.png` → PNG, `.webp` → WebP.
  Pass `--format webp` (or `jpg`/`png`) to force every output to a single
  format instead. The on-disk extension always matches the actual encoded
  format.
- The background is always composited to pure opaque white (`#FFFFFF`)
  regardless of output format; JPEG output is always converted to RGB
  before saving (JPEG has no alpha channel).
- `--padding` defaults to `0.02` (2% per side) — down from an earlier `0.08`
  default that left too much artificial white border. `--padding 0` fits the
  product edge-to-edge; the product is never cropped either way, since
  resizing always preserves aspect ratio.
- `--model` selects the rembg segmentation model. `birefnet-general` (the
  default) holds onto semi-dark/gray product parts (e.g. handles) far better
  than `u2net`, which tends to erase them as background. The model session
  is created **once** for the whole run and reused for every image — never
  recreated per image.
- **Input files are never modified** — they are only opened for reading.
- **Resume-safe**: before processing each image, the expected destination
  path is computed (honoring `--format`), and if it already exists the image
  is skipped (`status=skipped`, `error=output_exists` in the CSV report)
  **without ever opening the source file or invoking the segmentation
  model** — a restarted run never redoes already-completed work. `--force`
  overrides this and regenerates/overwrites existing output. If every image
  in a run already has an output, the rembg session is never created at all.
- Per-image failures are isolated and recorded; the batch continues.
- If background removal leaves no visible subject, the image is still saved
  but flagged with status `needs_review` so it can be checked by hand.

### Non-AI background removal — `--mode edge-background`

AI segmentation (including `birefnet-general`) can still remove real parts
of a product when their color is close to a uniform gray studio background —
a gray handle, for example. `--mode edge-background` is a conservative
alternative for exactly that situation: it never runs a segmentation model
and never classifies an *interior* region as background purely by color.

Algorithm:
1. Sample the background color from border/corner regions (a robust median
   over an `--edge-sample-width`-pixel band along all four edges, not a
   single assumed RGB value).
2. Flood-fill **outward from the image border only**, through pixels within
   `--background-tolerance` of that sampled color.
3. Only pixels that are BOTH close enough in color AND reachable from the
   border through such pixels are marked background; everything else keeps
   its original RGB, untouched.

This means a gray handle (or any gray product part) that's walled off from
the real background by other product pixels is never touched, no matter how
"background-like" its color looks in isolation. The image is never cropped
or resized during detection, and no AI mask is created.

```
python scripts/remove-background.py \
  --input "D:\FON-images\original" \
  --output "D:\FON-images\processed" \
  --mode edge-background \
  --limit 10
```

Tune it with `--background-tolerance` (higher = more colors count as
background; lower = stricter, more conservative) and `--edge-sample-width`
(wider = more robust to noisy corners, narrower = safer if the product sits
close to the frame edge). Add `--save-mask` to write a `<name>.mask.png`
next to each output — white (255) is exactly what was treated as background,
black (0) is exactly what was kept — useful for tuning tolerance before a
full run. `--model`, `--post-process-mask`, and `--alpha-matting` are
ignored in this mode (they're AI-only).

### Comparing models — `--compare-models`

Runs every supported model against the same first N images (`--limit`,
default 5 in this mode) so you can visually compare segmentation quality
before committing to one for a full run. Each model writes into its own
subdirectory — outputs never overwrite each other:

```
python scripts/remove-background.py \
  --input "D:\FON-images\original" \
  --output "D:\FON-images\compare" \
  --compare-models \
  --limit 5
```

produces:

```
D:\FON-images\compare\
  u2net\
  isnet-general-use\
  birefnet-general\
```

`--model` is ignored in this mode (all three run); `--format`, `--padding`,
`--quality`, `--post-process-mask`, and `--alpha-matting` still apply, the
same for every model. A separate CSV report is written per model
(`reports/run-background-compare-<model>-*.csv`).

## Reporting

Every run writes a CSV report to `reports/`:

```
reports/run-download-YYYYMMDD-HHMMSS.csv
reports/run-upload-YYYYMMDD-HHMMSS.csv
reports/run-background-YYYYMMDD-HHMMSS.csv
```

Columns: `source, destination, last_modified, size, status, duration_ms, error`
(`last_modified` is only populated for download reports).

Statuses: `downloaded`, `uploaded`, `processed`, `skipped`, `failed`, `needs_review`.

## Tests

R2 tests mock all S3/R2 calls (via a fake in-memory client and `unittest.mock`)
and never contact real Cloudflare infrastructure. `--mode ai` tests inject a
fake segmentation function instead of running the real `rembg` model, so
they run fast and deterministically. `--mode edge-background`
(`tests/test_edge_background.py`) needs no mocking at all — it's pure
Pillow/numpy/scipy, tested against the real algorithm.

```
pytest
```

## Safety

- Never deletes R2 objects.
- Never deletes local originals.
- Never uses sync/delete semantics.
- Never overwrites without `--force`.
- Never logs Cloudflare secrets.
- `.env` is git-ignored and must never be committed.
- These scripts do not run automatically against anything — they only act
  when you invoke them explicitly with the directories/prefixes you choose.
