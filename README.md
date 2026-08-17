# Listing-to-Reel

An auditable generative-media pipeline for real-estate marketing.

## Phase 0 status

Phase 0 establishes reproducibility only. It does not run image or video models.

- `local_mps` is the Apple-silicon development and preview profile.
- `remote_cuda` is the authoritative inference and benchmark profile.
- Runtime validation never silently falls back from MPS or CUDA to CPU.
- Source-image rights and hashes are recorded in `data/manifests/`.

## Deterministic baseline reel

Phase 1 uses Pillow and FFmpeg only—no model inference or AI editing. It creates a fixed 9:16 crop, a subtle deterministic zoom, crossfades, an H.264 MP4, and a JSON run manifest.

```bash
brew install ffmpeg # macOS, once
uv sync --extra dev --python 3.12
uv run python -m listing_to_reel reel assemble \
  --image data/source/unsplash_exteriors/001.jpg \
  --image data/source/unsplash_exteriors/002.jpg \
  --image data/source/unsplash_exteriors/003.jpg \
  --image data/source/unsplash_exteriors/004.jpg \
  --image data/source/unsplash_exteriors/005.jpg
```

The command prints the reel path and its companion JSON manifest. Output is placed under `runs/` and is intentionally ignored by Git.

## Input quality gate

Phase 2 captures EXIF metadata and evaluates blur, exposure clipping, color cast, and vertical-line quality before any model runs.

```bash
uv run python -m listing_to_reel analyze input \
  --image data/source/unsplash_exteriors/001.jpg \
  --image data/source/unsplash_exteriors/002.jpg \
  --config configs/input_quality.yaml
```

The command writes an `InputQualityReport` JSON file and an optional vertical-line diagnostic overlay under `runs/input-quality/`. Vertical-line analysis is warning-only by default because it is a perspective proxy, not proof of a bad photo.

## Baseline image editing

Phase 3 uses the MIT-licensed InstructPix2Pix checkpoint as a configurable, pretrained baseline. It requires an **accepted** Phase 2 report, saves source/config/seed/model-revision provenance, and leaves accept/reject decisions to Phase 4.

The local Apple-Silicon profile uses float32 decoding for reliable MPS output. CUDA runs may use a float16 configuration override after their own benchmark and quality evaluation.

```bash
uv sync --extra mps --python 3.12
uv run python -m listing_to_reel edit image \
  --image data/source/unsplash_exteriors/001.jpg \
  --input-quality-report runs/input-quality/input-574fd8d0144c5c94/report.json \
  --instruction "Convert this daylight exterior to a natural premium golden-hour scene; preserve exact house geometry, windows, landscaping, driveway, and vertical lines." \
  --seed 42
```

Use `--runtime-config configs/remote_cuda.yaml --profile remote_cuda` for authoritative CUDA evaluation. MPS is for local preview runs only.

For a slower, detail-preserving M5 experiment, pass `--config configs/image_editing_mps_high_detail.yaml`. This uses 768px, 30-step float32 MPS inference and remains a preview configuration.

## Candidate evaluation and human review

Phase 4 rejects broken artifacts (for example black frames), measures edge preservation, blur, luminance change, and vertical-line drift, then ranks viable candidates. It does **not** automatically certify property truthfulness: viable edits are queued for a blinded human decision.

```bash
uv run python -m listing_to_reel evaluate images \
  --edit-run-manifest runs/edits/edit-5cd62bdaf68072ca/manifest.json
```

This writes an evaluation report and `review.csv` under `runs/evaluations/`. Fill in its `decision`, `reviewer`, and `notes` fields with `accepted_by_human` or `rejected_by_human`, then record the final decision:

```bash
uv run python -m listing_to_reel evaluate import-review \
  --evaluation-report runs/evaluations/quality-5cd62bdaf68072ca/report.json \
  --worksheet runs/evaluations/quality-5cd62bdaf68072ca/review.csv
```

## LTX multi-shot video foundation

Phase 5 is moving to LTX image-to-video through ComfyUI on the CUDA PC. The final reel will be made from four to six independently generated, source-anchored shots of **one property**—never an invented camera flight between unrelated views.

Before rendering, create an auditable plan from independently approved source views. Each role may appear once; a wide exterior and a closing hero are required. The plan assigns 1.5–2 seconds per shot for an 8–10 second reel and records every source image used.

```bash
uv run python -m listing_to_reel video plan-ltx-multishot \
  --property-id property-001 \
  --shot wide_exterior=runs/evaluations/front/final-decision.json \
  --shot backyard=runs/evaluations/backyard/final-decision.json \
  --shot architectural_detail=runs/evaluations/detail/final-decision.json \
  --shot closing_hero=runs/evaluations/hero/final-decision.json
```

The planner does not render a video. LTX/ComfyUI rendering will be added when we have four to six approved images of the same property to validate it against.

## Local setup

```bash
uv sync --extra dev
uv run python -m listing_to_reel config-check --config configs/local_mps.yaml
uv run python -m listing_to_reel environment
uv run pytest
uv run ruff check .
```

To probe your M5's MPS backend before model work, install PyTorch only:

```bash
uv sync --extra dev --extra mps
uv run python -m listing_to_reel environment
```

Do not install the `gpu` extra on macOS for this phase. Install it in the remote NVIDIA worker environment.
