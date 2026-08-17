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

## LTX / ComfyUI multi-shot video

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

The planner creates an auditable shot plan. Rendering uses **ComfyUI/LTX only** and generates
native 16:9 landscape candidates; it never crops the foreground property to fill portrait.
Set `comfyui_root` in `configs/ltx_comfyui.yaml` to the local ComfyUI checkout, start ComfyUI,
then render the source-anchored candidates:

```powershell
uv run --no-sync python -m listing_to_reel video render-ltx `
  --property-id synthetic-simple-suburban-home `
  --source 03-front-day-left=data/source/synthetic_simple_suburban_home/03-front-day-left.png,slow_lateral_gimbal_glide `
  --source 05-front-day-wide=data/source/synthetic_simple_suburban_home/05-front-day-wide.png,gentle_dolly_in `
  --source 02-covered-patio=data/source/synthetic_simple_suburban_home/02-covered-patio.png,slow_lateral_gimbal_glide `
  --source 04-backyard-patio=data/source/synthetic_simple_suburban_home/04-backyard-patio.png,gentle_dolly_in `
  --source 01-front-twilight=data/source/synthetic_simple_suburban_home/01-front-twilight.png,slow_lateral_gimbal_glide `
  --bridge-candidate front-left-to-front-wide=3.5 `
  --bridge-candidate patio-to-backyard=3.5 `
  --bridge-candidate front-day-to-twilight=3.0

uv run --no-sync python -m listing_to_reel video qa-ltx `
  --render-manifest runs/ltx-videos/<run-id>/manifest.json
```

### Control reel pacing

Choose the total delivery length and desired invented-bridge duration. The timed planner gives
each selected compatible LTX bridge that duration, distributes all remaining time evenly across
its follow-on source views, records the resulting playback speeds, and uses cinematic dissolves
between unrelated areas. For example, this requests a 20-second reel with three-second bridges:

```powershell
uv run --no-sync python -m listing_to_reel video plan-ltx-reel `
  --render-manifest runs/ltx-videos/<run-id>/manifest.json `
  --total-seconds 20 `
  --bridge-seconds 3 `
  --scene-fade-seconds 0.45 `
  --bridge front-left-to-front-wide `
  --bridge patio-to-backyard `
  --bridge front-day-to-twilight
```

Use `--bridge-duration candidate=seconds` when one compatible transition needs a different
pace. This makes a 15-second reel with a five-second front transition and three-second remaining
transitions:

```powershell
uv run --no-sync python -m listing_to_reel video plan-ltx-reel `
  --render-manifest runs/ltx-videos/<run-id>/manifest.json `
  --total-seconds 15 `
  --bridge-seconds 3 `
  --bridge front-left-to-front-wide `
  --bridge patio-to-backyard `
  --bridge front-day-to-twilight `
  --bridge-duration front-left-to-front-wide=5
```

Bridge duration must be 2–6 seconds. Select only pairs with compatible visual overlap; a
front-to-backyard or otherwise unrelated change is planned as a 0.2–1.0 second cinematic
cross-dissolve, not an invented camera move. The fade duration is included in the requested final
reel length.

Render that saved plan—not merely source clips—into the final bridge-and-clip timeline:

```powershell
uv run --no-sync python -m listing_to_reel video assemble-ltx-timed-reel `
  --plan runs/ltx-timed-reels/<plan-run-id>/plan.json
```

For an arbitrary compatible pair from the supplied source views, define it explicitly at render
time. It remains queued for human review and is never auto-accepted:

```powershell
--bridge-pair patio-to-yard=02-covered-patio,04-backyard-patio,spatial_overlap `
--bridge-candidate patio-to-yard=3
```

The render manifest records the exact ComfyUI node graph, pinned model/encoder/VAE names,
prompt, source hashes, coverage, output hashes, and generated clips. QA writes a mandatory
human-review worksheet for every clip and for generated, user-selected spatial/lighting bridges.
Each `--bridge-candidate` accepts a per-transition 2–4 second invented duration. Nothing is
auto-accepted: reject a bridge if geometry changes or temporal artifacts occur; use clean cuts for
unrelated views. The final portrait editor retains the complete landscape foreground over a blurred
background fill. After recording human approval, assemble only the names that were approved, in
edit order:

### Choosing bridge candidates

Group source images by property and label their visible area and viewpoint, for example
`front-left`, `front-wide`, `backyard`, or `patio`. Select a spatial bridge only when its two
images show the same or an adjacent physical area with enough visual overlap for a plausible camera
move. LTX then invents a **3-second**, smooth, continuous camera transition between the two
endpoint images. It is not a crossfade, slideshow, or static morph. Use a restrained lateral or
forward gimbal treatment, and reject the candidate if architecture, landscaping, perspective, or
layout changes.

Images without a compatible, explicitly selected pair remain separate LTX shots and are joined by
intentional cuts. Select a lighting-only bridge only for near-identical framing of the same view,
such as `front-wide-daytime` to `front-wide-twilight`; it may alter sky, ambient light, and
practical lighting, but must keep camera framing and property geometry fixed. Any reel containing
invented bridges must be labelled **Synthetic architectural visualization** rather than a verified
walkthrough.

The current demo exposes three reviewed candidate IDs for the committed synthetic property:
`front-left-to-front-wide`, `patio-to-backyard`, and `front-day-to-twilight`. Arbitrary
user-defined view pairs and inserting approved bridges into `assemble-ltx` are not implemented yet;
the assembler currently accepts independently approved source clips only.

```powershell
uv run --no-sync python -m listing_to_reel video assemble-ltx `
  --render-manifest runs/ltx-videos/<run-id>/manifest.json `
  --accepted-clip 03-front-day-left `
  --accepted-clip 05-front-day-wide `
  --accepted-clip 02-covered-patio `
  --accepted-clip 04-backyard-patio `
  --accepted-clip 01-front-twilight
```

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
