# Listing-to-Reel

An auditable generative-media pipeline for real-estate marketing.

## Phase 0 status

Phase 0 establishes reproducibility only. It does not run image or video models.

- `local_mps` is the Apple-silicon development and preview profile.
- `remote_cuda` is the authoritative inference and benchmark profile.
- Runtime validation never silently falls back from MPS or CUDA to CPU.
- Source-image rights and hashes are recorded in `data/manifests/`.

## Local setup

```bash
uv sync --extra dev
uv run ltr config-check --config configs/local_mps.yaml
uv run ltr environment
uv run pytest
uv run ruff check .
```

To probe your M5's MPS backend before model work, install PyTorch only:

```bash
uv sync --extra dev --extra mps
uv run ltr environment
```

Do not install the `gpu` extra on macOS for this phase. Install it in the remote NVIDIA worker environment.
