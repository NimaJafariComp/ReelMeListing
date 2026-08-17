# No-zoom LTX timed-reel render bundle

This bundle preserves the complete reproducible inputs and outputs for two
synthetic architectural visualization candidates generated on 2026-08-16:

- `reels/20s/reel.mp4` — 20.0 seconds at 1080x1920 / 30 fps.
- `reels/15s/reel.mp4` — 15.0 seconds at 1080x1920 / 30 fps; five-second
  `front-left-to-front-wide` bridge, then three-second patio and lighting bridges.

Both plans use lateral-gimbal source motion only; no dolly-in treatment was requested.
`render/` contains the native 16:9 LTX source clips, bridges, and full ComfyUI
workflow/source-hash manifest. `plans/` records the exact timing allocation, and
`qa/` contains the structural/temporal report and human-review worksheet.

Status: **rejected by automated structural QA**. These files are preserved as
review candidates and are not approved final property reels. See `qa/report.json`
for per-clip and per-bridge reasons.
