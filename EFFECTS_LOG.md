# Effects improvement log — Fable 5 session (2026-07-05 weekend)

Working log of the effects overhaul. Each entry records the effect, what it
looked like **before**, what changed, and the **after** (visual + perf).
Timings are for the reference clip: 4s / 960x540 / 30fps (120 frames),
rendered locally on Karim's M5 Pro (`baseline/` in session scratchpad).

## Session fixes before effects work

- **Committed prior session's uncommitted hardening** (`abe56d3`): streamed
  uploads with 500MB cap, try/finally resource cleanup in process_video,
  abortable frontend requests, upload a11y.
- **Face effects were hard-crashing** (`1f1dd3c`): mediapipe 0.10.35 removed
  the legacy `mp.solutions` API → `ocular_overload`, `face_scanner`,
  `face_mesh` crashed (500) and person segmentation silently degraded for
  every subject-isolation effect. Pinned `mediapipe>=0.10.9,<0.10.30`
  (+ numpy<2, opencv<5 to match) and made `FaceDetector` degrade gracefully.
  ⚠️ Prod note: a fresh Render build before this pin would have shipped broken
  face effects.

## Baseline (all 36 presets, seconds for 4s clip)

blob_track 1.56 · numeric_aura 1.97 · ocular_overload (crash→fixed) ·
dither_trace 1.97 · contour_trace 1.15 · face_scanner (crash→fixed) ·
matrix_mode 2.24 · signal_map 0.90 · face_mesh (crash→fixed) ·
data_body 1.70 · motion_flow 3.48 · motion_trace 1.42 · grid_trace 1.35 ·
heat_map 2.39 · catodic_cube 1.41 · ember_trails 1.74 · soft_blobs 1.67 ·
codenet_overlay 1.97 · code_shadow 1.72 · binary_bloom 2.05 ·
signal_feedback 2.96 · signal_bloom 1.66 · glyph_trace 0.78 · slit_scan 0.69 ·
chromatic_ghost 1.09 · crystallize 3.59 · halftone 1.71 · light_trails 1.83 ·
ink 1.68 · kaleidoscope 1.28 · tv_static 2.86 · ascii_core 1.68 ·
neon_glow 1.55 · point_cloud 3.01 · blacktone 1.96 · cursor_cloud 1.29

## Inventory findings (issues to fix)

**Cross-cutting**
- Unseeded `random` per frame → 30Hz strobing in matrix_mode, numeric_aura
  path, binary_bloom (partially seeded).
- uint8 trail canvases (motion_trace, light_trails) → quantization banding,
  trails die in visible steps.
- `draw_scanlines` darkens rows in a Python loop.
- Static glyph tiles rebuilt every frame (ascii_core, glyph_trace,
  cursor_cloud); signal_feedback rebuilds full-res meshgrid + blurs full-res
  noise (51x51) every frame.
- Dead code: `draw_numeric_aura`, `draw_thermal_scan` (calls a function that
  doesn't exist), `draw_thermal_scan_slow`; stray root-level scripts
  `final_bloom_red_fix.py`, `new_bloom_refinement.py`, `debug_effects_crash.py`.

**Per-effect (planned upgrades)**
- matrix_mode: strobing glyphs + jittering trail lengths; flat look, no
  phosphor glow → rewrite with stable per-column state + deterministic glyph
  cycling + head bloom.
- blob_track: fresh detection each frame → boxes flicker, IDs hop between
  objects → temporal IoU matching, smoothed boxes, persistent IDs.
- crystallize: mesh re-seeded per frame → facet strobing; slowest effect →
  LK-tracked seed points (mesh flows with the image), vectorized facet color.
- number_cloud (numeric_aura preset): thousands of putText calls/frame, 30Hz
  digit strobe, contour-mask jitter → glyph-tile rewrite + person mask +
  eased flicker.
- binary_bloom: nested putText grids → tile-based, calmer flicker.
- neon_glow: per-frame Canny flicker → temporal edge persistence (like
  contour_trace already has).
- signal_map: full-frame copies per data box; boxes teleport every 3 frames.
- halftone/blacktone: duplicated pattern code → shared helper (+ classic 45°
  screen angle).
- motion_trace: Python-loop point collection → numpy; float canvas.

---

## Per-effect change log (before → after)

(appended as work lands; timings re-measured on the same clip)

### Phase A — cross-cutting smoothness + perf (all landed together)

- **Dead code removed**: `draw_numeric_aura` (orphaned — preset routes to
  `draw_number_cloud`), `draw_thermal_scan` (called a function that doesn't
  exist — would NameError if ever hit), `draw_thermal_scan_slow` (self-labelled
  "not used", O(h·w) Python pixel loop), plus orphaned
  `final_bloom_red_fix.py` / `new_bloom_refinement.py` (old signal_bloom
  iterations, imported nowhere). −370 lines.
- **draw_scanlines**: row-by-row Python loop → single vectorized slice op.
  Used by every `scanlines: true` preset.
- **Motion Flow / Motion Trace** (`draw_motion_trace`):
  - Before: uint8 trail canvas — decay quantized, faint trails died in visible
    steps with banding; point collection was a Python double loop over the
    whole grid; flow field upscaled to full frame only to be sampled at grid
    points.
  - After: float32 canvas (smooth exponential decay), strokes drawn on a uint8
    ink layer and merged via `maximum` (cv2 drawing on float is slow), flow
    sampled directly from the half-res field, grid collection vectorized.
  - Perf: ~parity on the worst-case synthetic (3.5→3.8s motion_flow, whole
    frame moving); visual: banding gone, trails melt smoothly.
- **Light Trails**: same uint8→float32 canvas fix; long streaks now fade
  continuously instead of stepping. ~parity perf (1.83→1.65s).
- **Signal Feedback**: noise field moved from full-res (with a 51×51 blur per
  frame!) to 1/8-res smoothed fields upscaled once; coordinate grid + vignette
  cached per resolution; **two independent x/y noise fields** — the old single
  field could only displace along the 45° diagonal, so the "liquid" warp was
  actually a diagonal shear. Now it swirls in every direction. 2.96→1.92s
  (−35%).
- **ASCII Core / Glyph Trace / Cursor Cloud**: glyph/cursor tile banks were
  rebuilt every frame → now cached (keyed by cell size/ramp).

### Phase B — headline rewrites

- **Matrix Mode** (`draw_matrix_mode`):
  - Before: every glyph re-rolled with unseeded `random` each frame (30Hz
    strobe), trail lengths jittered per frame, katakana charset that Hershey
    fonts can't render, flat unlit look, Python putText loops.
  - After: grid-locked rain — per-column deterministic speed/trail/phase,
    stable glyphs with ~1/3 of cells slowly mutating, quadratic-eased trail
    fade, subject-luminance reveal, phosphor glow, vectorized tile compose.
    Same render cost (2.2s), enormously calmer + more cinematic.
- **Blob Track** (`draw_blob_track`) — flagship preset:
  - Before: fresh contour detection each frame; boxes flickered in/out and
    "ID 00" hopped between objects (index = size order).
  - After: real temporal tracking — greedy IoU matching, exponential box
    smoothing, persistent per-object IDs, fade-in on birth / fade-out over 5
    missed frames. Reads as genuine surveillance tracking.
- **Crystallize** (`draw_crystallize`):
  - Before: goodFeaturesToTrack re-seeded every frame → whole mosaic strobed;
    per-triangle patch-mean color; slowest effect (3.59s).
  - After: seeds tracked with pyramidal LK (mesh flows with motion), fresh
    corners merged via occupancy grid only where empty, facet color = one
    pre-blurred frame sampled at centroids. 3.59→2.04s (−43%).
- **Numeric Aura** (`draw_number_cloud`):
  - Before: thousands of putText calls/frame across two Python grid loops,
    digits strobing at 30Hz, contour-only subject detection.
  - After: full glyph-tile rewrite — scrolling hex background with smooth
    pixel scroll (np.roll), binary foreground with per-cell 4-11-frame flip
    cadence, person segmentation with heuristic fallback, mask built/blurred
    at 1/8 res (the full-res sigma-25 blur dominated cost), cyan glow pass.
    Digit depth gradient: dim blue edge → cyan → white-hot core.
- **Binary Bloom** (`draw_binary_bloom`):
  - Before: two nested putText grids per frame; whole-field digit re-roll
    every 3 frames.
  - After: tile compose with measured/centered glyphs (old ones clipped),
    hash-staggered per-cell flips (5-12 frame periods), sparser interior via
    stable cell dropout, bloom on the silhouette edge band.
- **Neon Glow**: added tube persistence — previous edges decay at 0.58/frame
  via max-blend, so outlines cool down like real neon instead of blinking
  when Canny drops a line for a frame.
- **Signal Map**: the two full-frame `output.copy()` per data box replaced
  with crop-region blends (the hottest path in the effect).

### Phase C — polish + cohesion

- **CodeNet**: the tracker threw away ALL nodes every 12 frames — the whole
  mesh + every label popped and renumbered on a visible beat. Now nodes track
  continuously with persistent ids ("codecore N" stays on its feature), and
  fresh corners merge only into empty regions when the set runs thin.
- **Halftone + Blacktone**: deduplicated into shared pattern/luma helpers and
  moved onto a classic 45° rotated printing screen (reads as authentic
  newsprint instead of a computer grid).
- **TV Static**: bottom "signal melt" row loop vectorized to one gather.
- **Ocular Overload**: per-frame vignette rebuild cached per resolution.
- **Catodic Cube**: glitch was a single-frame tick exactly every 8 frames
  (metronome pop) → now 2-3-frame bursts on a hashed ~40% schedule with
  sine-eased strength.

### Frontend

- **Matrix Rain now reachable**: the "Matrix Mode" chip actually sent
  `data_body` — the real rain effect wasn't selectable at all (probably
  because the old one strobed). Added a proper "Matrix Rain" chip and
  relabelled "Data Body" honestly; copy bumped to 26 effects.
  ⚠️ Judgment call — see needs-input list.
- **prefers-reduced-motion**: decorative animation (rotating card glow,
  scanline sweep, chip scale transforms) now disabled for reduced-motion
  users; functional feedback kept.
- **Result stats count up** (~0.9s ease-out, reduced-motion guarded,
  tabular-nums so digits don't jitter).

### Full-suite regression (36/36 pass, seconds before → after)

Wins: crystallize 3.59→1.55 · signal_feedback 2.96→1.84 · matrix_mode
2.24→1.61 · tv_static 2.86→2.30 · point_cloud 3.01→1.39 · code_shadow
1.72→1.55 · light_trails 1.83→1.66 · motion_flow 3.48→3.33.
Face effects render again (were crashes). numeric_aura/binary_bloom/
cursor_cloud +0.5-0.6s each — that's mediapipe person segmentation actually
running now (it was silently broken); the visual upgrade is the point.

---

## Needs Karim's input

1. **`web/app/dashboard/` + `web/public/icons/*.png`** (untracked): a mock
   "Intern Docs" dashboard rebuilt from a Figma frame with fake revenue data —
   unrelated to Aftertrace. Left untouched and uncommitted. Keep, move to its
   own repo, or delete?
2. **Prod deploy**: the mediapipe pin + all effect improvements are committed
   locally only. Nothing pushed/deployed without your approval.
3. **Deleted scratch files**: `app/services/final_bloom_red_fix.py` and
   `new_bloom_refinement.py` (orphaned old signal_bloom iterations, imported
   nowhere) are removed — recoverable from git history if you want them.
   `backend/debug_effects_crash.py` left in place.
4. **"Matrix Rain" chip (judgment call)**: the picker's "Matrix Mode" chip
   secretly sent `data_body`. I split them: "Matrix Rain" → the (rewritten)
   `matrix_mode`, "Data Body" → `data_body`. If you'd hidden matrix_mode on
   purpose, revert the chip in `web/components/PresetPicker.tsx`.
5. **Sequence/overlay picker only exposes 26 of 36 presets** (grid_trace,
   heat_map, ember_trails, soft_blobs, face_scanner, face_mesh, motion_flow,
   signal_feedback, data_body-adjacent legacy ones are hidden). Intentional
   curation or worth surfacing some? Left as-is.
