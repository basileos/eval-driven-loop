# Spec — make `candidate_fast.py` fast

Status: Phases 0, A, B and C evaluated and implemented 2026-10-24.

### Phase C results and the Phase B correction

**Correction to Phase B.** On real drone footage, `SEED_BUDGET = 60` lost recall. These numbers
are frames with tracks, out of 125 frames from `REC_0023.mp4` 1:05–1:30 (every 5th frame):

| Budget | 60 | 100 | 140 | 180 | 220 | 280 | original |
|---|---|---|---|---|---|---|---|
| Frames with tracks | 63 | 76 | 86 | 89 | 90 | 90 | 91 |

The val and OOD sets were not representative of this footage. `SEED_BUDGET` is now **220**:

- val score 0.6184, the same as Phase A.
- OOD 7/7 positives and 0/8 negatives.
- Live time on 4K video frames: 0.51 s mean, 0.82 s p95. The original took ~1.46 s here
  (1.50 s through `process_image`, minus the PNG decode, plus the BMP round trip).
- On the dataset frames: 0.37 s.

`NO_RETRY_GATES` (B4) has no meaningful recall cost on the video: 65 vs 63 frames at budget 60.

End-to-end streaming run on the same clip (`run_candidate_video.py --streaming`):

| | Original | Final |
|---|---|---|
| Frames with tracks | 95/125 | 95/125 (same search/track pattern) |
| Full search | 1.44 s | 0.55 s |
| Mean per frame | 1.18 s | 0.48 s |

| Item | Outcome |
|---|---|
| **C1** half-resolution working image | Rejected. Val score 0.6211 → 0.5645, OOD 6 → 4/7, and *slower* on video frames (0.365 vs 0.329 s). LSD already runs at a fixed 1280×720, so C1 only shrinks the small strip work. Kept as the `WORK_SCALE` knob (default 1.0). |
| **C2** `process_frame(frame, reference_shape=…)` | Shipped. A crop is searched with full-frame limits and LSD scale. On 28 tracked frames (ROI ≈ 27% of the frame) it re-finds the track 18/28 vs 17/28 without the reference shape, at 0.207 vs 0.231 s. `run_candidate_video` picks it up automatically. |
| C2 seeding from the previous track (skip LSD) | Not done. The tracker API does not pass the predicted track. Also, ROI checks almost never run in practice (next row). |
| **Tracker finding** | On the video, 110/125 streaming frames were full searches. Optical-flow propagation fails its forward-backward check at 10-frame gaps (0.4 s, about the live interval): full-resolution 4K LK cannot follow drone motion. Fixing propagation (e.g. LK on a ¼-resolution pyramid) is the largest remaining live-throughput lever. A tracked frame costs ~0.02 s vs ~0.4 s for a search. |
| **C3** numba | Not done. numba is not installed. LSD (C++) and Python-level proposal work dominate, so a rewrite would be large for a modest gain. |
| LSD `REFINE_NONE` (extra experiment) | Not shipped, because it is slower (0.305 vs 0.261 s). But it *raised* val score 0.6211 → 0.6467 (precision 0.90, 16/23 detected, OOD unchanged), so it is worth a separate quality experiment. |

### Phase B results

Live path (`process_frame`), 81 frames, single-threaded. "Changed images" counts frames
whose output differs from the Phase A golden file.

| Variant | val score | val FPR | OOD pos | changed images | live mean / p95 |
|---|---|---|---|---|---|
| Phase A | 0.6184 | 0.024 | 7/7 | — | 0.498 / 0.808 s |
| **B4** retry gating | 0.6184 | 0.024 | 7/7 | 0 | 0.403 / 0.617 s |
| B4 + B1 budget 120 | 0.6211 | 0.000 | 6/7 | 3 | 0.306 / 0.439 s |
| **B4 + B1 budget 60 (shipped)** | **0.6211** | **0.000** | 6/7 | 3 | **0.261 / 0.385 s** |
| B4 + B2 suppress near failures | 0.6159 | 0.024 | 7/7 | 2 | 0.428 s (slower) |
| B4 + B7 LSD 960×540 | 0.6007 | 0.024 | 7/7 | 25 | 0.356 s |

Shipped settings: `NO_RETRY_GATES = {length, green, ridge, coherence}` and `SEED_BUDGET = 60`.

Changes from Phase A (3 of 81 images):

- `frame_00-08_151`: loses an extra, imprecise track (precision 0.47).
- `frame_00-34_554`: loses its false positive.
- `ood/positive/…veliko-tarnovo…`: no longer detected. Its only track verified at seed #200.
  Set `SEED_BUDGET = 280` to restore it, at about 2× the cost.

Rejected or skipped:

- **B2** loses tracks: seeds close to a failed seed do sometimes verify, and it is slower.
- **B3**: fit strength does not separate good seeds from bad ones. A threshold of 2.0 removes
  only 4% of verifications, and accepted seeds all have strength ≥ 9.3.
- **B5/B6**: with budget 60, verification plus fitting is only 0.065 s per frame, so there
  is little left for these to save.
- **B7** lowers val IoU.
- **Exact pair-prefilter for proposals**: gave no measurable gain. Proposal generation costs
  about 27 ms in pairs and 18 ms in local strips, spread as Python overhead across about
  880 segments.

The Phase A golden file is kept as `runs/golden_fast_phaseA.json`. `runs/golden_fast.json` now
holds the Phase B outputs.

Remaining time per frame (0.26 s):

| Stage | Time |
|---|---|
| LSD | 0.125 s |
| proposals | 0.057 s |
| verification | 0.042 s |
| fit | 0.023 s |
| preprocess | 0.012 s |

From here the large wins are Phase C: C1 (half resolution) and C2 (ROI mode).

### Phase 0 + A results (81 val/ood/images frames, 4K, `setNumThreads(1)`)

| | before | after |
|---|---|---|
| `process_image` mean / p95 | 1.14 s / 1.78 s | 0.65 s / 0.95 s |
| live path (BMP round trip → `process_frame`) mean / p95 | ~1.13 s / ~1.8 s | **0.50 s / 0.74 s** |
| verification stage | 0.65 s | 0.22 s |
| golden outputs | — | 81/81 identical |

What was done:

- Phase 0: `bench_fast.py`, golden file `runs/golden_fast.json`, `test_candidate_fast.py`, and
  exclusive stage timers (`tracing` became `seed_loop` + `trace`).
- A1: `process_frame(frame)`, without `reference_shape`. Accepting it would change ROI-mode
  thresholds, so it moves to C2.
- A2: `evidence` gates reordered.
- A3: `morphology` became a boolean `morphology_ok` with early exit.
- A4: the shift correlation now uses `sliding_window_view`, and the autocorrelation uses
  running sums.
- A6: `strip`/`remap` build contiguous 2-D maps and remap valid rows only.
- A8: the vegetation mask uses a uint8 LUT, exact over all (g, r) pairs.

What was skipped: A5 and A7. After the changes above they measured as negligible
(`seed_loop` and `trace` together take < 15 ms).

Remaining time on the live path, per frame:

| Stage | Time |
|---|---|
| verification | 0.22 s |
| LSD | 0.13 s |
| fit | 0.07 s |
| proposals | 0.06 s |

The next step is Phase B, which reduces the ~240 seed verifications per frame.

## 1. Goal

Today a 3840×2160 frame takes **~1.2 s** in a normal run and **~1.7–2 s** in the live
worker. The live worker calls `cv2.setNumThreads(1)` and writes a BMP for every frame.

| Target | Per-frame latency (live worker, 4K) |
|---|---|
| Phase A (no change to output) | ≤ 0.8 s |
| Phase B (budget/pruning, gated by eval) | ≤ 0.35 s |
| Phase C (resolution / ROI) | ≤ 0.15 s typical, full search ≤ 0.35 s |

Rule: detection quality on `val_dataset` and `ood_dataset`, as measured by `evaluator.py`,
must not regress. Phase A changes must give the **same output** as today (rail coordinates
within 1e-3 px, same number of detections).

## 2. Measurements (20 val frames, 4K, M-series Mac)

Stage means. In the code, `tracing` *contains* `profile_fitting` and `verification`
because the timers are nested.

| Stage | default threads | `setNumThreads(1)` (live) |
|---|---|---|
| decode (PNG `imread`) | 0.19 | 0.21 |
| preprocess | 0.02 | 0.05 |
| LSD (1280×720) | 0.13 | 0.18 |
| proposal generation | 0.07 | 0.10 |
| seed loop (`tracing`), total | **0.75** | **1.12** |
|  ↳ verification (`evidence`) | 0.60 | 0.92 |
|  ↳ fit_section | 0.14 | 0.17 |
| **total** | **1.16** | **1.66** |

Live-only cost: `cv2.imwrite` of a 4K BMP takes **0.14 s** per frame
(`run_candidate_video.detect_decoded_frame`), because `candidate_fast` has no `process_frame`.

Counts per frame:

- `fit_section` ≈ 950 calls, `verify_seed` ≈ 237, **`evidence` ≈ 972**, of which only
  **2.5 seeds get traced** and ~1 is accepted.
- Where `evidence` rejects:
  - 64.6% at the `morphology` gate (line 246)
  - 23.4% at the `coherence < 0.40` gate (line 275)
  - 6.4% at the green-core gate (line 244)
  - 0.4% pass
- Accepted detections in 64 frames (val + ood) came from seed check numbers
  `1,1,1,1,2,2,3,3,6,6,6,7,9,10,18,23,178,204`. That means **16/18 come from the first 23
  seeds**. Frames with no rail use up the full 280-seed budget, so **empty frames are the
  slowest** (1.6–1.9 s).
- cProfile hotspots (in-process): `morphology` 33% cumulative, 1.37 M `np.mean` calls
  (~20% just in numpy dispatch overhead), `corr` 288 k calls, `strip`/`remap` 15%,
  `GaussianBlur` 53 k calls.

Conclusion: the cost is mostly **too many small Python/numpy calls during seed
verification**, not heavy pixel work. Two fixes follow: make each call cheaper (Phase A)
and make far fewer calls (Phase B).

## 3. Phase 0 — safety net (do first)

1. **Benchmark script** `railways/bench_fast.py`:
   - Runs N frames under `cv2.setNumThreads(1)`.
   - Prints the mean, p50 and p95 of each stage plus the call counters above.
   - Optional `--profile` flag for cProfile.
2. **Golden outputs**: save today's `process_image` output for all val and ood frames to
   `railways/runs/golden_fast.json`. Add a test that checks Phase A changes against it,
   with coordinate tolerance 1e-3 and the same detection count.
3. **Fix the timers** so the stages do not overlap. Rename `tracing` to `seed_loop` and add
   `trace` (the stepping only). This way the stage sum equals `total` and the evaluator's
   stage report is correct.

## 4. Phase A — same output, cheaper calls (expected ~1.66 → ~0.7 s live)

| # | Change | Where | Est. saving |
|---|---|---|---|
| A1 | Add `process_frame(frame, reference_shape=None)` that takes decoded BGR pixels. Make `process_image` a thin wrapper around it. This removes the BMP encode/decode in live mode and turns on the tracker's ROI hook (Phase C). | top of file | 0.14–0.2 s live |
| A2 | **Reorder `evidence` gates from cheapest to most rejecting.** Every gate returns `0.0`, so `evidence` is a pure AND of gates and the order does not change the result. Move the coherence/cross-shift test (23% rejects, cheap once vectorized, see A4) and the ridge-strength test (line 305) **before** `morphology`. Let the benchmark's per-gate cost × reject rate decide the final order. | `evidence` | large: morphology runs on ≪ 65% of calls |
| A3 | **Vectorize `morphology`.** For polarity −1, `a`, `b` and `inner` are just the negatives of their polarity +1 values. Compute them once and take both signs. Replace `np.array_split(…,3)` + list comprehension with three slice means. Replace `np.mean(bool)` with `count_nonzero / n`. Use the `ndarray.mean` / `np.add.reduce` fast path instead of `np.mean` (removes about 1 M dispatches). | `morphology` | ~60–70% of morphology time |
| A4 | **Vectorize the shift correlation.** `scores` makes 2 × (2·limit+1) Python `corr` calls (up to 130 per `evidence`). Replace them with one `sliding_window_view` + centered normalized dot product: one matmul per side. Do the same for the autocorrelation over `lags`. The lengths vary, so use cumulative sums of x, x² and the product via `np.correlate`. | `evidence` | ~0.1–0.15 s |
| A5 | **Hoist invariants out of `_detect`.** The nested closures are rebuilt on every call. More importantly, `normal(direction)`, the `across`/`along` ranges, masks such as `groups`/`outer_masks`/`core`, and the GaussianBlur sigmas depend only on `(gauge, extent)`. Cache them with an LRU keyed on rounded `(gauge, extent, outer)`. | `strip`, `evidence`, `fit_section` | ~5–10% |
| A6 | **`remap` hygiene.** Build `xy` straight into float32 `mx`/`my` without going through float64 broadcasts. Compute the `valid` mask with one `cv2.inRange`-style min/max on the core columns, not four `.all()` passes. Skip the `green` remap when the caller does not need it (it does in `fit_section`, but that gate could use a single `cv2.mean` over the core region). | `remap` | ~0.05 s |
| A7 | **Seed-loop dedup.** `path_distance(center, old)` loops over `accepted` for every proposal. That is fine (≤ 6 paths), but `np.diff`/norm on each path gets recomputed: cache segment arrays per accepted path. | seed loop | small |
| A8 | **Preprocess.** Compute `vegetation` with `cv2.split` + `cv2.subtract`/`cv2.compare` on uint8 (saturating) instead of int16 numpy temporaries. Compute it lazily for the remapped strips only, or once at half resolution with `INTER_NEAREST` sampling (see C1). | preprocess | 0.02–0.04 s |

Exit criterion: the golden test passes, and the live benchmark shows ≤ 0.8 s mean and
≤ 1.0 s p95.

## 5. Phase B — fewer calls (behavior changes, gated by evaluator)

Each item goes behind a module-level constant so the eval loop can sweep it.

| # | Change | Rationale |
|---|---|---|
| B1 | **Adaptive seed budget.** Lower `seed_checks` from 280 to ~60. Also stop after *K* (≈40) consecutive failed `verify_seed` calls once `output` is non-empty, or after ~80 when it is empty. | 16/18 accepts happen at seed ≤ 23. The cap mostly burns time on empty frames. Report the recall impact of the two late accepts (178, 204) explicitly. |
| B2 | **Negative suppression.** When `verify_seed` fails at `(center, angle, gauge)`, skip later proposals within `0.5·gauge` and 0.1 rad of it. Today the `attempted` key quantizes too finely: each cell keeps up to 14 proposals, so close neighbours get re-verified. | Cuts the ~237 verifies per frame. |
| B3 | **Cheap pre-gate before `verify_seed`.** Use what `fit_section` already returns (`strength`, persistence) and require e.g. `strength ≥ 2.0` before paying for a 460-row strip. | Most seeds fail at morphology. A similar ridge test on the small strip is ~10× cheaper. |
| B4 | **Only retry sub-windows on a near miss.** `verify_seed` and `verify_path` call `evidence` up to 3 more times when the full window fails, which turns 237 verifies into 972 `evidence` calls. Retry only when the full-window failure got past the morphology gate (have `evidence` return a reason code). | Up to 4× fewer `evidence` calls. |
| B5 | **Normalize strip scale.** Strips for large gauges (up to 180 px at 4K) are ~460×510 px. Resample every strip so that gauge ≈ 40 px (`along`/`across` step = gauge/40). Rescale the gauge-relative thresholds, but keep intensity thresholds as they are. | Patch area falls by up to 20× near the camera, so remap, blur and matmul cost become roughly constant. |
| B6 | **Trace angle search.** Try `angle=0` first and test the other two angles only if that fit is missing, or its shift or strength is poor. | Most of the 950 `fit_section` calls come from tracing (3 per step × 2 directions). |
| B7 | **Smaller LSD input.** Run LSD at 960×540 instead of 1280×720, or tune `LSD_REFINE_NONE`. | LSD 0.13–0.18 s → ~0.07 s. Proposal recall must be checked. |

Order: B4 → B1 → B2 → B3 → B6 → B5 → B7, from easiest/safest to most invasive. Run the full
evaluator after each one and keep it only if the metrics hold.

## 6. Phase C — architecture (bigger wins for live)

- **C1 Half-resolution working image.** Run all of `_detect` on a 1920×1080 image (gray +
  vegetation), and scale the output back ×2. Most thresholds already scale with the image
  diagonal. The absolute ones need a review: `3.5` px minimum gauge, `extent` clips
  (28/80/460), `step` clip (10/30), and the `max(2, …)` floors. Decode stays at 4K, but
  `IMREAD_REDUCED_COLOR_2` makes the PNG decode about 3× cheaper when going through
  `process_image`.
- **C2 ROI mode.** With A1 in place, implement `reference_shape` so `StreamingTracker`
  can send ROI crops between full searches (`verify_every=10`, `full_search_every=150`).
  In ROI mode, skip LSD and proposals, and seed straight from the previous track
  (`trace` + `verify_path` only).
- **C3 Optional JIT.** If Phase A + B do not reach the target, move `morphology`,
  `evidence` and `fit_section` scoring to numba (`@njit`, float32). This adds a dependency,
  so decide only after measuring A and B.
- Not recommended: threading or multiprocessing inside a single frame. The patches are
  small and the GIL is held for most numpy calls this size, so it would add complexity for
  little gain.

## 7. Deliverables / acceptance

1. `bench_fast.py` plus golden-output test (Phase 0).
2. PR per phase: A (one PR, golden-identical), B (one PR per item with evaluator
   before/after in the description), C1/C2 separately.
3. Final report: per-stage table like §2, before vs after, for both empty and rail frames,
   with `setNumThreads(1)`.

## 8. Streaming tracker (implemented 2026-10-25)

**Finding.** In live use the tracker never tracked. Every processed frame was a full search
(0.5 s).

- A search takes about 0.5 s, so the first propagation after it must bridge about 12 frames.
  Drone motion over that gap is 60–160 px and is not a pure shift.
- Full-resolution LK optical flow failed its forward-backward check, and the
  `0.65 × roi_margin` (62 px) motion cap rejected the rest.
- Downscaling LK did not help.

**Changes in `streaming_tracker.py`:**

1. **Motion model.** Motion is now a RANSAC homography over ORB matches.
   - Matching runs at ≤ 960 px (`motion_max_side`), with sub-pixel refinement via
     `cornerSubPix`.
   - Features are computed once per frame.
   - Motion is measured from the **keyframe**: the last frame whose geometry came from the
     detector. Error therefore does not accumulate.
   - On 18 real frame pairs, propagation succeeds 18/18 at gaps of 1–15 frames. Its error
     against a fresh detection (18–27 px) equals the detector's own frame-to-frame variation
     (~23 px).
2. **Sanity checks.** Zoom 0.67–1.5× (altitude changes reach about 1.3× within a second),
   rotation ≤ 0.35 rad, and the rail spacing must follow the zoom (`gauge_changed`).
3. **Clipping.** Propagated rails are clipped at the frame border (`clip_polyline`)
   instead of dropping the track (`left_frame` was the top failure).
4. **Association** (`association_gauges=1.5`). It uses `support_distance`: the directed
   median distance from the predicted rails to the candidate. The limit is
   max(12 px, 1.5 × gauge).
   - The old fixed 12 px limit was below the detector's own about one-gauge variation, so
     every verification failed.
   - An adjacent parallel track (about 3 gauges away) is still rejected.
5. **Verification on the full frame** (`verify_region='frame'`, default). An ROI crop can
   never extend the track past the crop, so tracks only shrank. `'roi'` is still available.
6. `propagation_failure`, `motion_scale` and `motion_rotation` are reported in the
   diagnostics.

**Results on `REC_0023` 1:05–1:30, candidate_fast:**

Live simulation (1× playback, frames skipped while busy):

| | Old tracker | New tracker |
|---|---|---|
| Results per second | 1.84 | **5.08** |
| Tracked frames | 0 | 99/127, at 66 ms |
| Track shown | 74% of time | 70% of time |
| Tracked rails vs fresh detection | — | median 24 px, p90 42 px |
| Tracked length / detected length | — | median 1.00 |

`run_candidate_video.py --streaming` (every 5th frame):

| | Before | After |
|---|---|---|
| Frames with tracks | 95/125 | **99/125** |
| Full searches | 102 | **41** |
| Mean per frame | 0.48 s | **0.22 s** |
