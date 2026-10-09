# Project Overview: Eval-Driven Computer Vision Code Generator

This project is an eval-driven agentic loop designed to iteratively generate, test, and optimize Python computer vision algorithms using quantitative execution feedback.
Instead of relying on single-shot prompt generation, the system runs candidate scripts inside an isolated execution runtime, evaluates predictions against a local validation set (val_dataset/), computes task-specific objective metrics, and feeds stdout/stderr diagnostics back to the LLM to refine the code.

## DJI SRT height above terrain

`scripts/srt_agl.py` parses DJI subtitle telemetry, samples the frame nearest each
whole video second, queries SRTM 30 m terrain elevation, and outputs estimated
height above terrain (AGL):

```bash
./.venv/bin/python -B scripts/srt_agl.py \
  logs/DJI_20251112125720_0002_D.SRT
```

Output only the numeric AGL list with `--values-only`, write CSV with
`--format csv`, or use `--output PATH` to save the result. When the takeoff
coordinate is known, pass both `--home-lat` and `--home-lon` to calculate AGL
from takeoff-relative altitude and terrain change, avoiding absolute-altitude
datum bias.

Process every compatible `.SRT` file in a directory and save separate results
under `DIRECTORY/agl_results`:

```bash
./.venv/bin/python -B scripts/batch_srt_agl.py \
  --input-dir /path/to/srt/files \
  --values-only
```

Use `--recursive` to include nested directories and `--overwrite` to replace
results from an earlier run. The batch command skips incompatible `.SRT` files
and continues after individual processing failures.

List the minimum and maximum `height_above_terrain_m`, including the second at
which each extreme occurs, in every generated AGL result:

```bash
python3 scripts/summarize_agl_results.py
```

The default input is `logs/agl_results`. Use `--format json` or `--format csv`
for machine-readable output, and `--output PATH` to save the summary to a file.
Rows are sorted by minimum height in ascending order.

## System Architecture & File Responsibilities

- `railways/orchestrator.py`: Controls the railway search lifecycle. It evaluates generated source from persistent run directories and replaces `candidate_script.py` only after passing the fixed false-positive gate and strictly improving dataset IoU. The best positive-IoU zero-FPR detector is saved separately as `candidate_conservative.py`.
- `railways/candidate_script.py`: The retained railway detector. It implements `process_image` with OpenCV/NumPy and returns paired rail polylines.
- `railways/evaluator.py`: Runs validation images in a process pool, scores filled corridor polygons, emits scalar and geometry diagnostics, and exports separate native-resolution ground-truth-versus-prediction images.
- `railways/val_dataset/`: Railway validation images and annotations. An image without matching JSON is an intentional negative/unacceptable-quality sample.
- `docs/railways-fix-plan.md`: Detailed railway contract, evaluator, and search-controller decisions.

## Strict API & Interface Contracts
1. Common Candidate Contract: `candidate_script.py` must export `process_image(img_path: str)` and return `[]` if the image cannot be loaded or no acceptable targets are found.
2. Circles Contract: Return bounding boxes as `[[x1, y1, x2, y2], ...]`.
3. Railways Contract: Return one dictionary per track as `[{"left_rail": [[x1, y1], ...], "right_rail": [[x1, y1], ...]}, ...]`, with at least two finite in-bounds points per rail. The evaluator scores dataset-level IoU over all filled polygons bounded by rail pairs: total intersection pixels divided by total union pixels. Correct negative images add neither reward nor penalty; false-positive polygons increase the union.
   Candidate detections whose shorter rail or overall corridor span is below 5% of the image diagonal are considered undersized line-like clutter and should be rejected.
4. Evaluator Output Contract: `evaluator.py` emits `SCORE: <0.0000 to 1.0000>` and a machine-readable `METRICS` line. File acceptance requires positive dataset IoU strictly above the retained candidate and false-positive rate at or below the fixed configured ceiling. FPR may increase within that ceiling; lower FPR alone does not qualify an equal-IoU candidate. Coverage, polygon precision, and positive-image IoU guide search.

## Coding Guidelines for the LLM
- Code Integrity: Return complete, standalone, runnable Python code inside standard ```python blocks. Do not use placeholders or truncated functions.
- Allowed Libraries: Standard Library, cv2 (OpenCV), numpy, PIL, and torch.
- Optimization Focus: Treat incoming execution tracebacks and metric scores as the ground truth. Adjust image preprocessing, filtering, edge detection thresholds, and geometry extraction dynamically to improve the target IoU score.

## Railway Evaluator and Search

The railway evaluator processes images concurrently in up to four worker processes. Candidate `process_image` implementations must be deterministic, independent between images, and safe to execute in separate processes.

Detection runs at the source image resolution, including line and edge extraction.
The retained candidate no longer shrinks frames to a 1500-pixel long edge. Future
candidates are instructed to preserve native pixels; overlapping native-resolution
tiles are allowed, but downsampling rail evidence is not. This costs more CPU than
the former reduced-resolution detector, and its accuracy must be evaluated anew.
The evaluator already scores polygons in original image coordinates.

Model feedback also preserves source dimensions: selected originals and annotated
PNG copies are attached separately, with no thumbnail montage or combined-image
resize. Each generation uses up to four examples (eight image files), normally
two positives and two negatives. Parent diagnostics replace static references;
both bundles are not sent together. Every dataset image is still evaluated.
The evaluator's optional third positional argument is now a JSON manifest
path for these images, with companion files in `MANIFEST_STEM_images/`. Source
images remain unmodified. This controls the files submitted to Codex; the model
service's internal vision preprocessing is outside this script's control.

Run the optimization loop from the repository root with:

```bash
python3 railways/orchestrator.py
```

Export a full-resolution diagnostic image for every validation sample with:

```bash
./.venv/bin/python railways/evaluator.py \
  railways/candidate_script.py \
  railways/val_dataset \
  --debug-dir railways/debug_results
```

The directory contains one `__debug.jpg` per input image and `metrics.json` with exact per-image values. Ground truth uses red/blue rails and yellow fill; candidate output uses cyan/magenta rails and green fill. The image header reports IoU, coverage, precision, area ratio, normalized centroid offset, and minimum predicted rail length in pixels and as a fraction of the image diagonal.

Run the retained candidate on any directory of images without requiring ground-truth
JSON files:

```bash
./.venv/bin/python railways/run_candidate.py railways/images
```

By default this writes candidate overlays and raw rail coordinates to
`INPUT_DIR/candidate_results`. Use `--output-dir PATH` to choose another result
directory and `--recursive` to process nested image directories. The output has
one full-resolution `__debug.jpg` per input plus a `metrics.json` manifest with
per-image predictions, timings, size diagnostics, and errors.

Image and video overlays label each predicted track with `confidence=...`
(`uncalibrated`). The current candidate's `verified_patch_fraction_v1` score is
the fraction of sampled sections along the final track that pass its railway
evidence check. For example, 4 passing checks out of 5 gives 80%; it is **not**
an 80% probability of a correct detection, nor a localization/IoU estimate.
Checks can be correlated and confident false positives are still possible.
Each prediction in the JSON also includes `confidence` (0–1) and
`confidence_method`. Candidates without a score report `null` / `n/a`; an empty
prediction has no confidence score. These fields do not change detection
thresholds, geometry, or evaluation scoring. Existing results must be rerun to
include scores. Calibrated probabilities would require separate labeled data.

Run the railway candidate on every frame of a video file, stream URL, or camera:

```bash
./.venv/bin/python -B railways/run_candidate_video.py /path/to/video.mp4 \
  --output-dir railways/video_results

./.venv/bin/python -B railways/run_candidate_video.py /path/to/video.mp4 \
  --start-time 00:01:30 --end-time 00:02:15 \
  --every-n-frames 30 --output-dir railways/video_clip_results

./.venv/bin/python -B railways/run_candidate_video.py 'rtsp://host/stream' \
  --output-dir railways/stream_results --max-frames 100

./.venv/bin/python -B railways/run_candidate_video.py 0 \
  --output-dir railways/camera_results
```

The default candidate is `railways/candidate_script.py`; select another railway
candidate with `--candidate PATH`. All processed frames are saved in a single
`results.json` with `run` configuration, a `frames` array, and a `summary`.
Each frame record includes rail coordinates, source frame index, timestamp,
dimensions, inference timing, errors, and streaming diagnostics when enabled.
Optional `frame_00000000__debug.jpg` images use
the same overlay style as the image runner. Coordinates refer to the original
frame, excluding the overlay's added header. The retained detector accepts decoded
BGR arrays through `process_frame(frame)`, avoiding temporary image encoding and
repeated reads. Custom candidates exposing only `process_image(path)` use a
temporary lossless, uncompressed BMP; raw frames are not retained.

`results.json` is flushed after each result and is valid JSON between writes.
Appending rewrites only a small closing footer, not the previous frame records,
so results are not accumulated in memory. During processing `finalized` is false;
on completion or Ctrl-C it becomes true and `summary` contains totals and the
stop reason. Writes are not atomic against power loss or a forced process kill.
Separate frame JSONs, `frames.jsonl`, `run.json`, and `summary.json` are no longer
created; existing output directories are not migrated or deleted.
Completed results remain available if a later frame fails. Candidate errors are
recorded per frame and processing continues; the command exits nonzero if any
frame fails. Use a new or empty output directory to avoid overwriting results.
By default a file writes to `VIDEO_STEM_candidate_results` alongside the video;
streams and cameras write to `video_candidate_results` in the working directory.

For local files, use `--start-time` and/or `--end-time` to restrict processing.
Both accept seconds (`90.5`), `MM:SS`, or `HH:MM:SS`, with fractional seconds.
Start is inclusive and end is exclusive; end is an absolute video timestamp,
not a duration. The runner seeks to the start, preserves original frame indices
and timestamps, and records the bounds in `results.json` under `run`. Time ranges require a
valid reported FPS and are not supported for cameras or stream URLs.

Use `--every-n-frames 30` to process frames 0, 30, 60, etc. (or every 30th frame
starting with the first frame in the selected range), and `--max-frames 100`
to stop after 100 processed frames. The runner uses up to four worker processes
by default; change this with `--workers N`, or use `--workers 1` to disable
multiprocessing. Each worker loads the detector once. The number of queued frames
is bounded by the worker count, and results are written in source order.
OpenCV uses one thread per worker to avoid oversubscribing the CPU. Progress and
the `summary` in `results.json` report actual processing FPS (including output writes).
Use `--no-overlays` to skip JPEG rendering while retaining every frame's JSON.
Within the selected time range, no frames are intentionally skipped unless
`--every-n-frames` is specified (except latest-frame buffering for live streaming
mode, described below).
Live capture is best-effort: inference can lag
behind the source, and the backend may drop frames. Frame indices count decoded
frames; live timestamps measure elapsed capture-session time, not source PTS.
For files, timestamps use video position when available, falling back to
frame index / FPS; `timestamp_source` identifies which was used. An unreadable
frame ends capture; OpenCV does not reliably distinguish file EOF from a decode
failure. Stream disconnection exits nonzero without automatic reconnection.

### Real-time on-screen playback with best-effort detection

Use the live player to see what the current algorithm can detect **within the
video's actual duration**, instead of waiting for every frame to be processed:

```bash
.venv/bin/python -B railways/run_candidate_video_live.py /path/to/video.mp4 \
  --start-time 00:01:05 --end-time 00:01:30 \
  --streaming --results railways/live_clip.results.json
```

The selected 25 seconds play in approximately 25 seconds after detector loading.
Playback/decoding and inference run independently. Only one inference job may be
outstanding; once it finishes, the newest available frame is submitted. Missed
inference frames are skipped, never queued for later processing. Q, Esc, closing
the window, or Ctrl-C stops the player without waiting for in-flight inference.
The offline exporter below remains available when every selected frame matters.

The left panel is the live video; the right panel is the **last actually analyzed
frame**, with its own timestamp, rails, processing mode, and result age. Live
overlays disappear after `--max-overlay-age` seconds (default 0.25). Even a recent
overlay is delayed, not motion-corrected to the current display frame. Old results
remain visible only on their original analyzed snapshot. Cooldown/tracking-loss
replies hide the live overlay but do not erase the last useful snapshot. Slow full searches still
have real latency; asynchronous playback does not make the detector itself faster.

The HUD reports displayed FPS, useful updates/second (including optical-flow
tracking, excluding cooldown/lost/error replies), and detector calls/second.
Rates use a two-second rolling window. A tracker update is not a fresh full-frame
railway search. `--streaming` uses the existing tracker/ROI verification; omit it
to measure independent full-frame searches. Motion across skipped frames can
cause tracking loss and another expensive search. On overload, decoded/display
frames can also be skipped to stay on the playback clock.

Detection receives native-resolution pixels; only the two-panel display is
resized (`--width 1440` by default). This is a silent local-file player using
reported constant FPS, not exact variable-frame-rate/audio playback. `--headless`
benchmarks the same clock without a window (not GUI rendering performance).
`--results` is optional and writes one consolidated JSON with completed results,
input timestamps, result availability/age, skipped-frame counts, and summary.
No MP4 or per-frame files are written. An existing results file is not overwritten.

### Export an annotated video

Use the separate exporter when you want one playable video rather than individual
JPEG overlays:

```bash
.venv/bin/python -B railways/run_candidate_video_overlay.py /path/to/video.mp4 \
  --start-time 00:01:05 --end-time 00:01:30 \
  --streaming --output railways/annotated.mp4
```

This runs inference and writes `annotated.mp4` plus one consolidated
`annotated.results.json` containing run settings, frame predictions, and summary.
It does not create per-frame JPEG/JSON files. Overlays show cyan/magenta rails,
green corridors, confidence, original frame number/timestamp, and processing
mode. The input pixels retain their native resolution beneath an added header;
an extra black row/column is added only if the encoder needs even dimensions.

The exporter accepts local video files and processes in order with one worker.
Omit `--streaming` for independent full-frame detection. `--candidate`,
`--max-frames`, time ranges, and the streaming tuning options are supported.
`--every-n-frames N` saves only selected frames at source FPS / N, so sampling
does not speed up playback N times; it produces a lower-frame-rate video.
The output is constant-FPS based on reported source FPS (not exact variable-rate
timing), starts at playback time zero, and **does not include source audio**.
Source and output timestamps are both stored in the consolidated JSON.

Output paths must not already exist. Without `--output`, the exporter creates
`INPUT_STEM_overlay.mp4` alongside the input. `.mp4` defaults to `mp4v`; `.avi`
defaults to `MJPG`. Use `--codec FOURCC` only if that encoder is available in your
OpenCV installation. Ctrl-C finalizes the already-written video and JSON; forced
termination/power loss may leave a video without its final container metadata.

### Stateful video / streaming mode

```bash
.venv/bin/python -B railways/run_candidate_video.py /path/to/video.mp4 \
  --start-time 00:01:05 --end-time 00:01:30 \
  --streaming --no-overlays --output-dir railways/tracked_results
```

This optional mode runs the full detector for acquisition, propagates confirmed
rail geometry using local optical flow between frames, and periodically runs the
detector only on native-resolution crops around the tracked rails. Files retain
every selected frame; use `--every-n-frames 1` initially to keep motion small.
State is sequential, so `--streaming` always uses one worker. Default full-search
mode is unchanged. No UFLD/model installation or training is involved.

- `--verify-every 10`: fresh ROI evidence every 10 **processed** frames.
- `--search-every 30`: full-search cooldown when no track is available.
- `--full-search-every 150`: periodic global refresh, also discovering new tracks.
- `--roi-margin 96`: native-pixel margin for motion and local re-detection.
- `--max-track-gap 10`: hard timestamp-gap limit, allowing a handoff after slow
  acquisition. Even smaller gaps must pass forward/backward optical-flow and
  robust-transform checks; this does not allow blind extrapolation for 10 seconds.

Lost flow, failed ROI evidence, ambiguous association, and out-of-frame geometry
clear tracked output. During search cooldown the result is empty and marked
`search_wait` or `lost`: that is **not a fresh negative classification**. Tracking
estimates motion of the visible rails, not new geometry beyond the observed path;
ROI checks/refreshes supply new geometry. A local similarity transform is an
approximation and can fail under strong perspective changes or occlusion.

JSON's `streaming` object reports mode (`search`, `track`, `roi_verify`, `lost`,
`search_wait`, or `error`), detector call count/pixels/time, motion time/support,
and age since verification. Overlays show mode. Motion-only tracks have
`confidence: null`, not stale detector confidence. Motion support is a tracking
consistency measure, **not** railway probability. The full-image minimum-length
gate remains active for ROI outputs. The current candidate supports
`process_frame(frame, reference_shape=full_image_shape)` to preserve pixel-scale
thresholds on crops; legacy candidates without this optional argument may behave
differently on crops and need validation.

For live cameras/URLs, a background reader retains only the newest pending frame
and counts replacements as `dropped_pending_frames`. `--max-frames` limits saved
results, not captured frames. This bounds the application's queue, not buffering
inside the camera/decoder. `application_queue_seconds` and
`application_latency_seconds` exclude upstream capture/decoder latency. Capture
cleanup waits at most one second per close; a backend blocked in network I/O may
finish releasing later in its daemon reader thread.

Full acquisition still takes seconds and blocks inference; a moving scene can
invalidate the initial geometry before the handoff. ROI revalidation also adds
latency spikes. This is an experimental tracking accelerator, not a guaranteed
25-FPS detector or a validated autonomous navigation controller. Benchmark full
latency including acquisition/rechecks, track loss and false-track persistence;
do not report only fast motion frames as end-to-end FPS.

### Orchestrator settings

- `RAILWAYS_EVAL_WORKERS=<1..4>` controls the worker count.
- `RAILWAYS_EVAL_TIMEOUT=<seconds>` controls the entire dataset's pool deadline and defaults to 180 seconds. The outer evaluator process gets this budget plus 60 seconds for startup, cleanup, scoring and native-resolution diagnostics. Timeout messages report completed/total images; pending images include queued work, not just stalled frames. Timed-out evaluator process groups are stopped.
- `RAILWAYS_RUNS_DIR=<path>` defaults to `railways/runs`. Each run keeps its baseline source, generated source (including failed attempts), prompts, model responses, evaluation stdout/stderr, decisions, archive index, and native-resolution feedback. These files are not automatically deleted or resumed; native overlays can consume substantial disk space. The run directory and evaluation budgets are printed at startup.
- `RAILWAYS_CODEX_TIMEOUT=<seconds>` limits each Codex generation and defaults to 600 seconds, allowing for native-resolution image input. A persistent `GENERATION` line reports model, reasoning, timeout, image count, and total image-file size before each call. Timeout errors include the last available CLI output.
- `RAILWAYS_CODEX_MODEL=<model>` selects the implementation model and defaults to `gpt-5.6-luna`.
- `RAILWAYS_CODEX_REASONING_EFFORT=<effort>` defaults to `low` for bounded implementation turns; the loop ignores the user's global Codex profile.
- `RAILWAYS_ARCHITECT_MODEL=<model>` selects the plateau-analysis model and defaults to `gpt-5.6-sol`.
- `RAILWAYS_ARCHITECT_REASONING_EFFORT=<effort>` defaults to `medium`.
- `RAILWAYS_ARCHITECT_TIMEOUT=<seconds>` defaults to 600 seconds.
- `RAILWAYS_MAX_CANDIDATE_LINES=<lines>` defaults to 1000 (including comments and blank lines) and prevents unbounded candidate growth. The generation and repair prompts use this same configured cap.
- `RAILWAYS_MAX_REPAIR_ATTEMPTS=<count>` defaults to 1; set 0 to disable. Syntax errors, evaluation failures/timeouts, or excessive length trigger a focused repair of the same architecture within the same iteration before the search moves on. Each repair adds at most one generation call and one full-dataset evaluation, using the existing model and deadlines. A successful evaluation with poor IoU/FPR does not trigger a repair; all repaired candidates still face the unchanged acceptance gate. Authentication/transport/generation failures stop the search rather than consuming code-repair retries. Original and repaired source/logs are saved separately (`iteration_XX_strategy_repair_01.*`).
- `RAILWAYS_MAX_STAGNANT_ITERATIONS=<count>` defaults to 3 and triggers a disruption round after that many attempts without a new dataset-IoU best.
- `RAILWAYS_MAX_DISRUPTION_ROUNDS=<count>` defaults to 1. Set it to 0 to retain stop-on-plateau behavior.
- `RAILWAYS_PARETO_ARCHIVE_SIZE=<count>` defaults to 4 and limits the in-memory set of non-dominated search parents.
- `RAILWAYS_MAX_FALSE_POSITIVE_RATE=<0..1>` sets the fixed acceptance ceiling (default `0.10`): the fraction of negative images with any detection. With 41 negatives, at most 4 false-positive frames are permitted. The ceiling does not tighten when the retained detector's FPR decreases. It applies even without a valid baseline; an empty or zero-IoU detector cannot be accepted.
- The evaluator reports its image count, worker count, wall time, and slowest image before `SCORE`.
- Each result prints `SUMMARY`, `METRICS`, per-image geometry, and any `DIAGNOSIS`. Separate native-resolution originals and overlays supply the next model with the worst overlaps, a strong positive, and a false-positive negative.
- The orchestrator first evaluates the existing candidate as a baseline. File replacement requires the fixed false-positive gate plus strictly higher dataset IoU. Equal or lower IoU is never accepted. A baseline above the ceiling is preserved until a qualifying replacement is found; failure to find one is reported explicitly.
- `railways/candidate_conservative.py` independently preserves the highest dataset-IoU candidate with exactly zero validation FPR and positive overlap, even if it does not beat the main candidate's IoU or survive the Pareto archive. Baseline detectors qualify too. Existing conservative source is re-evaluated on each run (or reuses the baseline evaluation if identical), so changed datasets do not reuse stale metrics. Failed/nonqualifying alternatives remain on disk but are reported as unverified/nonqualifying; a newly verified alternative may replace them, with the previous source backed up in the run directory. Every improvement saves `conservative_best.py` and metrics in the run directory. This is validation performance, not a guarantee of zero false positives on unseen footage.
- Lower-scoring candidates or candidates failing the FPR ceiling may remain in the in-memory Pareto archive as search parents. They cannot replace the retained file or trigger target-score completion. Completion requires an accepted candidate with IoU at least `0.8`.
- All disruption branches and the post-disruption continuation use the lowest-FPR archive candidate retaining at least half of the archive-leading dataset IoU and positive-image IoU (ties favor dataset IoU; if none qualify, fall back to the dataset-IoU leader). No branch selects a parent solely for high positive-image IoU. Architecture diversity comes from branch instructions. This search-only heuristic avoids following near-empty detectors and does not relax the file-replacement gate.
- Empty-detector feedback asks for evidence-based recovery on positive examples while continuing to return `[]` for negatives. The loop never requires a detection on every decoded image.
- On a plateau, one architect turn plans four independent branches: segment graph/spline, corridor segmentation, multiscale path search, and piecewise LSD repair. Fast implementation turns then execute those branches from suitable archive parents.
