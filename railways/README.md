# Railways — quick commands

Run from the **repository root**, using the existing `.venv`. Replace `/path/to/...`
with your inputs and choose fresh output paths for each run.

## One frame — print rail coordinates as JSON

```bash
.venv/bin/python -B -c 'import json, sys; from railways.candidate_script import process_image; print(json.dumps(process_image(sys.argv[1]), indent=2))' \
  /path/to/frame.png
```

For a frame **with an overlay**, put it alone in a directory and use the next command.

## Images / OOD dataset — overlays and metrics

```bash
.venv/bin/python -B railways/run_candidate.py railways/ood_dataset \
  --recursive --output-dir railways/ood_results_new
```

Accepts a directory, not a single file. Saves `__debug.jpg` overlays and one
`metrics.json`; ground truth is not required. Omit `--recursive` for a flat directory.

## Video clip — frame overlays and consolidated JSON

```bash
.venv/bin/python -B railways/run_candidate_video.py /path/to/video.mp4 \
  --start-time 00:01:05 --end-time 00:01:30 \
  --every-n-frames 10 --output-dir railways/video_results_new
```

Saves selected-frame JPEGs and one `results.json`. Add `--no-overlays` for JSON only.
Omit timestamps to process the whole video. Sampling is relative to the clip's first
frame; at 25 FPS, every 10 frames means 2.5 selected frames per video second.

## Video clip — export an annotated MP4

```bash
.venv/bin/python -B railways/run_candidate_video_overlay.py /path/to/video.mp4 \
  --start-time 00:01:05 --end-time 00:01:30 \
  --streaming --output railways/annotated_new.mp4
```

Saves a silent MP4 plus `annotated_new.results.json`. This is **offline processing**;
export can take longer than the video's duration.

## Live player — 1× playback, best-effort detection

```bash
.venv/bin/python -B railways/run_candidate_video_live.py /path/to/video.mp4 \
  --start-time 00:01:05 --end-time 00:01:30 \
  --streaming --candidate railways/candidate_fast.py \
  --results railways/live_new.results.json
```

Shows live video and the last analyzed snapshot, FPS, and result age. Skips inference
frames when busy rather than slowing playback. **Q / Esc** exits. `--results` is optional.

Streaming has no search cooldown: empty detections are retried on the next
available frame; lost tracking triggers immediate full-frame reacquisition.
The old `--search-every` option has been removed. This does not shorten detector
latency; offline streaming also performs more work on negative frames.

For image/video runners, `--candidate PATH` selects `candidate_script.py` (default),
`candidate_fast.py` (speed alternative), or `candidate_conservative.py` (zero-FPR
validation alternative, not a guarantee on unseen images). `--streaming` enables
tracking and periodic verification; omit it for independent full-frame searches.

## Web review — saved live results

```bash
open railways/review.html
```

Choose the **original full-length video** and the JSON saved with `--results`.
The page plays the JSON's `start_time_seconds`–`end_time_seconds` range and draws
rail overlays on their original source frames. No server, upload, or inference
rerun is needed; files stay in your browser. The browser must support the video codec.

Use **Space** to play/pause, **← / →** to step one video frame, or **Shift + arrows**
to jump between recorded results. You can also scrub, jump to a frame number,
filter results to detections, loop the range, and change playback speed.
Shortcuts apply when focus is outside the controls.

Skipped frames show **No recorded result**, not a negative detection. Late results
appear retrospectively on the analyzed frame, not at their live arrival time.
Stepping assumes the constant FPS recorded in the JSON; use the matching source
video, not a trimmed or cropped export.

Review-page logic tests: `node --test railways/test_review.cjs`.

## Evaluate against ground truth

```bash
.venv/bin/python -B railways/evaluator.py \
  railways/candidate_fast.py railways/val_dataset \
  --debug-dir railways/eval_results_new
```

Reports IoU, FPR, and timing; saves annotated images and `metrics.json`.
Prefix with `RAILWAYS_EVAL_WORKERS=1` for single-worker timing comparisons.

## Generate and evaluate new candidates

```bash
RAILWAYS_CODEX_MODEL=gpt-6-astra \
RAILWAYS_CODEX_REASONING_EFFORT=medium \
RAILWAYS_ARCHITECT_MODEL=gpt-6-astra \
RAILWAYS_ARCHITECT_REASONING_EFFORT=medium \
caffeinate -is .venv/bin/python -B railways/orchestrator.py
```

`caffeinate` prevents idle sleep on macOS; keep the lid open. Omit it on other OSes.
Run artifacts go to `railways/runs/`. Main-candidate replacement requires strictly
higher dataset IoU and FPR ≤ 10%; speed alone does not qualify a replacement.

See the [main README](../README.md) for setup, detailed options, and limitations.

## Generate candidates with Claude CLI

Requires a recent `claude` CLI on PATH (integration checked with 2.1.280).
Sign in once if needed: `claude auth login`.

```bash
RAILWAYS_LLM_PROVIDER=claude \
RAILWAYS_CLAUDE_MODEL=sonnet \
RAILWAYS_CLAUDE_REASONING_EFFORT=medium \
RAILWAYS_ARCHITECT_MODEL=opus \
RAILWAYS_ARCHITECT_REASONING_EFFORT=medium \
caffeinate -is .venv/bin/python -B railways/orchestrator.py
```

Both generation/repair and architecture review use Claude. To use Opus for both,
set `RAILWAYS_CLAUDE_MODEL=opus`. Aliases follow the CLI's model selection; use
full model IDs to pin a version. Existing evaluation, acceptance gates, candidate
preservation, and run artifacts are unchanged. Set `RAILWAYS_LLM_PROVIDER=codex`
(the default) to switch back, with appropriate Codex/architect model settings.

`RAILWAYS_CLAUDE_TIMEOUT` and `RAILWAYS_ARCHITECT_TIMEOUT` default to 600 seconds.
Claude uses print mode with tools/MCP disabled and no persisted conversation.
Diagnostic images are embedded in stdin; large files are JPEG-encoded without
resizing the local attachment dimensions. Claude may resize images internally.
Only transport copies change; evaluator images remain untouched.
