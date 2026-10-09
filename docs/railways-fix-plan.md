# Railways Task Fix Plan

## Status

The railway task now uses one paired-rail track contract across its prompt, candidate stub, evaluator, and project documentation. Each pair of rails is converted into a filled track polygon for evaluation.

## Recommended contract

Return paired rail polylines for each railway track:

```python
def process_image(img_path: str) -> list[dict[str, list[list[float]]]]:
    return [{
        "left_rail": [[x1, y1], [x2, y2], ...],
        "right_rail": [[x1, y1], [x2, y2], ...],
    }]
```

- Each dictionary represents one track and preserves its two-rail pairing.
- Each point contains finite pixel coordinates `[x, y]`.
- Each rail contains at least two points.
- Return `[]` when no acceptable tracks are detected or the image cannot be loaded.
- The evaluator closes and fills the area between the two rails, then scores polygon IoU.
- Do not return unpaired rail lists or track bounding boxes.

Rail order is normalized by the evaluator, so either rail may be drawn from either endpoint.

## Phase 1 — Correctness blockers

### 1. Align the task and API contract — completed

The railway system prompt, task description, candidate stub, evaluator, and project documentation now describe the same paired-rail track output.

The orchestrator uses one canonical railway task description. The evaluator validates candidate output before passing it to OpenCV, including a clear rejection of bounding boxes.

Acceptance checks:

- The initial prompt mentions railway detection only.
- A valid paired-rail track result is accepted.
- `[]` is accepted as a no-detection result.
- A bounding box such as `[x1, y1, x2, y2]` is rejected with a clear contract error rather than an OpenCV assertion.

### 2. Document the missing-JSON convention — accepted

Missing JSON is an intentional dataset convention. It means either that the image contains no railway or that any railway present is below the acceptable quality threshold. The evaluator scores such an image as having no track polygons.

Current decisions:

- `frame_00-44_378_original.png` was removed from the dataset.
- `frame_00-19_449_original.png` intentionally has no JSON and is treated as a negative/unacceptable-quality sample.
- Explicit JSON files with an empty `tracks` list remain valid negative annotations as well.

Add a read-only dataset validation command that checks:

- JSON `image`, `width`, and `height` values against the actual file when JSON exists;
- required keys and supported types;
- at least two points per emitted polyline;
- finite, in-bounds coordinates;
- and reports images without JSON as conventional negatives rather than errors.

Acceptance checks:

- The validator reports annotated positives, explicit negatives, and missing-JSON negatives separately.
- It exits nonzero for malformed, mismatched, or out-of-bounds annotations.
- Missing JSON does not cause a validation failure.

### 3. Reserve visibility metadata — accepted

The optional third point value is reserved for future use. The current evaluator intentionally reads only `[x, y]` and ignores that value.

No visibility or occlusion behavior will be derived from it in the current version. Revisit the field only when its future semantics are defined.

### 4. Score filled track polygons — completed

The evaluator no longer scores separate 8-pixel rail lines. For each track, it joins the first ends of the paired rails, follows both rail boundaries, closes the opposite ends, and fills the enclosed area.

Ground-truth polygons and predicted polygons are rasterized into masks. The primary score aggregates overlap across the complete dataset:

`dataset polygon IoU = sum(intersection pixels) / sum(union pixels)`

This measures agreement over the whole railway corridor between the rails, so small boundary errors are proportional to the track width rather than being judged as nearly complete line failures.

Correct negative images add neither intersection nor union. False-positive polygons on negative images add union and therefore reduce the score. An empty detector scores `0.0`; it cannot earn credit merely by returning `[]` for the three negative images. Per-image IoU remains diagnostic only.

Initial calibration results:

- exact annotation polygons score `1.0` on all 16 positive images;
- translating both predicted rails by 1 pixel gives mean positive-image IoU `0.9709`;
- translating them by 2 pixels gives `0.9432`;
- translating them by 4 pixels gives `0.8915`;
- translating them by 8 pixels gives `0.8001`.

The existing `0.8` target is retained as an initial threshold. Revisit it after observing realistic detector errors, which will include shape and endpoint errors rather than translation alone.

Calibration set:

- exact ground truth prediction;
- acceptable 1-, 2-, 4-, and 8-pixel offsets;
- shortened and overextended rails;
- one missed rail;
- one false-positive rail;
- correct predictions on negative images.

Acceptance checks:

- Exact ground truth scores `1.0`.
- Small acceptable localization errors remain high-scoring.
- Missed and false-positive rails receive meaningful penalties.
- The `0.8` target is reviewed again after the first realistic detector baseline.

## Phase 2 — Evaluator and loop reliability

### 5. Validate candidate output before scoring — completed

The evaluator now requires a list of track dictionaries, paired `left_rail` and `right_rail` polylines, at least two `[x, y]` points per rail, finite coordinates, and in-bounds points. Unpaired rails, `None`, `[[]]`, flat bounding boxes, strings, NaNs, and malformed arrays are contract errors with concise diagnostics.

Annotation parse failures and unreadable images should still be changed from warnings into evaluation failures; this is separate from the intentional missing-JSON convention.

### 6. Make execution location-independent and reproducible — partially completed

Resolve `candidate_script.py` and `val_dataset` relative to the railway module directory, not the caller's current working directory.

Evaluator and orchestrator paths are now location-independent. The orchestrator also runs Codex and evaluator subprocesses from the railway directory, regardless of the caller's working directory, and prints captured evaluator errors instead of only a generic failure message.

Document one supported launch command and dependency installation process. The system Python currently lacks OpenCV; the checked local `.venv` contains OpenCV 5.0.0 and NumPy 2.5.1, but a virtual environment alone is not a portable dependency specification.

Acceptance checks:

- Evaluation works from both the repository root and the `railways` directory.
- A clean environment can be recreated from the documented dependency file.
- A preflight check verifies Python, OpenCV, NumPy, and Codex CLI before the first generation call.

### 7. Preserve the best candidate — completed

The orchestrator measures the on-disk candidate as a baseline before generation and evaluates generated code from a temporary path. It tracks the last runtime-valid attempt separately from the retained candidate. File replacement requires positive IoU and FPR at or below both `RAILWAYS_MAX_FALSE_POSITIVE_RATE` (default `0.10`) and the retained FPR. IoU must improve, or stay equal while FPR decreases. Lower IoU is never accepted. A baseline failing the new ceiling is preserved until an eligible replacement is found. Generated source is compiled before evaluation.

Each attempt reports its retention decision and reason. The next prompt receives per-image diagnostics and the acceptance ceiling of the retained candidate, even when searching from an archive parent. Empty-detector feedback requests recovery of evidence-supported railways on positive examples while preserving rejection on negatives; unconditional fallback detections are forbidden. Target-score completion requires an accepted candidate. Repeated non-improvements require a materially different strategy rather than another equivalent threshold tweak.

The terminal now prints evaluator `SUMMARY`, `METRICS`, and `DIAGNOSIS` lines after each candidate score. After three attempts without a new dataset-IoU best by default (`RAILWAYS_MAX_STAGNANT_ITERATIONS`), the loop enters the disruption controller described below instead of stopping immediately.

The initial zero-score detector was traced to an OpenCV-version mismatch: OpenCV 5 returns `HoughLinesP` output as `(N, 4)`, while the generated candidate assumed `(N, 1, 4)`. The resulting scalar-slice exception was hidden by `except Exception: return []`. The retained candidate now normalizes either representation with `np.asarray(lines).reshape(-1, 4)` and re-raises unexpected errors. The generator prompt explicitly requires both behaviors.

### 8. Align runtime enforcement

Use one processing-time limit everywhere. Full-resolution processing is now required; the earlier 500 ms prompt target and instruction to downsample were removed. The evaluator still warns above 1 second and the overall evaluation timeout is 60 seconds.

Decide whether runtime is a hard eligibility condition or a reported secondary metric, then enforce it consistently. Measure image decoding consistently so candidates cannot appear faster or slower solely because of duplicated loading.

### 9. Parallel evaluator — completed

Images are evaluated in a spawn-based process pool. The default worker count is the smaller of four workers and the available CPU count minus one; `RAILWAYS_EVAL_WORKERS` can reduce it, with a hard cap of four to control memory use. OpenCV internal threading and OpenCL are disabled in workers to avoid nested parallelism.

Each worker imports the candidate once and processes multiple images. The parent applies a hard pool deadline, terminates workers on failure, preserves deterministic score ordering, and reports total wall time plus the slowest image. `RAILWAYS_EVAL_TIMEOUT` controls the internal deadline and defaults to 55 seconds, finishing before the orchestrator's 60-second subprocess timeout.

The evaluator no longer fully decodes each image merely to obtain its dimensions. It reads dimensions from the PNG header; candidate code remains responsible for the one actual image decode.

Full decoded-image preloading is intentionally not implemented. With the current path-based API, candidate code would still call `cv2.imread`, while preloading all 19 RGBA 4K images would consume roughly 630 MiB without removing that decode. Keeping images across optimization iterations would also require a persistent evaluator service because the orchestrator currently launches a new evaluator process for every iteration.

On the candidate present during implementation, sequential evaluation took 11.24 seconds end to end. The four-worker evaluator produced the identical `0.0128` score in 3.46 seconds end to end, with 2.31 seconds spent inside the process pool.

Codex generation is bounded separately by `RAILWAYS_CODEX_TIMEOUT`, now defaulting to 600 seconds for native-resolution image input. Architecture calls also default to 600 seconds. Each call uses one balanced bundle of up to four examples (original plus annotated copy), with parent diagnostics replacing static references. Image count, file size, model, and deadline are printed before generation. Codex runs in its own process group; a timeout or `Ctrl-C` terminates the CLI and its descendants. Timeouts preserve the last evaluated candidate and report the last available CLI output.

The loop ignores the user's global Codex model profile and defaults to `gpt-5.6-luna` with low reasoning in an empty read-only workspace. This prevents a global high-reasoning agent profile from turning a code-only generation step into a long repository task. A live test using the actual 1,070-line candidate completed in 47.4 seconds and returned a 200-line candidate without modifying project files.

### 10. Disrupt search plateaus — completed

The first useful detector plateaued at dataset IoU `0.084260`. It emitted a polygon on every positive and every negative image, but aggregate diagnostics showed only `0.103480` positive-corridor coverage, `0.312075` polygon precision, and a `1.0` false-positive rate. The retained implementation chooses one extended straight segment pair, while the annotations contain curved and partly occluded tracks. Repeating local threshold edits cannot repair that representation mismatch reliably.

The search controller now separates exploitation from deliberate divergence:

- ordinary turns repair the latest runtime-valid working implementation;
- three attempts without a new final-score best trigger one bounded architecture turn by default;
- the architecture turn plans four materially different branches: a segment graph with smooth piecewise curves, corridor segmentation with boundary recovery, multiscale path search, and piecewise repair of the LSD approach;
- branch implementations start independently from an appropriate archive parent rather than repeatedly mutating the last regression;
- after all four branches, local work resumes from the positive-IoU leader.

The default routing uses `gpt-5.6-sol` with medium reasoning for the infrequent open-ended architecture decision and `gpt-5.6-luna` with low reasoning for bounded code-only implementation turns. Each call has its own timeout. The architecture call falls back to the static branch definitions if it fails or times out.

An in-memory Pareto archive retains up to four non-dominated candidates using dataset IoU, mean positive-image IoU, and false-positive rate. It can preserve search parents that fail the file-acceptance gate. Archive membership never bypasses that gate or triggers target completion.

The evaluator also produces per-image coverage, polygon precision, predicted-to-ground-truth area ratio, and normalized centroid offset. It exports selected native-resolution originals and annotated PNGs through a JSON manifest; thumbnail montages and combined-image resizing were removed. The primary IoU score is unchanged; FPR additionally gates file acceptance, while the remaining signals guide search.

Relevant controls:

- `RAILWAYS_MAX_STAGNANT_ITERATIONS` — local attempts before disruption, default `3`;
- `RAILWAYS_MAX_DISRUPTION_ROUNDS` — architecture resets, default `1`;
- `RAILWAYS_PARETO_ARCHIVE_SIZE` — maximum in-memory alternatives, default `4`;
- `RAILWAYS_ARCHITECT_MODEL`, `RAILWAYS_ARCHITECT_REASONING_EFFORT`, and `RAILWAYS_ARCHITECT_TIMEOUT` — architecture-call routing;
- `RAILWAYS_CODEX_MODEL`, `RAILWAYS_CODEX_REASONING_EFFORT`, and `RAILWAYS_CODEX_TIMEOUT` — implementation-call routing.

## Phase 3 — Evaluation integrity

The loop repeatedly optimizes against the same dataset, so its score is a tuning score rather than an unbiased validation result. Candidate code can also access adjacent JSON annotations unless execution is isolated.

Before treating results as benchmark quality:

- split data into tuning and held-out test sets;
- keep held-out annotations inaccessible to candidate code;
- do not feed held-out per-image results back into the optimization loop;
- version the dataset and metric configuration;
- report tuning and held-out scores separately.

## Proposed implementation order

1. ~~Confirm and implement the paired-rail track contract.~~ Completed.
2. Add annotation-file validation that respects the missing-JSON convention.
3. ~~Replace line-mask IoU with filled track-polygon IoU.~~ Completed.
4. Finish dependency documentation and preflight checks. Location-independent paths are complete.
5. ~~Preserve the best candidate.~~ Completed. Align runtime enforcement.
6. Add a held-out evaluation split if the score will be used beyond this experiment.

## Go-live gate

The task is ready to run only when all of the following are true:

- the paired-rail output contract remains documented and enforced end to end;
- annotated images pass structural validation and missing-JSON negatives are reported;
- the dataset validator passes cleanly;
- polygon-IoU behavior and the target score are calibrated;
- evaluator contract tests pass;
- evaluation works from a clean, documented environment;
- the orchestrator starts with the railway task and preserves its best result;
- a smoke run completes without modifying annotations or reading ground truth from candidate code.

## Current evidence snapshot

- 19 distinct 3840×2160 PNG images.
- 18 structurally valid JSON files.
- 16 annotated positives, 2 explicit negatives, and 1 conventional missing-JSON negative.
- 509 annotation points are numeric and within declared image bounds.
- Python source parses successfully.
- Exact annotation predictions produce dataset polygon IoU `1.000000`.
- Empty predictions produce `0.000000`, not the former false baseline of `3 / 19 = 0.1579` from rewarding correct negative images.
- Exact annotations plus a synthetic false-positive polygon on every negative image score `0.979177`; omitting one annotated positive scores `0.924101`.
- Evaluation works from the project root with location-independent paths.
- After the first successful disruption improvement and a scale-aware size guard, the retained detector scores dataset IoU `0.133153`, mean positive-image IoU `0.218662`, positive coverage `0.428306`, polygon precision `0.161933`, and false-positive rate `0.666667`.
- Detections whose shorter rail or corridor span is under 5% of the full image diagonal are rejected as line-like clutter. This removed a 56–59 px false positive from a 3840×2160 negative image without removing any current positive prediction; the shortest valid annotation is about 458 px.
- It emits predictions on all 16 positives and 2 of 3 negatives. The remaining disruption target is therefore corridor localization and stronger negative rejection, not nominal detection count.
- A deterministic controller test originally verified strict score-based retention through architecture escalation. Current acceptance regression tests additionally cover the FPR ceiling, non-regression, equal-IoU FPR improvements, empty-detector rejection, baseline preservation, and target completion without generation calls.
