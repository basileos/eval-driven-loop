#!/usr/bin/env python3
"""Run a railway candidate on every image in a directory.

This is an inference/debugging wrapper, not an evaluator: it does not require
ground-truth JSON files and does not calculate IoU.  It writes full-resolution
candidate overlays and a machine-readable metrics/predictions manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
from pathlib import Path
import sys
import time
from typing import Callable

import cv2
import numpy as np

if __package__:
    from .evaluator import (
        _draw_track_overlay,
        calculate_prediction_size_metrics,
        load_candidate_module,
        read_image_shape,
        validate_predicted_tracks,
    )
else:
    from evaluator import (
        _draw_track_overlay,
        calculate_prediction_size_metrics,
        load_candidate_module,
        read_image_shape,
        validate_predicted_tracks,
    )


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CANDIDATE_PATH = BASE_DIR / "candidate_script.py"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
Track = dict[str, object]
ProcessImage = Callable[[str], object]


def attach_confidence(raw_tracks: list[dict], tracks: list[Track]) -> None:
    """Preserve optional candidate scores after geometry-only validation.

    Never infer confidence from track length or assign a default score. The
    evaluator remains geometry-only and ignores this inference metadata.
    """
    scores = []
    for index, raw in enumerate(raw_tracks):
        score = raw.get("confidence")
        method = raw.get("confidence_method", "candidate_reported")
        if score is not None:
            if (isinstance(score, bool) or not isinstance(score, numbers.Real)
                    or not math.isfinite(float(score)) or not 0 <= score <= 1):
                raise ValueError(f"Track {index} confidence must be a finite number from 0 to 1.")
            if not isinstance(method, str) or not method.strip():
                raise ValueError(f"Track {index} confidence_method must be a non-empty string.")
        scores.append((float(score) if score is not None else None, method))
    for track, (score, method) in zip(tracks, scores):
        track["confidence"] = score
        track["confidence_method"] = method if score is not None else None


def confidence_label(track: Track, index: int) -> str:
    score = track.get("confidence")
    value = "n/a" if score is None else f"{float(score):.0%}"
    return f"T{index + 1} confidence={value} (uncalibrated)"


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def discover_images(
    input_directory: Path,
    *,
    output_directory: Path,
    recursive: bool,
) -> list[Path]:
    """Find supported input images without re-reading generated diagnostics."""
    candidates = (
        input_directory.rglob("*") if recursive else input_directory.iterdir()
    )
    resolved_output = output_directory.resolve()
    images = []

    for path in candidates:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem.endswith("__debug"):
            continue
        if _is_within(path.resolve(), resolved_output):
            continue
        images.append(path)

    return sorted(images, key=lambda path: path.relative_to(input_directory).as_posix())


def diagnostic_path(
    image_path: Path,
    *,
    input_directory: Path,
    output_directory: Path,
    recursive: bool,
) -> Path:
    relative_parent = (
        image_path.relative_to(input_directory).parent if recursive else Path()
    )
    return output_directory / relative_parent / f"{image_path.stem}__debug.jpg"


def write_debug_image(
    image_path: Path,
    output_path: Path,
    *,
    display_name: str,
    tracks: list[Track],
    elapsed_seconds: float,
    size_metrics: dict[str, object],
    error: str | None,
    image: np.ndarray | None = None,
    status_note: str | None = None,
) -> None:
    """Render one full-resolution candidate overlay in evaluator debug style."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image is None else image
    if image is None or image.size == 0:
        raise ValueError(f"Could not read image '{image_path}'.")
    canvas = render_debug_frame(
        image, display_name=display_name, tracks=tracks, elapsed_seconds=elapsed_seconds,
        size_metrics=size_metrics, error=error, status_note=status_note,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Could not write debug image '{output_path}'.")


def render_debug_frame(
    image: np.ndarray, *, display_name: str, tracks: list[Track],
    elapsed_seconds: float, size_metrics: dict[str, object], error: str | None,
    status_note: str | None = None,
) -> np.ndarray:
    """Shared image/video renderer; leaves the native-resolution input unchanged."""
    image = image.copy()

    height, width = image.shape[:2]
    line_thickness = max(3, int(round(width / 900)))
    _draw_track_overlay(
        image,
        tracks,
        1.0,
        1.0,
        fill_color=(70, 255, 70),
        left_color=(255, 255, 0),
        right_color=(255, 0, 255),
        alpha=0.20,
        line_thickness=line_thickness,
    )
    for index, track in enumerate(tracks):
        # Label the track itself, keeping many scores out of the header.
        points = np.asarray(track["left_rail"], dtype=float)
        anchor = points[len(points) // 2]
        label = confidence_label(track, index)
        label_scale = max(0.35, min(1.0, width / 2000))
        label_thickness = max(1, int(round(label_scale * 2)))
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thickness
        )
        x = int(np.clip(anchor[0], 0, max(0, width - text_width - 8)))
        y = int(np.clip(anchor[1], text_height + 8, max(text_height + 8, height - baseline - 4)))
        cv2.rectangle(image, (x, y - text_height - 5),
                      (x + text_width + 8, y + baseline + 3), (0, 0, 0), -1)
        cv2.putText(image, label, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX,
                    label_scale, (255, 255, 255), label_thickness, cv2.LINE_AA)

    minimum_length = size_metrics.get("minimum_predicted_rail_length_px")
    minimum_ratio = size_metrics.get("minimum_predicted_rail_length_ratio")
    if minimum_length is None or minimum_ratio is None:
        rail_text = "n/a"
    else:
        rail_text = f"{float(minimum_length):.1f}px ({float(minimum_ratio):.2%})"

    header_height = max(110, int(round(height * 0.055)))
    header = np.zeros((header_height, width, 3), dtype=np.uint8)
    font_scale = max(0.65, min(1.5, width / 2600))
    text_thickness = max(1, line_thickness // 2)
    status = f"ERROR: {error}" if error else "status=ok"
    if status_note:
        status += f" | {status_note}"
    lines = [
        f"{display_name} | predicted tracks={len(tracks)} | {status}",
        (
            f"elapsed={elapsed_seconds:.3f}s | minimum predicted rail={rail_text} | "
            "Candidate: cyan/magenta rails + green polygon"
        ),
    ]
    baseline_step = header_height // 3
    for index, line in enumerate(lines, start=1):
        cv2.putText(
            header,
            line,
            (18, baseline_step * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return np.vstack((header, image))


def process_one_image(
    image_path: Path,
    *,
    input_directory: Path,
    output_directory: Path,
    recursive: bool,
    process_image: ProcessImage,
    frame: np.ndarray | None = None,
    process_frame: Callable[[np.ndarray], object] | None = None,
    save_overlay: bool = True,
    status_note: Callable[[], str] | None = None,
) -> dict[str, object]:
    relative_name = image_path.relative_to(input_directory).as_posix()
    output_path = diagnostic_path(
        image_path,
        input_directory=input_directory,
        output_directory=output_directory,
        recursive=recursive,
    )
    elapsed_seconds = 0.0
    tracks: list[Track] = []
    size_metrics: dict[str, object] = {
        "minimum_predicted_rail_length_px": None,
        "minimum_predicted_rail_length_ratio": None,
        "undersized_predicted_tracks": 0,
    }
    error = None

    try:
        image_shape = read_image_shape(str(image_path)) if frame is None else frame.shape[:2]
        started_at = time.perf_counter()
        try:
            raw_prediction = (
                process_frame(frame)
                if frame is not None and process_frame is not None
                else process_image(str(image_path))
            )
        finally:
            elapsed_seconds = time.perf_counter() - started_at
        tracks = validate_predicted_tracks(raw_prediction, image_shape)
        attach_confidence(raw_prediction, tracks)
        size_metrics = calculate_prediction_size_metrics(tracks, image_shape)
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"

    debug_image = None
    try:
        if save_overlay:
            write_debug_image(
                image_path,
                output_path,
                display_name=relative_name,
                tracks=tracks,
                elapsed_seconds=elapsed_seconds,
                size_metrics=size_metrics,
                error=error,
                image=frame,
                status_note=status_note() if status_note is not None else None,
            )
            debug_image = output_path.relative_to(output_directory).as_posix()
    except Exception as caught:
        render_error = f"{type(caught).__name__}: {caught}"
        error = f"{error}; {render_error}" if error else render_error

    return {
        "image": relative_name,
        "predicted_tracks": len(tracks),
        "predictions": tracks,
        "elapsed_seconds": elapsed_seconds,
        **size_metrics,
        "debug_image": debug_image,
        "error": error,
    }


def load_process_image(candidate_path: Path) -> ProcessImage:
    candidate = load_candidate_module(str(candidate_path))
    process_image = getattr(candidate, "process_image", None)
    if not callable(process_image):
        raise AttributeError(
            f"Candidate '{candidate_path}' must export a callable process_image."
        )
    return process_image


def write_manifest(
    output_directory: Path,
    *,
    candidate_path: Path,
    input_directory: Path,
    results: list[dict[str, object]],
) -> Path:
    failures = sum(result["error"] is not None for result in results)
    detected_images = sum(int(result["predicted_tracks"]) > 0 for result in results)
    total_tracks = sum(int(result["predicted_tracks"]) for result in results)
    manifest = {
        "candidate": str(candidate_path),
        "input_directory": str(input_directory),
        "legend": {
            "candidate": "cyan/magenta rails with green polygon",
        },
        "summary": {
            "images": len(results),
            "successful_images": len(results) - failures,
            "failed_images": failures,
            "images_with_detections": detected_images,
            "predicted_tracks": total_tracks,
            "total_elapsed_seconds": sum(
                float(result["elapsed_seconds"]) for result in results
            ),
        },
        "images": results,
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "metrics.json"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a railway candidate on a directory of images and write "
            "debug overlays plus metrics.json."
        )
    )
    parser.add_argument("input_dir", type=Path, help="directory containing images")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="result directory (default: INPUT_DIR/candidate_results)",
    )
    parser.add_argument(
        "-c",
        "--candidate",
        type=Path,
        default=DEFAULT_CANDIDATE_PATH,
        help=f"candidate script (default: {DEFAULT_CANDIDATE_PATH})",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="scan subdirectories and preserve their layout in the results",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input_dir.is_dir():
        raise ValueError(f"Not a directory: {args.input_dir}")
    if not args.candidate.is_file():
        raise ValueError(f"Candidate script does not exist: {args.candidate}")

    input_directory = args.input_dir.resolve()
    candidate_path = args.candidate.resolve()
    output_directory = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_directory / "candidate_results"
    )
    images = discover_images(
        input_directory,
        output_directory=output_directory,
        recursive=args.recursive,
    )
    if not images:
        print(f"No supported images found in '{input_directory}'.", file=sys.stderr)
        return 1

    destinations: dict[Path, Path] = {}
    for image_path in images:
        destination = diagnostic_path(
            image_path,
            input_directory=input_directory,
            output_directory=output_directory,
            recursive=args.recursive,
        )
        if destination in destinations:
            raise ValueError(
                f"Output collision between '{destinations[destination]}' and "
                f"'{image_path}'. Rename one image or separate them into "
                "subdirectories and use --recursive."
            )
        destinations[destination] = image_path

    process_image = load_process_image(candidate_path)
    results = []
    for index, image_path in enumerate(images, start=1):
        relative_name = image_path.relative_to(input_directory).as_posix()
        print(f"PROCESS [{index}/{len(images)}]: {relative_name}", file=sys.stderr)
        result = process_one_image(
            image_path,
            input_directory=input_directory,
            output_directory=output_directory,
            recursive=args.recursive,
            process_image=process_image,
        )
        results.append(result)
        if result["error"] is not None:
            print(f"FAILED: {relative_name}: {result['error']}", file=sys.stderr)

    manifest_path = write_manifest(
        output_directory,
        candidate_path=candidate_path,
        input_directory=input_directory,
        results=results,
    )
    failures = sum(result["error"] is not None for result in results)
    detections = sum(int(result["predicted_tracks"]) for result in results)
    print(
        f"Batch complete: {len(results) - failures} succeeded, {failures} failed, "
        f"{detections} track(s) detected. Results: {output_directory}",
        file=sys.stderr,
    )
    print(manifest_path)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
