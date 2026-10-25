#!/usr/bin/env python3
"""Benchmark a candidate under live-worker conditions and guard its outputs.

Runs single-threaded OpenCV by default (as the live worker does), reports the
per-stage timing distribution and the candidate's call counters, and can write
or check a golden-output file so speed-only refactors stay output-identical.

    .venv/bin/python -B railways/bench_fast.py --limit 20
    .venv/bin/python -B railways/bench_fast.py --write-golden railways/runs/golden_fast.json
    .venv/bin/python -B railways/bench_fast.py --check-golden railways/runs/golden_fast.json
"""

from __future__ import annotations

import argparse
import cProfile
from collections import defaultdict
import json
from pathlib import Path
import pstats
import sys
import time

import cv2
import numpy as np

if __package__:
    from .evaluator import load_candidate_module
else:
    from evaluator import load_candidate_module


ROOT = Path(__file__).resolve().parent
DEFAULT_CANDIDATE = ROOT / "candidate_fast.py"
DEFAULT_INPUTS = (ROOT / "val_dataset", ROOT / "ood_dataset", ROOT / "images")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".avif", ".webp"}


def collect_images(inputs) -> list[Path]:
    images = []
    for item in map(Path, inputs):
        if item.is_dir():
            images.extend(sorted(
                path for path in item.rglob("*")
                if path.suffix.lower() in IMAGE_SUFFIXES
            ))
        elif item.is_file():
            images.append(item)
    return images


def image_key(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def compare_outputs(expected, actual, tolerance=1e-3) -> list[str]:
    """Describe differences between two detection lists (empty when equivalent)."""
    if len(expected) != len(actual):
        return [f"detection count {len(expected)} != {len(actual)}"]
    problems = []
    for index, (want, got) in enumerate(zip(expected, actual)):
        for rail in ("left_rail", "right_rail"):
            a = np.asarray(want[rail], float)
            b = np.asarray(got[rail], float)
            if a.shape != b.shape:
                problems.append(f"#{index} {rail} shape {a.shape} != {b.shape}")
            elif a.size and float(np.abs(a - b).max()) > tolerance:
                problems.append(
                    f"#{index} {rail} max diff {float(np.abs(a - b).max()):.4g} px"
                )
        if abs(float(want["confidence"]) - float(got["confidence"])) > 1e-6:
            problems.append(
                f"#{index} confidence {want['confidence']} != {got['confidence']}"
            )
    return problems


def run(candidate, images, use_frame=False):
    """Yield (path, detections, timings, counts, elapsed) per image."""
    process_image = candidate.process_image
    process_frame = getattr(candidate, "process_frame", None)
    if use_frame and not callable(process_frame):
        raise ValueError("--frame needs a candidate that exports process_frame.")
    for path in images:
        if use_frame:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            started = time.perf_counter()
            detections = process_frame(frame)
            elapsed = time.perf_counter() - started
            stats = process_frame
        else:
            started = time.perf_counter()
            detections = process_image(str(path))
            elapsed = time.perf_counter() - started
            stats = process_image
        yield (path, detections, dict(getattr(stats, "last_timings", {})),
               dict(getattr(stats, "last_counts", {})), elapsed)


def report(rows):
    def line(name, values):
        values = np.asarray(values, float)
        return (f"  {name:<22} mean {values.mean():7.3f}  p50 "
                f"{np.percentile(values, 50):7.3f}  p95 {np.percentile(values, 95):7.3f}")

    stages = defaultdict(list)
    counts = defaultdict(list)
    for row in rows:
        for name, value in row["timings"].items():
            stages[name].append(value)
        for name, value in row["counts"].items():
            counts[name].append(value)
    print(f"\n{len(rows)} images; seconds per image")
    print(line("elapsed", [row["elapsed"] for row in rows]))
    for name, values in stages.items():
        if name != "total":
            print(line(name, values))
    exclusive = [
        sum(v for k, v in row["timings"].items() if k != "total") for row in rows
    ]
    if any(exclusive):
        print(line("sum of stages", exclusive))
    for subset, label in ((True, "with detections"), (False, "empty")):
        chosen = [row["elapsed"] for row in rows if bool(row["detections"]) == subset]
        if chosen:
            print(line(f"elapsed ({label}, n={len(chosen)})", chosen))
    if counts:
        print("\ncalls per image (mean)")
        for name in sorted(counts):
            print(f"  {name:<34} {np.mean(counts[name]):9.1f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("-c", "--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--limit", type=int, default=0, help="first N images only")
    parser.add_argument("--threads", type=int, default=1,
                        help="cv2.setNumThreads value (live worker uses 1)")
    parser.add_argument("--frame", action="store_true",
                        help="time process_frame on pre-decoded pixels (live path)")
    parser.add_argument("--profile", action="store_true", help="print cProfile hotspots")
    parser.add_argument("--write-golden", type=Path)
    parser.add_argument("--check-golden", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args(argv)

    cv2.setNumThreads(args.threads)
    candidate = load_candidate_module(str(args.candidate.resolve()))
    images = collect_images(args.inputs)
    if args.limit:
        images = images[:args.limit]
    if not images:
        parser.error("no images found")
    golden = json.loads(args.check_golden.read_text()) if args.check_golden else None

    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    rows, failures = [], []
    for path, detections, timings, counts, elapsed in run(candidate, images, args.frame):
        key = image_key(path)
        rows.append(dict(key=key, detections=detections, timings=timings,
                         counts=counts, elapsed=elapsed))
        if golden is not None:
            if key not in golden:
                failures.append(f"{key}: missing from golden")
            else:
                failures.extend(
                    f"{key}: {problem}"
                    for problem in compare_outputs(golden[key], detections, args.tolerance)
                )
    if profiler:
        profiler.disable()

    report(rows)
    if profiler:
        print()
        pstats.Stats(profiler).sort_stats("tottime").print_stats(25)
    if args.write_golden:
        args.write_golden.parent.mkdir(parents=True, exist_ok=True)
        args.write_golden.write_text(json.dumps(
            {row["key"]: row["detections"] for row in rows}, indent=1
        ))
        print(f"\nwrote golden outputs for {len(rows)} images to {args.write_golden}")
    if golden is not None:
        print(f"\ngolden check: {len(rows) - len({f.split(':')[0] for f in failures})}"
              f"/{len(rows)} images identical")
        for failure in failures:
            print("  MISMATCH", failure)
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
