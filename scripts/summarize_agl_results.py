#!/usr/bin/env python3
"""List the minimum and maximum AGL height in each batch result file."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import numbers
from pathlib import Path
import sys
from typing import TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "logs" / "agl_results"


def discover_result_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.name.lower().endswith("_agl.json")
    )


def load_height_samples(path: Path) -> list[tuple[int, float]]:
    """Load (second, height) pairs from full-record or values-only JSON."""
    with path.open(encoding="utf-8") as result_file:
        data = json.load(result_file)

    if not isinstance(data, list):
        raise ValueError("top-level JSON value must be a list")
    if not data:
        raise ValueError("result contains no AGL samples")

    samples = []
    for index, sample in enumerate(data):
        if isinstance(sample, dict):
            if "height_above_terrain_m" not in sample:
                raise ValueError(
                    f"sample {index} has no height_above_terrain_m field"
                )
            value = sample["height_above_terrain_m"]
            second = sample.get("second")
        else:
            value = sample
            second = index

        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                f"sample {index} height_above_terrain_m is not a finite number"
            )
        if (
            isinstance(second, bool)
            or not isinstance(second, numbers.Real)
            or not math.isfinite(float(second))
            or not float(second).is_integer()
            or float(second) < 0
        ):
            raise ValueError(f"sample {index} second is not a non-negative integer")
        samples.append((int(second), float(value)))

    return samples


def summarize_file(path: Path) -> dict[str, object]:
    samples = load_height_samples(path)
    min_second, min_height = min(samples, key=lambda item: (item[1], item[0]))
    max_second, max_height = min(samples, key=lambda item: (-item[1], item[0]))
    return {
        "file": path.name,
        "samples": len(samples),
        "min_height_above_terrain_m": min_height,
        "min_second": min_second,
        "max_height_above_terrain_m": max_height,
        "max_second": max_second,
    }


def write_table(rows: list[dict[str, object]], output: TextIO) -> None:
    headers = (
        "file",
        "samples",
        "min_height_above_terrain_m",
        "min_second",
        "max_height_above_terrain_m",
        "max_second",
    )
    formatted = [
        {
            "file": str(row["file"]),
            "samples": str(row["samples"]),
            "min_height_above_terrain_m": (
                f"{float(row['min_height_above_terrain_m']):.3f}"
            ),
            "min_second": str(row["min_second"]),
            "max_height_above_terrain_m": (
                f"{float(row['max_height_above_terrain_m']):.3f}"
            ),
            "max_second": str(row["max_second"]),
        }
        for row in rows
    ]
    widths = {
        header: max(len(header), *(len(row[header]) for row in formatted))
        for header in headers
    }
    output.write("  ".join(header.ljust(widths[header]) for header in headers))
    output.write("\n")
    output.write("  ".join("-" * widths[header] for header in headers))
    output.write("\n")
    for row in formatted:
        output.write(
            f"{row['file']:<{widths['file']}}  "
            f"{row['samples']:>{widths['samples']}}  "
            f"{row['min_height_above_terrain_m']:>{widths['min_height_above_terrain_m']}}  "
            f"{row['min_second']:>{widths['min_second']}}  "
            f"{row['max_height_above_terrain_m']:>{widths['max_height_above_terrain_m']}}  "
            f"{row['max_second']:>{widths['max_second']}}\n"
        )


def render(rows: list[dict[str, object]], output_format: str) -> str:
    output = io.StringIO()
    if output_format == "json":
        json.dump(rows, output, indent=2)
        output.write("\n")
    elif output_format == "csv":
        fieldnames = [
            "file",
            "samples",
            "min_height_above_terrain_m",
            "min_second",
            "max_height_above_terrain_m",
            "max_second",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    else:
        write_table(rows, output)
    return output.getvalue()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "List each AGL result file with its minimum and maximum "
            "height_above_terrain_m."
        )
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"AGL result directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json", "csv"),
        default="table",
        help="summary format (default: table)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write the summary to this file instead of stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.directory.is_dir():
        raise ValueError(f"Not a directory: {args.directory}")

    result_files = discover_result_files(args.directory.resolve())
    if not result_files:
        print(f"No *_agl.json files found in '{args.directory}'.", file=sys.stderr)
        return 1

    summaries = []
    failures = 0
    for result_path in result_files:
        try:
            summaries.append(summarize_file(result_path))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            failures += 1
            print(f"SKIP {result_path.name}: {error}", file=sys.stderr)

    if not summaries:
        print("No valid AGL result files found.", file=sys.stderr)
        return 1

    summaries.sort(
        key=lambda row: (
            float(row["min_height_above_terrain_m"]),
            str(row["file"]),
        )
    )
    rendered = render(summaries, args.format)
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(output_path)

    if failures:
        print(
            f"Summary contains {len(summaries)} file(s); {failures} invalid "
            "file(s) were skipped.",
            file=sys.stderr,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
