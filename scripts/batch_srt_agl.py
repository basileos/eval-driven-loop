#!/usr/bin/env python3
"""Run srt_agl.py for every compatible DJI SRT file in a directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

from srt_agl import DEFAULT_API_URL, parse_dji_srt


SINGLE_FILE_SCRIPT = Path(__file__).with_name("srt_agl.py")


def discover_srt_files(directory: Path, recursive: bool) -> list[Path]:
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() == ".srt"
    )


def result_path(
    srt_file: Path,
    *,
    input_directory: Path,
    output_directory: Path,
    recursive: bool,
    output_format: str,
) -> Path:
    relative_parent = (
        srt_file.relative_to(input_directory).parent if recursive else Path()
    )
    return (
        output_directory
        / relative_parent
        / f"{srt_file.stem}_agl.{output_format}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate one-second height-above-terrain results for every "
            "compatible DJI SRT file in a directory."
        )
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help="directory containing SRT files (legacy positional form)",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        help="directory containing SRT files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="result directory (default: DIRECTORY/agl_results)",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="scan subdirectories recursively"
    )
    parser.add_argument(
        "--format", choices=("json", "csv"), default="json", help="output format"
    )
    parser.add_argument(
        "--values-only",
        action="store_true",
        help="for JSON, save only the numeric AGL values",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing result files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and show result paths without terrain requests",
    )
    parser.add_argument(
        "--home-lat",
        type=float,
        help="takeoff latitude to apply to every compatible file",
    )
    parser.add_argument(
        "--home-lon",
        type=float,
        help="takeoff longitude to apply to every compatible file",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"terrain endpoint forwarded to srt_agl.py (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="terrain request timeout in seconds (default: 30)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.directory is not None and args.input_dir is not None:
        raise ValueError("use either positional DIRECTORY or --input-dir, not both")
    selected_directory = args.input_dir or args.directory
    if selected_directory is None:
        raise ValueError("an input directory is required; use --input-dir PATH")
    if not selected_directory.is_dir():
        raise ValueError(f"Not a directory: {selected_directory}")
    if (args.home_lat is None) != (args.home_lon is None):
        raise ValueError("--home-lat and --home-lon must be supplied together")
    if args.values_only and args.format != "json":
        raise ValueError("--values-only is supported only with --format json")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")

    input_directory = selected_directory.resolve()
    output_directory = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_directory / "agl_results"
    )
    srt_files = discover_srt_files(input_directory, args.recursive)
    if not srt_files:
        print(f"No .SRT files found in {input_directory}", file=sys.stderr)
        return 1

    compatible: list[tuple[Path, Path]] = []
    skipped_incompatible = 0
    for srt_file in srt_files:
        destination = result_path(
            srt_file,
            input_directory=input_directory,
            output_directory=output_directory,
            recursive=args.recursive,
            output_format=args.format,
        )
        try:
            parse_dji_srt(srt_file)
        except (OSError, UnicodeError, ValueError) as error:
            skipped_incompatible += 1
            print(f"SKIP incompatible: {srt_file} ({error})", file=sys.stderr)
            continue
        compatible.append((srt_file, destination))

    if not compatible:
        print(
            f"No compatible DJI SRT files found ({skipped_incompatible} skipped).",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        for source, destination in compatible:
            print(f"{source} -> {destination}")
        print(
            f"Dry run: {len(compatible)} compatible, "
            f"{skipped_incompatible} incompatible.",
            file=sys.stderr,
        )
        return 0

    processed = 0
    skipped_existing = 0
    failed = 0
    for file_index, (source, destination) in enumerate(compatible):
        if destination.exists() and not args.overwrite:
            skipped_existing += 1
            print(f"SKIP existing: {destination}", file=sys.stderr)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-B",
            str(SINGLE_FILE_SCRIPT),
            str(source),
            "--output",
            str(destination),
            "--format",
            args.format,
            "--api-url",
            args.api_url,
            "--timeout",
            str(args.timeout),
        ]
        if args.values_only:
            command.append("--values-only")
        if args.home_lat is not None:
            command.extend(
                ["--home-lat", str(args.home_lat), "--home-lon", str(args.home_lon)]
            )

        print(f"PROCESS: {source} -> {destination}", file=sys.stderr)
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            processed += 1
        else:
            failed += 1
            print(
                f"FAILED ({result.returncode}): {source}",
                file=sys.stderr,
            )

        # The public OpenTopoData service permits at most one request per second.
        if file_index < len(compatible) - 1:
            time.sleep(1.05)

    print(
        "Batch complete: "
        f"{processed} processed, {skipped_existing} existing, "
        f"{skipped_incompatible} incompatible, {failed} failed. "
        f"Results: {output_directory}",
        file=sys.stderr,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
