#!/usr/bin/env python3
"""Estimate drone height above terrain once per second from a DJI SRT file.

The default calculation is:

    AGL = DJI absolute altitude - terrain elevation

Terrain elevation is queried from the public OpenTopoData SRTM 30 m service.
If the takeoff/home coordinate is known, pass --home-lat and --home-lon to use
the more robust takeoff-calibrated calculation:

    AGL = DJI relative altitude + terrain(home) - terrain(current)

The script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import bisect
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import ssl
import sys
import time
from typing import Iterable, TextIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_API_URL = "https://api.opentopodata.org/v1/srtm30m"
MAX_LOCATIONS_PER_REQUEST = 100

CUE_RE = re.compile(
    r"(?ms)^\s*(?P<cue>\d+)\s*\r?\n"
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})\s*\r?\n"
    r"(?P<body>.*?)(?=^\s*\d+\s*\r?\n"
    r"\d{2}:\d{2}:\d{2},\d{3}\s*-->|\Z)"
)


@dataclass(frozen=True)
class TelemetryRecord:
    video_time_s: float
    frame: int
    timestamp: str
    latitude: float
    longitude: float
    relative_altitude_m: float
    absolute_altitude_m: float


@dataclass(frozen=True)
class AglSample:
    second: int
    frame: int
    timestamp: str
    latitude: float
    longitude: float
    relative_altitude_m: float
    absolute_altitude_m: float
    terrain_elevation_m: float
    height_above_terrain_m: float


def _srt_time_seconds(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, milliseconds = rest.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds) / 1000.0
    )


def _required(pattern: str, body: str, label: str, cue: int) -> str:
    match = re.search(pattern, body)
    if match is None:
        raise ValueError(f"SRT cue {cue} is missing {label}")
    return match.group(1).strip()


def parse_dji_srt(path: Path) -> list[TelemetryRecord]:
    text = path.read_text(encoding="utf-8-sig")
    records: list[TelemetryRecord] = []

    for match in CUE_RE.finditer(text):
        cue = int(match.group("cue"))
        body = match.group("body")
        records.append(
            TelemetryRecord(
                video_time_s=_srt_time_seconds(match.group("start")),
                frame=int(_required(r"FrameCnt:\s*(\d+)", body, "FrameCnt", cue)),
                timestamp=_required(
                    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)",
                    body,
                    "timestamp",
                    cue,
                ),
                latitude=float(
                    _required(r"\[latitude:\s*([-+\d.]+)\]", body, "latitude", cue)
                ),
                longitude=float(
                    _required(
                        r"\[longitude:\s*([-+\d.]+)\]", body, "longitude", cue
                    )
                ),
                relative_altitude_m=float(
                    _required(r"\[rel_alt:\s*([-+\d.]+)", body, "rel_alt", cue)
                ),
                absolute_altitude_m=float(
                    _required(r"abs_alt:\s*([-+\d.]+)\]", body, "abs_alt", cue)
                ),
            )
        )

    if not records:
        raise ValueError(f"No DJI telemetry cues found in {path}")

    for previous, current in zip(records, records[1:]):
        if current.video_time_s < previous.video_time_s:
            raise ValueError("SRT cue timestamps are not monotonically increasing")

    return records


def sample_each_second(records: list[TelemetryRecord]) -> list[TelemetryRecord]:
    """Return the telemetry record nearest each integer video second."""
    times = [record.video_time_s for record in records]
    final_second = int(math.floor(times[-1]))
    sampled: list[TelemetryRecord] = []

    for second in range(final_second + 1):
        right = bisect.bisect_left(times, float(second))
        candidates = []
        if right < len(records):
            candidates.append(records[right])
        if right > 0:
            candidates.append(records[right - 1])
        sampled.append(
            min(candidates, key=lambda record: abs(record.video_time_s - second))
        )

    return sampled


def _chunks(
    values: list[tuple[float, float]], size: int
) -> Iterable[list[tuple[float, float]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _post_json(
    url: str, payload: dict[str, object], timeout_s: float
) -> dict[str, object]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "srt-agl/1.0",
        },
        method="POST",
    )

    ssl_context = _ssl_context()
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=timeout_s, context=ssl_context) as response:
                result = json.loads(response.read().decode("utf-8"))
            if not isinstance(result, dict):
                raise RuntimeError("Terrain service returned a non-object response")
            return result
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(2**attempt)

    raise RuntimeError(f"Terrain service request failed: {last_error}")


def _ssl_context() -> ssl.SSLContext:
    """Build a verified TLS context, including common macOS Python CA paths."""
    default_paths = ssl.get_default_verify_paths()
    candidates = [
        os.environ.get("SSL_CERT_FILE"),
        default_paths.cafile,
        default_paths.openssl_cafile,
        "/etc/ssl/cert.pem",
        "/opt/homebrew/etc/ca-certificates/cert.pem",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return ssl.create_default_context(cafile=candidate)
    return ssl.create_default_context()


def fetch_terrain_elevations(
    coordinates: list[tuple[float, float]],
    *,
    api_url: str,
    timeout_s: float,
) -> list[float]:
    elevations: list[float] = []

    batches = list(_chunks(coordinates, MAX_LOCATIONS_PER_REQUEST))
    for batch_index, batch in enumerate(batches):
        if batch_index:
            # The public OpenTopoData endpoint permits one request per second.
            time.sleep(1.05)
        payload = {
            "locations": "|".join(f"{lat:.7f},{lon:.7f}" for lat, lon in batch),
            "interpolation": "bilinear",
        }
        response = _post_json(api_url, payload, timeout_s)
        if response.get("status") != "OK":
            raise RuntimeError(
                f"Terrain service error: {response.get('error') or response}"
            )

        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(batch):
            raise RuntimeError(
                "Terrain service returned a different number of elevations "
                f"({len(results) if isinstance(results, list) else 'invalid'}) "
                f"than requested ({len(batch)})"
            )

        for index, item in enumerate(results):
            if not isinstance(item, dict) or item.get("elevation") is None:
                raise RuntimeError(
                    f"No terrain elevation for coordinate {batch[index]}"
                )
            elevations.append(float(item["elevation"]))

    return elevations


def calculate_agl_samples(
    records: list[TelemetryRecord],
    terrain_elevations: list[float],
    *,
    home_terrain_elevation_m: float | None,
) -> list[AglSample]:
    if len(records) != len(terrain_elevations):
        raise ValueError("Telemetry and terrain sample counts do not match")

    samples = []
    for second, (record, terrain_m) in enumerate(
        zip(records, terrain_elevations)
    ):
        if home_terrain_elevation_m is None:
            agl_m = record.absolute_altitude_m - terrain_m
        else:
            agl_m = (
                record.relative_altitude_m
                + home_terrain_elevation_m
                - terrain_m
            )

        samples.append(
            AglSample(
                second=second,
                frame=record.frame,
                timestamp=record.timestamp,
                latitude=record.latitude,
                longitude=record.longitude,
                relative_altitude_m=record.relative_altitude_m,
                absolute_altitude_m=record.absolute_altitude_m,
                terrain_elevation_m=round(terrain_m, 3),
                height_above_terrain_m=round(agl_m, 3),
            )
        )

    return samples


def _write_json(samples: list[AglSample], output: TextIO, values_only: bool) -> None:
    data: object
    if values_only:
        data = [sample.height_above_terrain_m for sample in samples]
    else:
        data = [asdict(sample) for sample in samples]
    json.dump(data, output, indent=2)
    output.write("\n")


def _write_csv(samples: list[AglSample], output: TextIO) -> None:
    fieldnames = list(asdict(samples[0]).keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(asdict(sample) for sample in samples)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Output estimated height above terrain once per second from a DJI SRT file."
        )
    )
    parser.add_argument("srt_file", type=Path, help="DJI .SRT telemetry file")
    parser.add_argument(
        "-o", "--output", type=Path, help="write output to this file instead of stdout"
    )
    parser.add_argument(
        "--format", choices=("json", "csv"), default="json", help="output format"
    )
    parser.add_argument(
        "--values-only",
        action="store_true",
        help="for JSON, output only the numeric AGL values",
    )
    parser.add_argument(
        "--home-lat",
        type=float,
        help="takeoff latitude; must be provided together with --home-lon",
    )
    parser.add_argument(
        "--home-lon",
        type=float,
        help="takeoff longitude; must be provided together with --home-lat",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"OpenTopoData-compatible terrain endpoint (default: {DEFAULT_API_URL})",
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

    if (args.home_lat is None) != (args.home_lon is None):
        raise ValueError("--home-lat and --home-lon must be supplied together")
    if args.values_only and args.format != "json":
        raise ValueError("--values-only is supported only with --format json")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")

    records = sample_each_second(parse_dji_srt(args.srt_file))
    coordinates = [(record.latitude, record.longitude) for record in records]

    home_terrain_m = None
    lookup_coordinates = list(coordinates)
    if args.home_lat is not None:
        lookup_coordinates.append((args.home_lat, args.home_lon))

    elevations = fetch_terrain_elevations(
        lookup_coordinates, api_url=args.api_url, timeout_s=args.timeout
    )
    if args.home_lat is not None:
        home_terrain_m = elevations.pop()

    samples = calculate_agl_samples(
        records, elevations, home_terrain_elevation_m=home_terrain_m
    )

    if args.output is None:
        output = sys.stdout
        if args.format == "json":
            _write_json(samples, output, args.values_only)
        else:
            _write_csv(samples, output)
    else:
        with args.output.open("w", encoding="utf-8", newline="") as output:
            if args.format == "json":
                _write_json(samples, output, args.values_only)
            else:
                _write_csv(samples, output)

    mode = (
        "takeoff-calibrated relative altitude"
        if home_terrain_m is not None
        else "DJI absolute altitude"
    )
    print(
        f"Wrote {len(samples)} one-second AGL samples using {mode} and SRTM 30 m terrain.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
