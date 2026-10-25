#!/usr/bin/env python3
"""Run a railway candidate on a video, stream URL, or camera."""

from __future__ import annotations

import argparse
import copy
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import json
import inspect
import math
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

import cv2

if __package__:
    from .run_candidate import DEFAULT_CANDIDATE_PATH, load_candidate_module, process_one_image
    from .streaming_tracker import StreamingTracker
    from .latest_frame import LatestFrameReader
else:
    from run_candidate import DEFAULT_CANDIDATE_PATH, load_candidate_module, process_one_image
    from streaming_tracker import StreamingTracker
    from latest_frame import LatestFrameReader


def load_candidate_functions(candidate_path):
    candidate = load_candidate_module(str(candidate_path))
    process_image = getattr(candidate, "process_image", None)
    if not callable(process_image):
        raise ValueError("Candidate must export process_image(path).")
    process_frame = getattr(candidate, "process_frame", None)
    return process_image, process_frame if callable(process_frame) else None


def initialize_processor(candidate_path, output_directory, save_overlay, ignore_sigint=False,
                         streaming_options=None):
    global _process_image, _process_frame, _output_directory, _save_overlay, _stream_tracker, _roi_reference_shape
    if ignore_sigint:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Avoid multiplying OpenCV thread pools by the number of worker processes.
    cv2.setNumThreads(1)
    _process_image, _process_frame = load_candidate_functions(candidate_path)
    _roi_reference_shape = _process_frame is not None and 'reference_shape' in inspect.signature(_process_frame).parameters
    _output_directory = output_directory
    _save_overlay = save_overlay
    _stream_tracker = StreamingTracker(
        detect_decoded_frame,
        detect_roi=(lambda pixels, shape: detect_decoded_frame(pixels, shape)) if _roi_reference_shape else None,
        **streaming_options,
    ) if streaming_options is not None else None


def detect_decoded_frame(frame, reference_shape=None):
    """Run the candidate detector on decoded pixels; path-only candidates get a temporary BMP."""
    if _process_frame is not None:
        if reference_shape is not None and _roi_reference_shape:
            return _process_frame(frame, reference_shape=reference_shape)
        return _process_frame(frame)
    with tempfile.TemporaryDirectory(prefix='railway-detection-') as temporary:
        path = Path(temporary) / 'frame.bmp'
        if not cv2.imwrite(str(path), frame):
            raise OSError('Could not encode detector input.')
        return _process_image(str(path))


def process_video_frame(frame, metadata):
    started_at = time.perf_counter()
    stem = f"frame_{metadata['frame_index']:08d}"
    processor = _process_frame
    if _stream_tracker is not None:
        processor = lambda pixels: _stream_tracker.process(pixels, metadata.get('timestamp_seconds'))
    # Legacy candidates still accept paths: uncompressed BMP avoids PNG encoding.
    context = tempfile.TemporaryDirectory(prefix="railway-video-") if processor is None else nullcontext(None)
    with context as temporary:
        input_directory = Path(temporary) if temporary else Path(".")
        image_path = input_directory / f"{stem}.bmp"
        if temporary and not cv2.imwrite(str(image_path), frame):
            raise OSError(f"Could not encode frame {metadata['frame_index']}.")
        result = process_one_image(
            image_path,
            input_directory=input_directory,
            output_directory=_output_directory,
            recursive=False,
            process_image=_process_image,
            frame=frame,
            process_frame=processor,
            save_overlay=_save_overlay,
            status_note=(lambda: 'mode=' + _stream_tracker.diagnostics['mode']) if _stream_tracker is not None else None,
        )
    result.pop("image")
    result.update(metadata)
    result.update({"width": int(frame.shape[1]), "height": int(frame.shape[0])})
    result["processing_seconds"] = time.perf_counter() - started_at
    if _stream_tracker is not None:
        result['streaming'] = dict(_stream_tracker.diagnostics)
        if result['error']:
            _stream_tracker.reset()
    if 'application_queue_seconds' in metadata:
        result['application_latency_seconds'] = metadata['application_queue_seconds'] + result['processing_seconds']
    return result


def read_frames(capture, args, is_file, fps, summary, started_at, first_frame=0):
    previous_media_ms = -1.0
    submitted = 0
    eligible = 0
    while args.max_frames is None or submitted < args.max_frames:
        # grab() skips conversion to BGR for frames excluded by the sampling interval.
        if not capture.grab():
            summary["stop_reason"] = "end_of_video" if is_file else "stream_read_failed"
            return
        received_seconds = time.perf_counter() - started_at
        frame_index = first_frame + int(summary["decoded_frames"])
        summary["decoded_frames"] += 1
        timestamp_seconds = received_seconds
        timestamp_source = "elapsed_monotonic"
        if is_file:
            media_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            if math.isfinite(media_ms) and media_ms >= 0 and media_ms > previous_media_ms:
                timestamp_seconds = media_ms / 1000.0
                timestamp_source = "video_position"
                previous_media_ms = media_ms
            else:
                timestamp_seconds = frame_index / fps if fps else None
                timestamp_source = "frame_index_over_fps" if fps else "unavailable"
        if timestamp_seconds is not None:
            if timestamp_seconds + 1e-9 < args.start_time:
                continue
            if args.end_time is not None and timestamp_seconds + 1e-9 >= args.end_time:
                summary["stop_reason"] = "end_time"
                return
        eligible += 1
        if (eligible - 1) % args.every_n_frames:
            continue
        ok, frame = capture.retrieve()
        if not ok or frame is None or frame.size == 0:
            summary["stop_reason"] = "frame_decode_failed"
            return
        submitted += 1
        yield frame, {
            "frame_index": frame_index,
            "timestamp_seconds": timestamp_seconds,
            "timestamp_source": timestamp_source,
        }
    summary["stop_reason"] = "max_frames"


def ordered_results(frames, pool, workers):
    if pool is None:
        for frame, metadata in frames:
            yield process_video_frame(frame, metadata)
        return
    # At most one outstanding frame per worker: do not queue an entire video in RAM.
    pending = deque()
    try:
        for _ in range(workers):
            item = next(frames, None)
            if item is None:
                break
            pending.append(pool.submit(process_video_frame, *item))
        while pending:
            yield pending.popleft().result()
            item = next(frames, None)
            if item is not None:
                pending.append(pool.submit(process_video_frame, *item))
    finally:
        for future in pending:
            future.cancel()


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be finite and greater than zero')
    return number


def timestamp(value: str) -> float:
    """Parse seconds, MM:SS, or HH:MM:SS (fractional seconds allowed)."""
    try:
        parts = value.split(":")
        if not 1 <= len(parts) <= 3:
            raise ValueError
        numbers = [float(part) for part in parts]
        if any(not math.isfinite(number) or number < 0 for number in numbers):
            raise ValueError
        if len(parts) > 1:
            if any(not part.isdecimal() for part in parts[:-1]):
                raise ValueError
            if any(number >= 60 for number in numbers[1:]):
                raise ValueError
        return sum(number * 60 ** index for index, number in enumerate(reversed(numbers)))
    except ValueError:
        raise argparse.ArgumentTypeError("use non-negative seconds, MM:SS, or HH:MM:SS") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="video file, stream URL, or camera index (e.g. 0)")
    parser.add_argument("-c", "--candidate", type=Path, default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument(
        "-o", "--output-dir", type=Path,
        help="empty result directory (default: VIDEO_STEM_candidate_results or video_candidate_results)",
    )
    parser.add_argument(
        "--every-n-frames", type=positive_int, default=1,
        help="process every Nth frame starting at the selected range's first frame (default: 1)",
    )
    parser.add_argument(
        "--start-time", type=timestamp, default=0.0,
        help="local video start, inclusive: seconds, MM:SS, or HH:MM:SS (default: 0)",
    )
    parser.add_argument(
        "--end-time", type=timestamp,
        help="local video end, exclusive, in the same formats (default: EOF)",
    )
    parser.add_argument(
        "--max-frames", type=positive_int,
        help="stop after processing this many frames (default: until EOF or Ctrl-C)",
    )
    parser.add_argument(
        "--workers", type=positive_int, default=min(4, os.cpu_count() or 1),
        help="parallel detector processes (default: up to 4 CPU cores; 1 disables multiprocessing)",
    )
    parser.add_argument(
        "--no-overlays", action="store_true",
        help="save consolidated results.json without rendering JPEG overlays",
    )
    add_streaming_arguments(parser)
    return parser


def add_streaming_arguments(parser, *, live_sources=True):
    parser.add_argument('--streaming', action='store_true',
                        help='stateful tracking + periodic ROI checks; uses one worker'
                        + (', latest-frame buffering for live sources' if live_sources else ''))
    parser.add_argument('--verify-every', type=positive_int, default=10,
                        help='streaming: revalidate tracked ROIs every N processed frames (default: 10)')
    parser.add_argument('--full-search-every', type=positive_int, default=150,
                        help='streaming: full-frame refresh interval while tracking (default: 150 processed frames)')
    parser.add_argument('--roi-margin', type=positive_int, default=96,
                        help='streaming: native-pixel margin around tracked rail pairs (default: 96)')
    parser.add_argument('--max-track-gap', type=positive_float, default=10.0,
                        help='streaming: reacquire after a timestamp gap exceeding this many seconds (default: 10; motion must still pass consistency checks)')


class VideoResultsWriter:
    """Append frames to one JSON document without buffering/re-writing history.

    Only the small closing footer is overwritten after each frame. The file is
    valid JSON after each flush, and finalized on normal exit or Ctrl-C. Like
    ordinary file writes this is not atomic against power loss / SIGKILL.
    """
    def __init__(self, path, run):
        header = json.dumps(run, indent=2, allow_nan=False)
        self.output = path.open('x+', encoding='utf-8')
        self.count = 0
        try:
            self.output.write('{\n"schema_version": 1,\n"run": ' + header + ',\n"frames": [\n')
            self.position = self.output.tell()
            self.footer = self._footer({'stop_reason': 'running', 'processed_frames': 0}, False)
            self._write_footer(self.footer)
        except BaseException:
            self.output.close()
            raise

    @staticmethod
    def _footer(summary, finalized):
        return '\n],\n"summary": ' + json.dumps(summary, indent=2, allow_nan=False) + ',\n"finalized": ' + ('true' if finalized else 'false') + '\n}\n'

    def _write_footer(self, footer):
        self.output.write(footer)
        self.output.truncate()
        self.output.flush()

    def append(self, result):
        serialized = json.dumps(result, allow_nan=False)
        previous_position = self.position
        self.output.seek(previous_position)
        try:
            self.output.write((',' if self.count else '') + serialized + '\n')
            position = self.output.tell()
            footer = self._footer({'stop_reason': 'running', 'processed_frames': self.count + 1}, False)
            self._write_footer(footer)
        except BaseException:
            self.output.seek(previous_position)
            self._write_footer(self.footer)
            raise
        self.position, self.footer = position, footer
        self.count += 1

    def finish(self, summary):
        try:
            self.output.seek(self.position)
            self._write_footer(self._footer(summary, True))
        finally:
            self.output.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.end_time is not None and args.end_time <= args.start_time:
        parser.error("--end-time must be greater than --start-time")
    local_path = Path(args.source).expanduser()
    is_file = local_path.is_file()
    if is_file:
        source: str | int = str(local_path.resolve())
    elif args.source.isdecimal():
        source = int(args.source)
    elif "://" in args.source:
        source = args.source
    else:
        raise ValueError(f"Video file does not exist: {args.source}")
    bounded = args.start_time > 0 or args.end_time is not None
    if bounded and not is_file:
        parser.error("--start-time and --end-time require a local video file")
    streaming_options = None
    if args.streaming:
        streaming_options = dict(verify_every=args.verify_every,
                                 full_search_every=args.full_search_every, roi_margin=args.roi_margin,
                                 max_gap_seconds=args.max_track_gap)
        if args.workers != 1:
            print('STREAMING: using one ordered worker to preserve tracking state.', file=sys.stderr)
        args.workers = 1

    candidate_path = args.candidate.resolve()
    if not candidate_path.is_file():
        raise ValueError(f"Candidate script does not exist: {candidate_path}")
    output_directory = (
        args.output_dir
        if args.output_dir is not None
        else local_path.with_name(f"{local_path.stem}_candidate_results")
        if is_file
        else Path("video_candidate_results")
    ).resolve()
    if output_directory.exists() and (
        not output_directory.is_dir() or any(output_directory.iterdir())
    ):
        raise ValueError(f"Output directory must be empty: {output_directory}")

    # Validate before opening capture or launching workers.
    process_image, process_frame = load_candidate_functions(candidate_path)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise OSError("Could not open video source.")

    summary: dict[str, object] = {
        "decoded_frames": 0,
        "processed_frames": 0,
        "failed_frames": 0,
        "frames_with_detections": 0,
        "predicted_tracks": 0,
        "total_inference_seconds": 0.0,
        "stop_reason": None,
    }
    started_at = time.perf_counter()
    exit_code = 0
    pool = None
    results = None
    latest_reader = None
    capture_summary = summary
    manifest = None
    run_config = {
        "source": source if is_file or isinstance(source, int) else "stream URL",
        "candidate": str(candidate_path),
        "reported_fps": None,
        "every_n_frames": args.every_n_frames,
        "max_frames": args.max_frames,
        "start_time_seconds": args.start_time,
        "end_time_seconds": args.end_time,
        "time_range": "start inclusive, end exclusive; original video timestamps",
        "workers": args.workers,
        "overlays": not args.no_overlays,
        "candidate_input": "BGR array" if process_frame else "temporary BMP",
        "frame_index_base": 0,
        "coordinates": "original decoded frame pixels; overlay header excluded",
        "streaming": streaming_options,
        "live_latest_frame": bool(args.streaming and not is_file),
    }
    previous_threads = cv2.getNumThreads()
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = fps if math.isfinite(fps) and fps > 0 else None
        run_config['reported_fps'] = fps
        manifest = VideoResultsWriter(output_directory / 'results.json', run_config)
        first_frame = 0
        if bounded and fps is None:
            raise ValueError("Time ranges require a video with a valid reported frame rate.")
        if args.start_time > 0:
            # Seek instead of decoding the entire prefix. The timestamp gate below
            # discards any frames before the requested start after seeking.
            target_frame = math.floor(args.start_time * fps)
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame):
                raise ValueError("Video backend could not seek to --start-time.")
            position = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if not math.isfinite(position) or position < 0 or position > target_frame + 0.5:
                raise ValueError("Video backend reported an invalid frame position after seeking.")
            first_frame = round(position)
        if args.workers == 1:
            initialize_processor(candidate_path, output_directory, not args.no_overlays,
                                 streaming_options=streaming_options)
        else:
            pool = ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_processor,
                initargs=(candidate_path, output_directory, not args.no_overlays, True),
            )
        capture_args = copy.copy(args)
        if args.streaming and not is_file:
            capture_args.max_frames = None  # Limit consumed results, not producer frames.
            capture_summary = {'decoded_frames': 0, 'stop_reason': None}
        frames = read_frames(capture, capture_args, is_file, fps, capture_summary, started_at, first_frame)
        if args.streaming and not is_file:
            latest_reader = LatestFrameReader(frames, stop_capture=capture.release)
            frames = latest_reader
        results = ordered_results(frames, pool, args.workers)
        last_progress = started_at
        for result in results:
            frame_index = result["frame_index"]
            manifest.append(result)
            summary["processed_frames"] += 1
            summary["failed_frames"] += int(result["error"] is not None)
            summary["frames_with_detections"] += int(result["predicted_tracks"] > 0)
            summary["predicted_tracks"] += result["predicted_tracks"]
            summary["total_inference_seconds"] += result["elapsed_seconds"]
            if args.streaming:
                details = result['streaming']
                modes = summary.setdefault('streaming_modes', {})
                modes[details['mode']] = modes.get(details['mode'], 0) + 1
                summary['detector_calls'] = summary.get('detector_calls', 0) + details['detector_calls']
                summary['detector_seconds'] = summary.get('detector_seconds', 0.0) + details['detector_seconds']
                summary['motion_seconds'] = summary.get('motion_seconds', 0.0) + details['motion_seconds']
            now = time.perf_counter()
            if now - last_progress >= 1.0 or result["error"]:
                rate = summary["processed_frames"] / max(now - started_at, 1e-9)
                status = result["error"] or f"{result['predicted_tracks']} track(s)"
                if args.streaming:
                    status += f" [{result['streaming']['mode']}]"
                print(f"FRAME {frame_index}: {status} | {rate:.2f} processed fps", file=sys.stderr, flush=True)
                last_progress = now
            if latest_reader is not None and args.max_frames is not None and summary['processed_frames'] >= args.max_frames:
                latest_reader.close()
                summary['stop_reason'] = 'max_frames'
                break
        if latest_reader is not None and summary['stop_reason'] is None:
            summary['stop_reason'] = capture_summary['stop_reason']
        if summary["stop_reason"] in {"stream_read_failed", "frame_decode_failed"}:
            exit_code = 1
    except KeyboardInterrupt:
        summary["stop_reason"] = "interrupted"
        exit_code = 130
    except Exception as error:
        summary["stop_reason"] = "error"
        summary["error"] = f"{type(error).__name__}: {error}"
        print(f"error: {summary['error']}", file=sys.stderr)
        exit_code = 1
    finally:
        if latest_reader is not None:
            latest_reader.close()
            summary['dropped_pending_frames'] = latest_reader.dropped
            summary['decoded_frames'] = capture_summary['decoded_frames']
        if latest_reader is None:
            capture.release()
        if results is not None:
            results.close()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        cv2.setNumThreads(previous_threads)
        summary["wall_seconds"] = time.perf_counter() - started_at
        summary["processed_fps"] = summary["processed_frames"] / max(summary["wall_seconds"], 1e-9)
        if manifest is not None:
            manifest.finish(summary)

    if not summary["processed_frames"] or summary["failed_frames"]:
        exit_code = exit_code or 1
    print(f"Saved {summary['processed_frames']} frame results to {output_directory / 'results.json'}")
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
