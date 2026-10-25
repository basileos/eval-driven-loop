#!/usr/bin/env python3
"""Run the railway detector on a local video and save an annotated MP4 or AVI.

Native image pixels are retained beneath an added diagnostic header. The output
is silent, constant-FPS video; selected frames play at source FPS / sampling step.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import cv2

if __package__:
    from . import run_candidate_video as video
    from .run_candidate import render_debug_frame
else:
    import run_candidate_video as video
    from run_candidate import render_debug_frame


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='local input video file')
    parser.add_argument('-o', '--output', type=Path,
                        help='new .mp4 or .avi file (default: INPUT_STEM_overlay.mp4)')
    parser.add_argument('-c', '--candidate', type=Path, default=video.DEFAULT_CANDIDATE_PATH)
    parser.add_argument('--start-time', type=video.timestamp, default=0.0,
                        help='inclusive start: seconds, MM:SS, or HH:MM:SS')
    parser.add_argument('--end-time', type=video.timestamp,
                        help='exclusive absolute end time in the same formats')
    parser.add_argument('--every-n-frames', type=video.positive_int, default=1,
                        help='process/save every Nth frame; output FPS = source FPS / N (default: 1)')
    parser.add_argument('--max-frames', type=video.positive_int,
                        help='maximum processed/output frames')
    parser.add_argument('--codec', help='four-character encoder code (default: mp4v for MP4, MJPG for AVI)')
    video.add_streaming_arguments(parser, live_sources=False)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.end_time is not None and args.end_time <= args.start_time:
        parser.error('--end-time must be greater than --start-time')
    source = args.source.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f'Video file does not exist: {source}')
    output = (args.output.expanduser() if args.output else source.with_name(source.stem + '_overlay.mp4')).resolve()
    if output.suffix.lower() not in {'.mp4', '.avi'}:
        parser.error('--output must end in .mp4 or .avi')
    results_path = output.with_name(output.stem + '.results.json')
    for path in (output, results_path):
        if path.exists():
            raise ValueError(f'Refusing to overwrite existing output: {path}')
    codec = args.codec or ('mp4v' if output.suffix.lower() == '.mp4' else 'MJPG')
    if len(codec) != 4 or not codec.isascii():
        parser.error('--codec must be a four-character ASCII encoder code')
    candidate = args.candidate.expanduser().resolve()
    video.load_candidate_functions(candidate)  # Fail before creating output files.
    streaming_options = None
    if args.streaming:
        streaming_options = dict(verify_every=args.verify_every,
                                 full_search_every=args.full_search_every, roi_margin=args.roi_margin,
                                 max_gap_seconds=args.max_track_gap)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise OSError(f'Could not open video: {source}')
    writer = None
    manifest = None
    previous_threads = cv2.getNumThreads()
    started = time.perf_counter()
    summary = dict(decoded_frames=0, processed_frames=0, failed_frames=0,
                   frames_with_detections=0, predicted_tracks=0, stop_reason=None,
                   total_inference_seconds=0.0)
    exit_code = 0
    writer_shape = None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Video must report a valid FPS to preserve playback timing.')
        output_fps = fps / args.every_n_frames
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest = video.VideoResultsWriter(results_path, dict(
            source=str(source), candidate=str(candidate), output_video=str(output),
            source_fps=fps, output_fps=output_fps, codec=codec, audio=False,
            start_time_seconds=args.start_time, end_time_seconds=args.end_time,
            every_n_frames=args.every_n_frames, max_frames=args.max_frames,
            streaming=streaming_options, workers=1,
            coordinates='original decoded frame pixels; added header excluded',
            timing='constant FPS; one output frame per selected input frame',
        ))
        first_frame = 0
        if args.start_time > 0:
            target = math.floor(args.start_time * fps)
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, target):
                raise ValueError('Video backend could not seek to --start-time.')
            position = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if not math.isfinite(position) or position < 0 or position > target + 0.5:
                raise ValueError('Invalid frame position after seeking.')
            first_frame = round(position)
        video.initialize_processor(candidate, output.parent, False, streaming_options=streaming_options)
        last_progress = started
        for frame, metadata in video.read_frames(capture, args, True, fps, summary, started, first_frame):
            frame_started = time.perf_counter()
            result = video.process_video_frame(frame, metadata)
            mode = result.get('streaming', {}).get('mode', 'full_search')
            stamp = result['timestamp_seconds']
            canvas = render_debug_frame(
                frame, display_name=f"Frame {result['frame_index']} | time={stamp:.3f}s",
                tracks=result['predictions'], elapsed_seconds=result['elapsed_seconds'],
                size_metrics=result, error=result['error'], status_note='mode=' + mode,
            )
            # Video encoders commonly require even dimensions. Pad, never resize
            # or silently crop away a row of source pixels.
            h, w = canvas.shape[:2]
            canvas = cv2.copyMakeBorder(canvas, 0, h % 2, 0, w % 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            if writer is None:
                writer_shape = (canvas.shape[1], canvas.shape[0])
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), output_fps, writer_shape)
                if not writer.isOpened():
                    raise OSError(f'Could not open {codec} video encoder for {output}. Try --codec or an .avi output.')
                summary['output_width'], summary['output_height'] = writer_shape
            if (canvas.shape[1], canvas.shape[0]) != writer_shape:
                raise ValueError('Frame dimensions changed; cannot append to this video.')
            writer.write(canvas)
            result['output_frame_index'] = summary['processed_frames']
            result['output_timestamp_seconds'] = summary['processed_frames'] / output_fps
            result['overlay_header_height'] = h - frame.shape[0]
            result['video_processing_seconds'] = time.perf_counter() - frame_started
            manifest.append(result)
            summary['processed_frames'] += 1
            summary['failed_frames'] += int(result['error'] is not None)
            summary['frames_with_detections'] += int(result['predicted_tracks'] > 0)
            summary['predicted_tracks'] += result['predicted_tracks']
            summary['total_inference_seconds'] += result['elapsed_seconds']
            modes = summary.setdefault('processing_modes', {})
            modes[mode] = modes.get(mode, 0) + 1
            now = time.perf_counter()
            if now - last_progress >= 1 or result['error']:
                print(f"FRAME {result['frame_index']}: {result['predicted_tracks']} track(s) [{mode}] | "
                      f"{summary['processed_frames'] / max(now - started, 1e-9):.2f} processed fps",
                      file=sys.stderr, flush=True)
                last_progress = now
        if summary['stop_reason'] == 'frame_decode_failed':
            exit_code = 1
    except KeyboardInterrupt:
        summary['stop_reason'] = 'interrupted'
        exit_code = 130
    except Exception as error:
        summary.update(stop_reason='error', error=f'{type(error).__name__}: {error}')
        print(f"error: {summary['error']}", file=sys.stderr)
        exit_code = 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()  # Finalize even partial videos after Ctrl-C/errors.
        cv2.setNumThreads(previous_threads)
        summary['wall_seconds'] = time.perf_counter() - started
        summary['processed_fps'] = summary['processed_frames'] / max(summary['wall_seconds'], 1e-9)
        if manifest is not None:
            manifest.finish(summary)
    if not summary['processed_frames'] or summary['failed_frames']:
        exit_code = exit_code or 1
    if summary['processed_frames']:
        print(f"Saved {summary['processed_frames']} annotated frames to {output}")
    if manifest is not None:
        print(f'Results: {results_path}')
    return exit_code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'error: {error}', file=sys.stderr)
        raise SystemExit(1)
