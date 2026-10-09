#!/usr/bin/env python3
"""Play a local video at 1x with asynchronous, best-effort railway detection.

No inference backlog: one job in flight; the next job uses the newest decoded
frame. The right panel shows the actual analyzed frame, never a fictional live
detection. Playback uses reported constant FPS, without audio.
"""
from __future__ import annotations

import argparse
from collections import deque
import math
import multiprocessing as mp
from pathlib import Path
import queue
import sys
import threading
import time

import cv2
import numpy as np

if __package__:
    from . import run_candidate_video as video
    from .evaluator import _draw_track_overlay
else:
    import run_candidate_video as video
    from evaluator import _draw_track_overlay


def detection_worker(jobs, replies, candidate, options):
    """Own the detector/tracker in a separate process (including its Python GIL)."""
    try:
        video.initialize_processor(candidate, Path('.'), False, ignore_sigint=True,
                                   streaming_options=options)
        replies.put(('ready', None))
        while True:
            frame, metadata = jobs.get()
            result = video.process_video_frame(frame, metadata)
            replies.put(('result', result))
    except Exception as error:
        replies.put(('error', f'{type(error).__name__}: {error}'))


class DetectorProcess:
    """A single outstanding job, not a queue of video frames to catch up on."""
    def __init__(self, candidate, options):
        context = mp.get_context('spawn')
        self.jobs = context.Queue(maxsize=1)
        self.replies = context.Queue(maxsize=1)
        self.process = context.Process(target=detection_worker,
                                       args=(self.jobs, self.replies, candidate, options), daemon=True)
        self.ready = False
        self.busy = False
        self.process.start()

    def poll(self):
        try:
            kind, payload = self.replies.get_nowait()
        except queue.Empty:
            if not self.process.is_alive():
                raise RuntimeError(f'Detector process exited ({self.process.exitcode}).')
            return None
        if kind == 'error':
            raise RuntimeError(payload)
        if kind == 'ready':
            self.ready = True
            return None
        self.busy = False
        return payload

    def submit(self, frame, metadata):
        if not self.ready or self.busy:
            return False
        self.jobs.put_nowait((frame, metadata))
        self.busy = True
        return True

    def close(self):
        # Do NOT wait for expensive inference after playback ends / Q is pressed.
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=0.5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=0.5)
        for channel in (self.jobs, self.replies):
            # A pending 4K frame must not block shutdown in a queue feeder thread.
            channel.cancel_join_thread()
            channel.close()


class ClockedDecoder:
    """Decode off the GUI thread, retaining only the latest due frame.

    Capture ownership passes here, including release. If decoding falls behind,
    grab/discard old frames before converting the newest due frame to BGR.
    """
    def __init__(self, capture, fps, first_frame, start, end, origin):
        self.capture, self.fps = capture, fps
        self.first_frame, self.start, self.end, self.origin = first_frame, start, end, origin
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.finished_at = None
        self.error = None
        self.skipped = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def get_latest(self):
        with self.lock:
            return self.latest

    def _run(self):
        index = self.first_frame
        try:
            while not self.stop.is_set():
                media_time = self.start + time.perf_counter() - self.origin
                if media_time >= self.end:
                    break
                due = index / self.fps - media_time
                if due > 0:
                    self.stop.wait(min(due, 0.05))
                    continue
                target = min(math.floor(media_time * self.fps + 1e-7),
                             math.ceil(self.end * self.fps - 1e-7) - 1)
                while index <= target and not self.stop.is_set():
                    if not self.capture.grab():
                        self.finished_at = index / self.fps
                        return
                    if index < target:
                        self.skipped += 1
                    index += 1
                if self.stop.is_set():
                    break
                ok, frame = self.capture.retrieve()
                if not ok:
                    raise OSError(f'Could not decode frame {index - 1}.')
                with self.lock:
                    self.latest = (frame, dict(frame_index=index - 1,
                                              timestamp_seconds=(index - 1) / self.fps,
                                              timestamp_source='frame_index_over_fps'))
        except Exception as error:
            self.error = f'{type(error).__name__}: {error}'
        finally:
            self.capture.release()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=0.5)


def result_age(result, media_time):
    return max(0.0, media_time - result['timestamp_seconds'])


def fresh_result(result, media_time, maximum_age):
    return (result is not None and not result.get('error')
            and result.get('streaming', {}).get('mode') not in {'search_wait', 'lost', 'error'}
            and result_age(result, media_time) <= maximum_age)


def thumbnail(frame, width):
    height = max(1, round(frame.shape[0] * width / frame.shape[1]))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def draw_tracks(canvas, result):
    _draw_track_overlay(canvas, result['predictions'],
                        canvas.shape[1] / result['width'], canvas.shape[0] / result['height'],
                        fill_color=(70, 255, 70), left_color=(255, 255, 0),
                        right_color=(255, 0, 255), alpha=0.2, line_thickness=2)


def compose_display(live, snapshot, result, media_time, max_age, status, snapshot_result=None):
    left = live.copy()
    if fresh_result(result, media_time, max_age):
        draw_tracks(left, result)
        label = f'LIVE 1x | recent overlay, age {result_age(result, media_time):.2f}s'
    else:
        label = 'LIVE 1x | no fresh overlay'
    right = snapshot.copy() if snapshot is not None else np.zeros_like(left)
    right = cv2.resize(right, (left.shape[1], left.shape[0]))
    title = 'LAST ANALYZED FRAME | waiting for first result'
    detail = 'Original-resolution detection; display only is resized'
    shown_result = snapshot_result if snapshot_result is not None else result
    if shown_result is not None:
        age = result_age(shown_result, media_time)
        mode = shown_result.get('streaming', {}).get('mode', 'full_search')
        title = f"ANALYZED t={shown_result['timestamp_seconds']:.2f}s | age={age:.2f}s"
        detail = f"{mode} | tracks={shown_result['predicted_tracks']} | compute={shown_result['processing_seconds']:.3f}s"
        if shown_result.get('error'):
            detail = 'DETECTOR ERROR (see console / JSON)'
    canvas = np.zeros((left.shape[0] + 110, left.shape[1] * 2, 3), np.uint8)
    canvas[110:, :left.shape[1]] = left
    canvas[110:, left.shape[1]:] = right
    for text, position in [(label, (10, 24)), (title, (left.shape[1] + 10, 24)),
                           (status, (10, 53)), (detail, (left.shape[1] + 10, 53)),
                           (f'Video time {media_time:.2f}s | Q / Esc: exit | silent playback', (10, 85))]:
        cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='local constant-FPS video file')
    parser.add_argument('-c', '--candidate', type=Path, default=video.DEFAULT_CANDIDATE_PATH)
    parser.add_argument('--start-time', type=video.timestamp, default=0.0)
    parser.add_argument('--end-time', type=video.timestamp)
    parser.add_argument('--results', type=Path, help='optional new consolidated JSON file (no MP4/JPEG export)')
    parser.add_argument('--width', type=video.positive_int, default=1440, help='combined display width; inference stays native resolution')
    parser.add_argument('--max-overlay-age', type=video.positive_float, default=0.25,
                        help='hide live overlay once older than this many seconds (default: 0.25)')
    parser.add_argument('--headless', action='store_true', help='benchmark the same 1x clock without opening a window')
    video.add_streaming_arguments(parser, live_sources=False)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.end_time is not None and args.end_time <= args.start_time:
        parser.error('--end-time must be greater than --start-time')
    if args.width < 640:
        parser.error('--width must be at least 640')
    source = args.source.expanduser().resolve()
    candidate = args.candidate.expanduser().resolve()
    if not source.is_file() or not candidate.is_file():
        raise ValueError('Source video and candidate must exist.')
    options = None
    if args.streaming:
        options = dict(verify_every=args.verify_every, search_every=args.search_every,
                       full_search_every=args.full_search_every, roi_margin=args.roi_margin,
                       max_gap_seconds=args.max_track_gap)
    capture = cv2.VideoCapture(str(source))
    decoder = worker = manifest = None
    window = 'Railways: live video | last analyzed frame'
    window_open = False
    old_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    summary = dict(processed_frames=0, submitted_frames=0, displayed_frames=0,
                   failed_frames=0, frames_with_detections=0, processing_modes={},
                   useful_update_frames=0, fresh_detection_frames=0,
                   detector_calls=0, stop_reason='end_of_video')
    origin = None
    exit_code = 0
    try:
        if not capture.isOpened():
            raise OSError(f'Could not open video: {source}')
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Video must report a valid constant FPS.')
        duration = count / fps if math.isfinite(count) and count > 0 else math.inf
        end = min(args.end_time if args.end_time is not None else math.inf, duration)
        if args.start_time >= end:
            raise ValueError('Start time is beyond the video / selected interval.')
        first = math.ceil(args.start_time * fps - 1e-7)
        if first and not capture.set(cv2.CAP_PROP_POS_FRAMES, first):
            raise ValueError('Could not seek to --start-time.')
        position = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if not math.isfinite(position) or abs(position - first) > 0.5:
            raise ValueError('Video backend did not seek to the requested frame.')
        if args.results:
            path = args.results.expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            manifest = video.VideoResultsWriter(path, dict(
                source=str(source), candidate=str(candidate), source_fps=fps,
                start_time_seconds=args.start_time, end_time_seconds=end if math.isfinite(end) else None,
                streaming=options, max_overlay_age_seconds=args.max_overlay_age,
                timing='1x wall clock, constant source FPS; skip inference frames when busy',
                coordinates='original decoded frame pixels', audio=False))
        worker = DetectorProcess(candidate, options)
        # Initialize once before starting the playback clock, not halfway through
        # a short clip. No inference is done here; only load candidate imports.
        startup = time.perf_counter()
        print('Loading detector...', flush=True)
        while not worker.ready:
            worker.poll()
            if time.perf_counter() - startup > 30:
                raise TimeoutError('Detector initialization exceeded 30 seconds.')
            time.sleep(0.01)
        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            window_open = True
        print('Playing at 1x. Inference frames are skipped when busy; Q/Esc exits.', flush=True)
        origin = time.perf_counter()
        decoder = ClockedDecoder(capture, fps, first, args.start_time, end, origin)
        last_displayed = last_submitted = -1
        live = np.zeros((round(args.width / 2 * 9 / 16), args.width // 2, 3), np.uint8)
        pending_snapshot = snapshot = result = snapshot_result = None
        completions = deque()
        displayed = deque()
        detector_calls = deque()
        last_progress = origin
        while True:
            now = time.perf_counter()
            media_time = args.start_time + now - origin
            if media_time >= end or (decoder.finished_at is not None and media_time >= decoder.finished_at):
                break
            if decoder.error:
                raise RuntimeError(decoder.error)
            update = worker.poll()
            if update is not None:
                result = update
                result['result_available_at_video_seconds'] = media_time
                result['result_age_seconds'] = result_age(result, media_time)
                result['live_overlay_eligible_on_arrival'] = fresh_result(result, media_time, args.max_overlay_age)
                mode = result.get('streaming', {}).get('mode', 'full_search')
                summary['processing_modes'][mode] = summary['processing_modes'].get(mode, 0) + 1
                summary['processed_frames'] += 1
                summary['failed_frames'] += int(bool(result.get('error')))
                summary['frames_with_detections'] += int(result['predicted_tracks'] > 0)
                summary['fresh_detection_frames'] += int(result['predicted_tracks'] > 0
                                                          and result['live_overlay_eligible_on_arrival'])
                calls = result.get('streaming', {}).get('detector_calls', 1)
                summary['detector_calls'] += calls
                detector_calls.extend([now] * calls)
                # Empty cooldown/lost/error replies are NOT useful detection FPS.
                useful = not result.get('error') and mode not in {'search_wait', 'lost', 'error'}
                if useful:
                    completions.append(now)
                    summary['useful_update_frames'] += 1
                if result.get('error'):
                    print(result['error'], file=sys.stderr, flush=True)
                if manifest is not None:
                    manifest.append(result)
                # Cooldown replies did not run detection; do not immediately
                # replace the analyzed snapshot with an unexamined empty frame.
                if not args.headless and useful:
                    snapshot_result = result
                    snapshot = pending_snapshot
                    draw_tracks(snapshot, result)
            newest = decoder.get_latest()
            if newest is not None:
                frame, metadata = newest
                index = metadata['frame_index']
                if index != last_displayed:
                    if not args.headless:
                        live = thumbnail(frame, args.width // 2)
                    last_displayed = index
                    summary['displayed_frames'] += 1
                    displayed.append(now)
                # Never submit the same frame twice, or an old decoder frame.
                if index != last_submitted and media_time - metadata['timestamp_seconds'] < 1 / fps + 0.1:
                    if worker.submit(frame, metadata):
                        last_submitted = index
                        summary['submitted_frames'] += 1
                        if not args.headless:
                            pending_snapshot = live.copy()
            for events in (completions, displayed, detector_calls):
                while events and events[0] < now - 2:
                    events.popleft()
            interval = max(0.1, min(2.0, now - origin))
            status = (f'View {len(displayed)/interval:.1f} FPS | updates {len(completions)/interval:.1f}/s'
                      f' | detector {len(detector_calls)/interval:.1f}/s | {"busy" if worker.busy else "idle"}')
            if now - last_progress >= 1:
                print(f't={media_time:.2f}s | {status}', flush=True)
                last_progress = now
            if args.headless:
                time.sleep(0.003)
            else:
                cv2.imshow(window, compose_display(live, snapshot, result, media_time,
                                                   args.max_overlay_age, status, snapshot_result))
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    summary['stop_reason'] = 'user_quit'
                    break
    except KeyboardInterrupt:
        summary['stop_reason'] = 'interrupted'
        exit_code = 130
    except Exception as error:
        summary.update(stop_reason='error', error=f'{type(error).__name__}: {error}')
        print(summary['error'], file=sys.stderr)
        exit_code = 1
    finally:
        elapsed = time.perf_counter() - origin if origin is not None else 0.0
        summary['playback_wall_seconds'] = elapsed
        summary['completed_results_per_second'] = summary['processed_frames'] / max(elapsed, 1e-9)
        summary['useful_updates_per_second'] = summary['useful_update_frames'] / max(elapsed, 1e-9)
        summary['cancelled_inflight_frames'] = int(worker is not None and worker.busy)
        available_frames = (max(0, math.ceil(min(end, args.start_time + elapsed) * fps - 1e-7) - first)
                            if origin is not None else 0)
        summary['source_frames_not_submitted'] = max(0, available_frames - summary['submitted_frames'])
        summary['decode_skipped_frames'] = decoder.skipped if decoder else 0
        if decoder is not None:
            decoder.close()
        else:
            capture.release()
        if worker is not None:
            worker.close()
        if window_open:
            cv2.destroyWindow(window)
        cv2.setNumThreads(old_threads)
        if manifest is not None:
            manifest.finish(summary)
    print(f"Playback {summary['playback_wall_seconds']:.2f}s; {summary['useful_update_frames']} useful updates "
          f"({summary['useful_updates_per_second']:.2f}/s); {summary['processed_frames']} total replies; "
          f"detections on {summary['frames_with_detections']} frames "
          f"({summary['fresh_detection_frames']} fresh on arrival); skipped inference on "
          f"{summary['source_frames_not_submitted']} source frames.")
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
