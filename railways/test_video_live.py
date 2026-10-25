"""Headless, real-clock tests: slow detection must not slow video playback."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from railways import run_candidate_video_live as live


class LivePlayerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='railway-live-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'input.avi'
        self.candidate = self.root / 'candidate.py'
        self.results = self.root / 'results.json'
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*'MJPG'), 25, (320, 240))
        self.assertTrue(writer.isOpened())
        for index in range(75):
            writer.write(np.full((240, 320, 3), index, np.uint8))
        writer.release()

    def candidate_code(self, delay=0.2, error=False):
        # A deliberately slow native-resolution candidate in the actual child
        # process, not a mock that hides blocking or multiprocessing behavior.
        self.candidate.write_text(
            'import time\n'
            'def process_image(path): return []\n'
            'def process_frame(frame):\n'
            f'    time.sleep({delay})\n'
            '    assert frame.shape == (240, 320, 3)\n'
            + ('    raise RuntimeError("test detector error")\n' if error else
               '    return [{"left_rail": [[100,30],[100,200]], '
               '"right_rail": [[140,30],[140,200]], "confidence": 0.8}]\n'))

    def run_player(self, *args, headless=True):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return live.main([str(self.source), '--candidate', str(self.candidate),
                              '--results', str(self.results),
                              *(['--headless'] if headless else []), *args])

    def saved(self):
        return json.loads(self.results.read_text())

    def test_slow_detector_skips_frames_and_keeps_wall_clock(self):
        self.candidate_code(0.2)
        self.assertEqual(self.run_player('--start-time', '0.4', '--end-time', '2.4'), 0)
        data = self.saved()
        self.assertTrue(data['finalized'])
        summary, results = data['summary'], data['frames']
        self.assertGreaterEqual(summary['playback_wall_seconds'], 1.98)
        self.assertLess(summary['playback_wall_seconds'], 2.4)
        self.assertGreaterEqual(len(results), 5)
        self.assertLessEqual(len(results), 10)
        self.assertGreaterEqual(summary['displayed_frames'], 30)
        self.assertGreater(summary['source_frames_not_submitted'], 30)
        indices = [result['frame_index'] for result in results]
        self.assertTrue(all(b - a >= 4 for a, b in zip(indices, indices[1:])))
        for result in results:
            self.assertGreaterEqual(result['timestamp_seconds'], 0.4)
            self.assertLess(result['timestamp_seconds'], 2.4)
            self.assertGreaterEqual(result['result_age_seconds'], 0.2)
            # No stale inference queue: at submission this input was current.
            self.assertLess(result['result_age_seconds'], 0.5)
            self.assertEqual(result['predicted_tracks'], 1)
            self.assertEqual(result['width'], 320)

    def test_ending_does_not_wait_for_inflight_five_second_job(self):
        self.candidate_code(5)
        started = time.perf_counter()
        self.assertEqual(self.run_player('--end-time', '0.4'), 0)
        self.assertLess(time.perf_counter() - started, 3)
        summary = self.saved()['summary']
        self.assertEqual(summary['processed_frames'], 0)
        self.assertEqual(summary['submitted_frames'], 1)
        self.assertEqual(summary['cancelled_inflight_frames'], 1)

    def test_detector_errors_saved_without_stalling_playback(self):
        self.candidate_code(0.05, error=True)
        self.assertEqual(self.run_player('--end-time', '0.4'), 0)
        data = self.saved()
        self.assertGreater(data['summary']['failed_frames'], 0)
        self.assertTrue(all('test detector error' in result['error'] for result in data['frames']))
        self.assertTrue(all(not result['live_overlay_eligible_on_arrival'] for result in data['frames']))

    def test_import_failure_is_reported_and_finalizes_json(self):
        self.candidate.write_text('raise RuntimeError("bad import")\n')
        self.assertEqual(self.run_player('--end-time', '0.4'), 1)
        data = self.saved()
        self.assertTrue(data['finalized'])
        self.assertIn('bad import', data['summary']['error'])

    def test_existing_results_are_not_overwritten(self):
        self.candidate_code(0)
        self.results.write_text('preserve me')
        self.assertEqual(self.run_player('--end-time', '0.4'), 1)
        self.assertEqual(self.results.read_text(), 'preserve me')

    def test_gui_exit_stops_worker_without_waiting(self):
        self.candidate_code(5)
        shown = []
        def key(_):
            time.sleep(0.003)
            return ord('q') if len(shown) > 5 else -1
        with (
            patch.object(cv2, 'namedWindow'), patch.object(cv2, 'destroyWindow') as close,
            patch.object(cv2, 'imshow', side_effect=lambda _, frame: shown.append(frame.shape)),
            patch.object(cv2, 'waitKey', side_effect=key),
            patch.object(cv2, 'getWindowProperty', return_value=1),
        ):
            self.assertEqual(self.run_player(headless=False), 0)
        close.assert_called_once()
        self.assertEqual(self.saved()['summary']['stop_reason'], 'user_quit')
        self.assertGreater(len(shown), 5)

    def test_streaming_reacquires_without_cooldown_after_flow_failure(self):
        self.candidate_code(0.01)
        self.assertEqual(self.run_player('--end-time', '0.4', '--streaming'), 0)
        data = self.saved()
        self.assertIn('search', data['summary']['processing_modes'])
        self.assertTrue(all('streaming' in result for result in data['frames']))
        # The uniform test video has no optical-flow features. Each failed
        # propagation must trigger immediate search, not a cooldown reply.
        self.assertGreater(len(data['frames']), 1)
        self.assertEqual(data['summary']['useful_update_frames'], data['summary']['processed_frames'])
        self.assertEqual(data['summary']['processing_modes'], {'search': len(data['frames'])})
        self.assertNotIn('search_every', data['run']['streaming'])
        for result in data['frames'][1:]:
            self.assertEqual(result['streaming']['reason'], 'tracking_lost')
            self.assertEqual(result['streaming']['detector_calls'], 1)

    def test_invalid_interval(self):
        self.candidate_code(0)
        with self.assertRaises(SystemExit):
            self.run_player('--start-time', '2', '--end-time', '1')
        self.assertFalse(self.results.exists())


class FreshnessTests(unittest.TestCase):
    def setUp(self):
        self.result = dict(timestamp_seconds=1.0, width=320, height=240,
                           predictions=[dict(left_rail=[[100, 30], [100, 200]],
                                             right_rail=[[140, 30], [140, 200]])],
                           predicted_tracks=1, processing_seconds=0.2, error=None)

    def test_stale_coordinates_are_hidden_on_live_panel_only(self):
        frame = np.full((240, 320, 3), 80, np.uint8)
        snapshot = frame.copy()
        live.draw_tracks(snapshot, self.result)
        shown = live.compose_display(frame, snapshot, self.result, 1.5, 0.25, 'test')
        np.testing.assert_array_equal(shown[110:, :320], frame)
        self.assertFalse(np.array_equal(shown[110:, 320:], frame))
        fresh = live.compose_display(frame, snapshot, self.result, 1.2, 0.25, 'test')
        self.assertFalse(np.array_equal(fresh[110:, :320], frame))
        np.testing.assert_array_equal(frame, np.full((240, 320, 3), 80, np.uint8))

    def test_age_gate_and_errors(self):
        self.assertTrue(live.fresh_result(self.result, 1.1, 0.25))
        self.assertFalse(live.fresh_result(self.result, 1.3, 0.25))
        self.assertFalse(live.fresh_result(None, 1, 0.25))
        self.result['error'] = 'broken'
        self.assertFalse(live.fresh_result(self.result, 1.1, 0.25))

    def test_lost_reply_hides_live_rails_but_preserves_analyzed_snapshot(self):
        frame = np.full((240, 320, 3), 80, np.uint8)
        snapshot = frame.copy()
        live.draw_tracks(snapshot, self.result)
        lost = dict(self.result, predictions=[], predicted_tracks=0,
                    streaming={'mode': 'lost'}, timestamp_seconds=1.1)
        shown = live.compose_display(frame, snapshot, lost, 1.15, 0.25, 'test', self.result)
        np.testing.assert_array_equal(shown[110:, :320], frame)
        np.testing.assert_array_equal(shown[110:, 320:], snapshot)
        self.assertFalse(live.fresh_result(lost, 1.15, 0.25))


if __name__ == '__main__':
    unittest.main()
