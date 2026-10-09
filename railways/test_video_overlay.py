"""Exercise the overlay exporter with real local video encoding/decoding."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from railways import run_candidate_video_overlay as overlay
from railways.run_candidate import render_debug_frame


class VideoOverlayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='railway-overlay-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'input.avi'
        self.output = self.root / 'annotated.avi'
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (320, 240))
        self.assertTrue(writer.isOpened())
        for index in range(5):
            writer.write(np.full((240, 320, 3), 40 + index * 20, np.uint8))
        writer.release()
        self.track = dict(left_rail=[[100, 30], [100, 200]],
                          right_rail=[[140, 30], [140, 200]], confidence=0.8)

    def run_overlay(self, *args, predict=None):
        predict = predict or (lambda frame: [self.track])
        with (
            patch('railways.run_candidate_video.load_candidate_functions', return_value=(lambda _: [], predict)),
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()),
        ):
            return overlay.main([str(self.source), '--output', str(self.output), *args])

    def results(self):
        return json.loads(self.output.with_suffix('.results.json').read_text())

    def decoded(self):
        capture = cv2.VideoCapture(str(self.output))
        self.assertTrue(capture.isOpened())
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            capture.release()
        return fps, frames

    def test_video_has_rail_overlay_and_one_consolidated_json(self):
        self.assertEqual(self.run_overlay(), 0)
        fps, frames = self.decoded()
        self.assertAlmostEqual(fps, 10)
        self.assertEqual(len(frames), 5)
        self.assertEqual(frames[0].shape, (350, 320, 3))
        # Away from text: left rail cyan, right rail magenta, corridor green.
        blue, green, red = frames[0][160, 100].astype(int)
        self.assertGreater(blue, red + 50)
        self.assertGreater(green, red + 50)
        blue, green, red = frames[0][160, 140].astype(int)
        self.assertGreater(blue, green + 50)
        self.assertGreater(red, green + 50)
        blue, green, red = frames[0][160, 120].astype(int)
        self.assertGreater(green, red + 15)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),
                         ['annotated.avi', 'annotated.results.json', 'input.avi'])
        result = self.results()
        self.assertTrue(result['finalized'])
        self.assertEqual(result['summary']['processed_frames'], 5)
        self.assertEqual(result['frames'][0]['predictions'][0]['confidence'], 0.8)
        self.assertIsNone(result['frames'][0]['debug_image'])

    def test_sampling_adjusts_playback_fps_and_keeps_source_timestamps(self):
        self.assertEqual(self.run_overlay('--start-time', '0.1', '--end-time', '0.4',
                                          '--every-n-frames', '2'), 0)
        fps, frames = self.decoded()
        self.assertAlmostEqual(fps, 5)
        self.assertEqual(len(frames), 2)
        results = self.results()['frames']
        self.assertEqual([r['frame_index'] for r in results], [1, 3])
        self.assertEqual([r['output_frame_index'] for r in results], [0, 1])
        self.assertAlmostEqual(results[1]['timestamp_seconds'], 0.3)
        self.assertAlmostEqual(results[1]['output_timestamp_seconds'], 0.2)

    def test_streaming_and_frame_limit(self):
        predict = Mock(return_value=[])
        self.assertEqual(self.run_overlay('--streaming', '--max-frames', '3', predict=predict), 0)
        self.assertEqual(predict.call_count, 1)
        self.assertEqual(len(self.decoded()[1]), 3)
        self.assertEqual(self.results()['summary']['processing_modes'], {'search': 1, 'search_wait': 2})

    def test_mp4_default_encoder(self):
        self.output = self.root / 'annotated.mp4'
        self.assertEqual(self.run_overlay('--max-frames', '1'), 0)
        self.assertEqual(len(self.decoded()[1]), 1)
        self.assertEqual(self.results()['run']['codec'], 'mp4v')

    def test_interrupt_finalizes_partial_video_and_results(self):
        predict = Mock(side_effect=[[self.track], KeyboardInterrupt])
        self.assertEqual(self.run_overlay(predict=predict), 130)
        self.assertEqual(len(self.decoded()[1]), 1)
        self.assertEqual(self.results()['summary']['stop_reason'], 'interrupted')
        self.assertEqual(len(self.results()['frames']), 1)

    def test_candidate_errors_are_annotated_and_saved(self):
        predict = Mock(side_effect=[[self.track], RuntimeError('detector broke'), [], [], []])
        self.assertEqual(self.run_overlay(predict=predict), 1)
        self.assertEqual(len(self.decoded()[1]), 5)
        self.assertIn('detector broke', self.results()['frames'][1]['error'])
        self.assertEqual(self.results()['summary']['failed_frames'], 1)

    def test_existing_output_not_overwritten(self):
        self.assertEqual(self.run_overlay('--max-frames', '1'), 0)
        before = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            self.run_overlay()
        self.assertEqual(self.output.read_bytes(), before)

    def test_existing_sidecar_not_overwritten(self):
        self.assertEqual(self.run_overlay('--start-time', '10'), 1)
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            self.run_overlay()

    def test_encoder_failure_reported(self):
        writer = Mock()
        writer.isOpened.return_value = False
        with patch('railways.run_candidate_video_overlay.cv2.VideoWriter', return_value=writer):
            self.assertEqual(self.run_overlay(), 1)
        writer.release.assert_called_once()
        self.assertIn('encoder', self.results()['summary']['error'])
        self.assertEqual(self.results()['frames'], [])

    def test_renderer_does_not_modify_or_resize_source(self):
        frame = np.full((120, 160, 3), 75, np.uint8)
        before = frame.copy()
        with patch.object(cv2, 'resize', side_effect=AssertionError('No resizing')):
            rendered = render_debug_frame(frame, display_name='test', tracks=[], elapsed_seconds=0,
                                          size_metrics={}, error=None)
        np.testing.assert_array_equal(frame, before)
        np.testing.assert_array_equal(rendered[-120:], before)


if __name__ == '__main__':
    unittest.main()
