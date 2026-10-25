"""Deterministic motion, loss, ROI mapping, and bounded live buffering tests."""
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from railways.latest_frame import LatestFrameReader
from railways.streaming_tracker import StreamingTracker, offset_track, roi_of


class StreamingTrackerTests(unittest.TestCase):
    def setUp(self):
        self.track = dict(left_rail=[[125., 30.], [125., 110.], [125., 205.]],
                          right_rail=[[160., 30.], [160., 110.], [160., 205.]],
                          confidence=0.9, confidence_method='test_evidence')
        rng = np.random.default_rng(17)
        gray = np.zeros((240, 320), np.uint8)
        gray[15:225, 105:180] = rng.integers(0, 256, (210, 75), dtype=np.uint8)
        self.frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def moved(self, dx=2, dy=1):
        return cv2.warpAffine(self.frame, np.float32([[1, 0, dx], [0, 1, dy]]), (320, 240))

    def test_tracks_translation_without_rerunning_detector_or_reusing_confidence(self):
        detect = Mock(return_value=[self.track])
        tracker = StreamingTracker(detect, verify_every=20)
        self.assertEqual(len(tracker.process(self.frame, 0.0)), 1)
        self.assertEqual(tracker.diagnostics['mode'], 'search')
        found = tracker.process(self.moved(), 0.04)
        self.assertEqual(len(found), 1)
        self.assertEqual(tracker.diagnostics['mode'], 'track')
        self.assertEqual(detect.call_count, 1)
        self.assertIsNone(found[0]['confidence'])
        self.assertIsNone(found[0]['confidence_method'])
        np.testing.assert_allclose(found[0]['left_rail'],
                                   offset_track(self.track, 2, 1)['left_rail'], atol=0.25)

    def test_blank_frame_drops_track_and_does_not_extrapolate(self):
        detect = Mock(side_effect=[[self.track], []])
        tracker = StreamingTracker(detect)
        tracker.process(self.frame, 0.0)
        self.assertEqual(tracker.process(np.zeros_like(self.frame), 0.04), [])
        self.assertEqual(tracker.diagnostics['mode'], 'search')
        self.assertEqual(tracker.diagnostics['reason'], 'tracking_lost')
        self.assertEqual(tracker.tracks, [])
        self.assertEqual(detect.call_count, 2)

    def test_failed_acquisition_searches_every_available_frame(self):
        detect = Mock(return_value=[])
        tracker = StreamingTracker(detect)
        for i in range(4):
            self.assertEqual(tracker.process(self.frame, i * 0.04), [])
            self.assertEqual(tracker.diagnostics['mode'], 'search')
            self.assertEqual(tracker.diagnostics['detector_calls'], 1)
        self.assertEqual(detect.call_count, 4)

    def test_lost_track_is_reacquired_on_same_frame_without_cooldown(self):
        detect = Mock(return_value=[self.track])
        tracker = StreamingTracker(detect)
        tracker.process(self.frame, 0.)
        current = self.moved()
        with patch.object(tracker, '_propagate', return_value=None):
            self.assertEqual(len(tracker.process(current, 0.04)), 1)
        self.assertEqual(detect.call_count, 2)
        self.assertEqual(tracker.diagnostics['mode'], 'search')
        self.assertEqual(tracker.diagnostics['reason'], 'tracking_lost')
        self.assertIs(detect.call_args.args[0], current)
        self.assertEqual(tracker.previous_timestamp, 0.04)

    def test_crop_detector_receives_full_frame_reference_shape(self):
        x0, y0, _, _ = roi_of(self.track, self.frame.shape, 16)
        detect = Mock(return_value=[self.track])
        detect_roi = Mock(return_value=[offset_track(self.track, -x0, -y0)])
        tracker = StreamingTracker(detect, detect_roi=detect_roi, verify_every=1, roi_margin=16,
                                   verify_region='roi')
        tracker.process(self.frame, 0.)
        with patch.object(tracker, '_propagate', return_value=[self.track]):
            tracker.process(self.frame, 0.04)
        self.assertEqual(detect.call_count, 1)
        self.assertEqual(detect_roi.call_args.args[1], (240, 320))

    def test_roi_verification_preserves_native_pixels_and_maps_back(self):
        predicted = offset_track(self.track, 2, 1)
        roi = roi_of(predicted, self.frame.shape, 16)
        x0, y0, x1, y1 = roi
        detect = Mock(side_effect=[[self.track], [offset_track(predicted, -x0, -y0)]])
        tracker = StreamingTracker(detect, verify_every=1, roi_margin=16, verify_region='roi')
        tracker.process(self.frame, 0.0)
        moved = self.moved()
        with patch.object(tracker, '_propagate', return_value=[predicted]), patch.object(
            cv2, 'resize', side_effect=AssertionError('No resizing detector input')
        ):
            found = tracker.process(moved, 0.04)
        self.assertEqual(tracker.diagnostics['mode'], 'roi_verify')
        self.assertEqual(found[0]['left_rail'], predicted['left_rail'])
        self.assertEqual(found[0]['confidence'], 0.9)
        np.testing.assert_array_equal(detect.call_args.args[0], moved[y0:y1, x0:x1])
        self.assertLess(tracker.diagnostics['detector_pixels'], 320 * 240)

    def test_roi_rejection_clears_old_geometry(self):
        detect = Mock(side_effect=[[self.track], [], []])
        tracker = StreamingTracker(detect, verify_every=1)
        tracker.process(self.frame, 0.)
        with patch.object(tracker, '_propagate', return_value=[self.track]):
            self.assertEqual(tracker.process(self.frame, 0.04), [])
        self.assertEqual(tracker.tracks, [])
        self.assertEqual(tracker.diagnostics['mode'], 'search')
        self.assertEqual(tracker.diagnostics['reason'], 'tracking_lost')
        self.assertEqual(tracker.diagnostics['verification_failure'], 'no_roi_detection')
        self.assertEqual(tracker.diagnostics['detector_calls'], 2)
        self.assertIs(detect.call_args.args[0], self.frame)

    def test_roi_cannot_silently_switch_to_other_track(self):
        # An adjacent parallel track lies ~3 gauges (here 35 px) to the side.
        x0, y0, _, _ = roi_of(self.track, self.frame.shape, 120)
        wrong = offset_track(self.track, 105 - x0, -y0)
        detect = Mock(side_effect=[[self.track], [wrong], []])
        tracker = StreamingTracker(detect, verify_every=1, roi_margin=120, verify_region='roi')
        tracker.process(self.frame, 0.)
        with patch.object(tracker, '_propagate', return_value=[self.track]):
            self.assertEqual(tracker.process(self.frame, 0.04), [])
        self.assertEqual(tracker.diagnostics['reason'], 'tracking_lost')

    def test_roi_accepts_redetection_within_detector_jitter(self):
        # Consecutive detections of one track differ by up to ~1 gauge.
        x0, y0, _, _ = roi_of(self.track, self.frame.shape, 32)
        jittered = offset_track(self.track, 20 - x0, -y0)
        detect = Mock(side_effect=[[self.track], [jittered]])
        tracker = StreamingTracker(detect, verify_every=1, roi_margin=32, verify_region='roi')
        tracker.process(self.frame, 0.)
        with patch.object(tracker, '_propagate', return_value=[self.track]):
            found = tracker.process(self.frame, 0.04)
        self.assertEqual(tracker.diagnostics['mode'], 'roi_verify')
        self.assertEqual(found[0]['left_rail'], offset_track(self.track, 20, 0)['left_rail'])

    def test_frame_verification_adopts_longer_redetection_of_same_line(self):
        # The prediction covers only part of the line; the re-detection on the
        # whole frame extends it and must be associated, not rejected.
        short = dict(left_rail=[[125., 120.], [125., 205.]], right_rail=[[160., 120.], [160., 205.]],
                     confidence=0.9, confidence_method='test_evidence')
        detect = Mock(side_effect=[[short], [self.track]])
        tracker = StreamingTracker(detect, verify_every=1)
        tracker.process(self.frame, 0.)
        with patch.object(tracker, '_propagate', return_value=[short]):
            found = tracker.process(self.frame, 0.04)
        self.assertEqual(tracker.diagnostics['mode'], 'roi_verify')
        self.assertEqual(found[0]['left_rail'], self.track['left_rail'])
        self.assertEqual(detect.call_args.args[0].shape, self.frame.shape)

    def test_time_gap_resolution_change_and_periodic_refresh_reacquire(self):
        for reason in ['timestamp_gap', 'resolution_changed', 'periodic_refresh']:
            with self.subTest(reason=reason):
                detect = Mock(return_value=[self.track])
                tracker = StreamingTracker(detect, max_gap_seconds=1.0,
                                           full_search_every=1 if reason == 'periodic_refresh' else 150)
                tracker.process(self.frame, 0.0)
                frame = np.zeros((300, 400, 3), np.uint8) if reason == 'resolution_changed' else self.frame
                tracker.process(frame, 2.0 if reason == 'timestamp_gap' else 0.04)
                self.assertEqual(detect.call_count, 2)
                self.assertEqual(tracker.diagnostics['reason'], reason)

    def test_detector_error_invalidates_state_and_propagates(self):
        detect = Mock(side_effect=[[self.track], RuntimeError('broken detector')])
        tracker = StreamingTracker(detect, verify_every=1)
        tracker.process(self.frame, 0.)
        with patch.object(tracker, '_propagate', return_value=[self.track]):
            with self.assertRaisesRegex(RuntimeError, 'broken detector'):
                tracker.process(self.frame, 0.04)
        self.assertEqual(tracker.tracks, [])

    def test_small_detections_rejected_relative_to_full_image(self):
        tiny = dict(left_rail=[[1, 1], [1, 3]], right_rail=[[3, 1], [3, 3]])
        tracker = StreamingTracker(Mock(return_value=[tiny]))
        self.assertEqual(tracker.process(self.frame, 0.), [])


class LatestFrameTests(unittest.TestCase):
    def test_only_latest_pending_frame_is_kept(self):
        reader = LatestFrameReader([(i, {'frame_index': i}) for i in range(100)])
        self.addCleanup(reader.close)
        reader.thread.join(timeout=2)
        frame, metadata = next(reader)
        self.assertEqual(frame, 99)
        self.assertEqual(metadata['frame_index'], 99)
        self.assertEqual(reader.dropped, 99)
        self.assertGreaterEqual(metadata['application_queue_seconds'], 0)
        with self.assertRaises(StopIteration):
            next(reader)

    def test_capture_errors_not_hidden(self):
        def frames():
            yield 1, {}
            raise RuntimeError('capture broke')
        reader = LatestFrameReader(frames())
        self.addCleanup(reader.close)
        reader.thread.join(timeout=2)
        next(reader)
        with self.assertRaisesRegex(RuntimeError, 'capture broke'):
            next(reader)

    def test_close_discards_pending_frames(self):
        reader = LatestFrameReader([(1, {})])
        reader.thread.join(timeout=2)
        reader.close()
        with self.assertRaises(StopIteration):
            next(reader)


if __name__ == '__main__':
    unittest.main()
