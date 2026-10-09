"""Confidence is optional inference metadata, never a substitute for geometry."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from railways.evaluator import validate_predicted_tracks
from railways.run_candidate import attach_confidence, confidence_label, process_one_image
from railways import run_candidate_video as video


class ConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.track = {"left_rail": [[20, 10], [20, 100]],
                      "right_rail": [[40, 10], [40, 100]]}

    def test_scores_preserved_without_changing_geometry(self):
        raw = [{**self.track, "confidence": 0.8,
                "confidence_method": "verified_patch_fraction_v1"}]
        tracks = validate_predicted_tracks(raw, (120, 160))
        before = copy.deepcopy(tracks)
        attach_confidence(raw, tracks)
        self.assertEqual(tracks[0]["confidence"], 0.8)
        self.assertEqual(tracks[0]["confidence_method"], "verified_patch_fraction_v1")
        self.assertEqual(validate_predicted_tracks(tracks, (120, 160)), before)
        self.assertEqual(confidence_label(tracks[0], 0), "T1 confidence=80% (uncalibrated)")

    def test_missing_scores_are_unknown_not_zero(self):
        tracks = validate_predicted_tracks([self.track], (120, 160))
        attach_confidence([self.track], tracks)
        self.assertIsNone(tracks[0]["confidence"])
        self.assertIsNone(tracks[0]["confidence_method"])
        self.assertIn("confidence=n/a", confidence_label(tracks[0], 0))
        attach_confidence([], [])

    def test_zero_and_one_are_valid_scores(self):
        for score in [0, 1, np.float32(0.5)]:
            raw = [{**self.track, "confidence": score}]
            tracks = validate_predicted_tracks(raw, (120, 160))
            attach_confidence(raw, tracks)
            self.assertEqual(tracks[0]["confidence"], float(score))
            json.dumps(tracks, allow_nan=False)

    def test_bad_scores_and_methods_rejected(self):
        for score in [True, "0.8", -0.1, 1.1, float("nan"), float("inf")]:
            with self.subTest(score=score), self.assertRaisesRegex(ValueError, "confidence"):
                attach_confidence([{**self.track, "confidence": score}], [self.track.copy()])
        for method in [None, 12, " "]:
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "confidence_method"):
                attach_confidence([{**self.track, "confidence": 0.5, "confidence_method": method}],
                                  [self.track.copy()])

    def test_image_result_and_overlay_keep_each_track_score(self):
        raw = [{**self.track, "confidence": 0.8, "confidence_method": "verified_patch_fraction_v1"},
               {**self.track, "confidence": 0.5}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "frame.png"
            cv2.imwrite(str(source), np.zeros((120, 160, 3), dtype=np.uint8))
            with patch("railways.run_candidate.cv2.putText", wraps=cv2.putText) as draw:
                result = process_one_image(source, input_directory=root,
                                           output_directory=root / "out", recursive=False,
                                           process_image=lambda _: raw)
            self.assertIsNone(result["error"])
            self.assertEqual([track["confidence"] for track in result["predictions"]], [0.8, 0.5])
            labels = [call.args[1] for call in draw.call_args_list]
            self.assertIn("T1 confidence=80% (uncalibrated)", labels)
            self.assertIn("T2 confidence=50% (uncalibrated)", labels)
            self.assertTrue((root / "out" / result["debug_image"]).is_file())
            json.dumps(result, allow_nan=False)

    def test_video_array_and_legacy_path_keep_score_without_overlays(self):
        raw = [{**self.track, "confidence": 0.8}]
        for array_input in (False, True):
            with self.subTest(array_input=array_input), tempfile.TemporaryDirectory() as directory:
                with patch.object(video, "load_candidate_functions", return_value=(
                    lambda _: raw, (lambda _: raw) if array_input else None
                )):
                    previous_threads = cv2.getNumThreads()
                    self.addCleanup(cv2.setNumThreads, previous_threads)
                    video.initialize_processor("unused", Path(directory), False)
                    result = video.process_video_frame(np.zeros((120, 160, 3), np.uint8),
                                                       {"frame_index": 20, "timestamp_seconds": 0.8})
                self.assertIsNone(result["error"])
                self.assertEqual(result["predictions"][0]["confidence"], 0.8)
                self.assertIsNone(result["debug_image"])


if __name__ == "__main__":
    unittest.main()
