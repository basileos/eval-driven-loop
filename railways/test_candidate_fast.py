"""candidate_fast speed refactors must keep outputs identical."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from railways import bench_fast, candidate_fast


GOLDEN = Path(__file__).resolve().parent / "runs" / "golden_fast.json"


def synthetic_track(width=1280, height=720):
    """Two bright rails with dark sleepers on textured ground."""
    rng = np.random.default_rng(7)
    image = rng.integers(70, 110, (height, width, 3), dtype=np.uint8)
    for y in range(0, height, 18):
        cv2.line(image, (560, y), (720, y), (40, 40, 40), 6)
    for x in (600, 680):
        cv2.line(image, (x, 0), (x, height - 1), (200, 200, 200), 4)
    return image


class CandidateFastTests(unittest.TestCase):
    def test_process_frame_matches_process_image(self):
        # Prefer a real frame with detections; the synthetic one yields none.
        positives = [
            bench_fast.ROOT / key
            for key, tracks in (json.loads(GOLDEN.read_text()).items()
                                if GOLDEN.is_file() else ())
            if tracks and (bench_fast.ROOT / key).is_file()
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = positives[0] if positives else Path(directory) / "frame.png"
            if not positives:
                cv2.imwrite(str(path), synthetic_track())
            from_path = candidate_fast.process_image(str(path))
            from_frame = candidate_fast.process_frame(cv2.imread(str(path)))
        self.assertEqual(bench_fast.compare_outputs(from_path, from_frame, 0.0), [])

    def test_stages_are_exclusive_and_sum_to_total(self):
        candidate_fast.process_frame(synthetic_track())
        timings = candidate_fast.process_frame.last_timings
        self.assertEqual(tuple(timings), candidate_fast.STAGES)
        stages = sum(value for name, value in timings.items() if name != "total")
        self.assertLessEqual(stages, timings["total"] + 1e-9)
        self.assertGreater(stages, timings["total"] * 0.9)
        self.assertIn("fit_section", candidate_fast.process_frame.last_counts)

    def test_reference_shape_of_own_frame_changes_nothing(self):
        image = synthetic_track()
        plain = candidate_fast.process_frame(image)
        referenced = candidate_fast.process_frame(image, reference_shape=image.shape)
        self.assertEqual(bench_fast.compare_outputs(plain, referenced, 0.0), [])

    def test_crop_with_reference_shape_returns_crop_coordinates(self):
        image = synthetic_track(3840, 2160)
        crop = np.ascontiguousarray(image[400:1600, 1000:2600])
        for track in candidate_fast.process_frame(crop, reference_shape=image.shape):
            for key in ("left_rail", "right_rail"):
                points = np.asarray(track[key])
                self.assertTrue((points >= 0).all())
                self.assertTrue((points[:, 0] <= crop.shape[1] - 1).all())
                self.assertTrue((points[:, 1] <= crop.shape[0] - 1).all())

    def test_seed_verification_respects_budget(self):
        # The synthetic frame has no accepted track, so it exhausts the budget.
        candidate_fast.process_frame(synthetic_track())
        counts = candidate_fast.process_frame.last_counts
        self.assertEqual(counts["verify_seed"], candidate_fast.SEED_BUDGET)
        self.assertGreater(counts.get("subwindow_retry_skipped", 0), 0)

    def test_vegetation_lookup_matches_reference_formula(self):
        g, r = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
        reference = (g - r > 9) & (g > 1.08 * r)
        self.assertTrue(np.array_equal(
            reference, g > candidate_fast._GREEN_OVER_RED[r]
        ))

    def test_unreadable_and_tiny_inputs_return_no_tracks(self):
        self.assertEqual(candidate_fast.process_image("/nonexistent.png"), [])
        self.assertEqual(candidate_fast.process_frame(np.zeros((16, 16, 3), np.uint8)), [])

    @unittest.skipUnless(GOLDEN.is_file(), "golden outputs not generated")
    def test_outputs_match_golden(self):
        """Regenerate with bench_fast.py --write-golden only for intended changes."""
        cv2.setNumThreads(1)
        self.assertEqual(bench_fast.main(["--check-golden", str(GOLDEN)]), 0)


if __name__ == "__main__":
    unittest.main()
