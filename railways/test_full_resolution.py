"""Check native detector input and native visual feedback without model calls."""
from contextlib import nullcontext, redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from railways import evaluator, orchestrator


class FullResolutionTests(unittest.TestCase):
    def test_coarse_proposals_still_receive_native_input_and_score_native_coordinates(self):
        source = np.zeros((2160, 3840, 3), dtype=np.uint8)
        seen = []
        tracks = [{"left_rail": [[1200, 600], [1200, 1800]],
                   "right_rail": [[1240, 600], [1240, 1800]]}]
        def process_image(path):
            original = cv2.imread(path)
            small = cv2.resize(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), (1280, 720))
            seen.append((original.shape, small.shape))
            return tracks
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'native.png'
            self.assertTrue(cv2.imwrite(str(path), source))
            with (
                patch.object(evaluator, '_WORKER_PROCESS_IMAGE', process_image),
                patch.object(evaluator, '_WORKER_INIT_ERROR', None),
            ):
                result = evaluator.evaluate_sample((path.name, str(path), tracks))
        self.assertIsNone(result['error'])
        self.assertEqual(seen, [((2160, 3840, 3), (720, 1280))])
        self.assertEqual(result['image_shape'], (2160, 3840))
        self.assertEqual(result['score'], 1.0)

    def test_context_preserves_dimensions_and_separate_originals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "val_dataset"
            dataset.mkdir()
            dimensions = [(1080, 1920), (700, 1000)]
            tracks = [{"left_rail": [[50, 100], [50, 600]],
                       "right_rail": [[70, 100], [70, 600]]}]
            rows = []
            for index, (height, width) in enumerate(dimensions):
                name = f"sample_{index}.png"
                pixels = np.zeros((height, width, 3), dtype=np.uint8)
                pixels[:, ::2] = 255  # One-pixel features must not be resampled.
                self.assertTrue(cv2.imwrite(str(dataset / name), pixels))
                actual_tracks = tracks if index == 0 else []
                if actual_tracks:
                    (dataset / name).with_suffix(".json").write_text(json.dumps({"tracks": actual_tracks}))
                rows.append({
                    "image": name, "ground_truth_tracks": len(actual_tracks),
                    "predicted_tracks": 1, "score": 0.5 if actual_tracks else 0,
                    "ground_truth_data": actual_tracks, "prediction_data": tracks,
                    "gt_area": 100 if actual_tracks else 0, "pred_area": 100,
                    "centroid_dx": 0, "centroid_dy": 0, "coverage": 0.5, "precision": 0.5,
                })
            reference = root / "reference.json"
            diagnostic = root / "diagnostic.json"
            with (
                patch.object(orchestrator, "BASE_DIR", str(root)),
                patch.object(cv2, "resize", side_effect=AssertionError("No thumbnails")),
            ):
                self.assertEqual(orchestrator.build_reference_montage(str(reference)), str(reference))
                self.assertTrue(evaluator.write_diagnostic_montage(rows, str(dataset), str(diagnostic)))
            for manifest in (reference, diagnostic):
                entries = json.loads(manifest.read_text())["images"]
                self.assertEqual(len(entries), 4)
                for original, overlay in zip(entries[::2], entries[1::2]):
                    source = cv2.imread(original["path"])
                    rendered = cv2.imread(overlay["path"])
                    self.assertEqual(source.shape, rendered.shape)
                    np.testing.assert_array_equal(source[300:400, 400:600], rendered[300:400, 400:600])
            context = orchestrator.load_model_context_images(str(reference), str(diagnostic))
            self.assertEqual(len(context), 4)  # Parent diagnostics replace static references.

            process = Mock(returncode=0)
            process.communicate.return_value = ("OK", "")
            with (
                patch.object(orchestrator, "build_reference_montage", side_effect=AssertionError("Do not duplicate references")),
                patch.object(orchestrator.subprocess, "Popen", return_value=process) as launch,
                patch.object(orchestrator, "Spinner", return_value=nullcontext()),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(orchestrator.call_codex_cli("test", diagnostic_montage=str(diagnostic)), "OK")
            command = launch.call_args.args[0]
            attached = [command[i + 1] for i, token in enumerate(command) if token == "--image"]
            self.assertEqual(attached, [entry["path"] for entry in context])
            self.assertIn("Image 4:", process.communicate.call_args.kwargs["input"])
            self.assertEqual(process.communicate.call_args.kwargs["timeout"], orchestrator.CODEX_TIMEOUT)

    def test_generation_timeout_stops_process_and_reports_cli_progress(self):
        process = Mock()
        process.communicate.side_effect = orchestrator.subprocess.TimeoutExpired(
            "codex", 600, stderr=b"waiting for server response"
        )
        stderr = io.StringIO()
        with (
            patch.object(orchestrator, "build_reference_montage", return_value=None),
            patch.object(orchestrator.subprocess, "Popen", return_value=process),
            patch.object(orchestrator, "terminate_process_group") as terminate,
            patch.object(orchestrator, "Spinner", return_value=nullcontext()),
            redirect_stdout(io.StringIO()), redirect_stderr(stderr),
        ):
            with self.assertRaisesRegex(RuntimeError, "600s limit"):
                orchestrator.call_codex_cli("test", timeout=600)
        terminate.assert_called_once_with(process)
        self.assertIn("waiting for server response", stderr.getvalue())

    def test_diagnostic_bundle_is_balanced_and_bounded(self):
        rows = [dict(
            image=f"sample_{index}.png", ground_truth_tracks=int(index < 5),
            predicted_tracks=1, score=index / 10 if index < 5 else 0,
            ground_truth_data=[], prediction_data=[], gt_area=100 if index < 5 else 0,
            pred_area=100 + index, centroid_dx=0, centroid_dy=0, coverage=0, precision=0,
        ) for index in range(10)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = np.zeros((120, 200, 3), dtype=np.uint8)
            for row in rows:
                cv2.imwrite(str(root / row["image"]), frame)
            manifest = root / "diagnostics.json"
            self.assertTrue(evaluator.write_diagnostic_montage(rows, str(root), str(manifest)))
            images = json.loads(manifest.read_text())["images"]
            self.assertEqual(len(images), 8)
            selected = [Path(entry["path"]).name for entry in images[::2]]
            self.assertEqual(selected, ["sample_0.png", "sample_4.png", "sample_9.png", "sample_8.png"])


if __name__ == "__main__":
    unittest.main()
