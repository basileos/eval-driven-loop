"""Integration checks using a tiny local video; no camera/network required."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from railways.run_candidate_video import VideoResultsWriter, build_parser, main


class VideoRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="railway-video-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.video = self.root / "input.avi"
        self.output = self.root / "results"
        writer = cv2.VideoWriter(
            str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120)
        )
        self.assertTrue(writer.isOpened())
        try:
            for index in range(5):
                writer.write(np.full((120, 160, 3), index * 30, dtype=np.uint8))
        finally:
            writer.release()

    def run_video(self, *options, process_image=None):
        def predict(path):
            self.assertEqual(cv2.imread(path).shape, (120, 160, 3))
            return [{"left_rail": [[20, 10], [20, 100]],
                     "right_rail": [[40, 10], [40, 100]]}]

        with (
            patch("railways.run_candidate_video.load_candidate_functions",
                  return_value=(process_image or predict, None)),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return main([str(self.video), "-o", str(self.output), "--workers", "1", *options])

    def read_results(self, directory=None):
        return json.loads(((directory or self.output) / 'results.json').read_text())

    def read_frames(self, directory=None):
        return self.read_results(directory)['frames']

    def read_frame(self, index):
        return next(row for row in self.read_frames() if row['frame_index'] == index)

    def test_every_frame_has_predictions_overlay_and_timestamp(self):
        self.assertEqual(self.run_video(), 0)
        records = self.read_frames()
        self.assertEqual([row["frame_index"] for row in records], list(range(5)))
        for index, row in enumerate(records):
            self.assertAlmostEqual(row["timestamp_seconds"], index / 10.0)
            self.assertEqual(row["predicted_tracks"], 1)
            self.assertIsNone(row["error"])
            overlay = cv2.imread(str(self.output / row["debug_image"]))
            self.assertEqual(overlay.shape[1], 160)
            self.assertGreater(overlay.shape[0], 120)
        self.assertEqual(self.read_results()['summary']["stop_reason"], "end_of_video")
        self.assertEqual(self.read_results()['summary']["processed_frames"], 5)
        self.assertEqual(sorted(p.name for p in self.output.glob('*.json*')), ['results.json'])
        self.assertTrue(self.read_results()['finalized'])

    def test_sampling_and_limit_keep_original_indices(self):
        self.assertEqual(self.run_video("--every-n-frames", "2", "--max-frames", "2"), 0)
        self.assertEqual([row['frame_index'] for row in self.read_frames()], [0, 2])
        self.assertEqual(self.read_results()['summary']["stop_reason"], "max_frames")
        self.assertAlmostEqual(self.read_frame(2)["timestamp_seconds"], 0.2)

    def test_time_range_keeps_original_indices_and_excludes_end(self):
        decoded_values = []

        def predict(path):
            decoded_values.append(round(float(cv2.imread(path).mean())))
            return []

        self.assertEqual(self.run_video("--start-time", "00:00.2", "--end-time", "00:00:00.4",
                                        process_image=predict), 0)
        self.assertEqual(decoded_values, [60, 90])
        records = self.read_frames()
        self.assertEqual([row["frame_index"] for row in records], [2, 3])
        self.assertAlmostEqual(records[0]["timestamp_seconds"], 0.2)
        self.assertAlmostEqual(records[1]["timestamp_seconds"], 0.3)
        self.assertEqual(self.read_results()['summary']["stop_reason"], "end_time")
        self.assertEqual(self.read_results()['summary']["decoded_frames"], 3)
        self.assertEqual(self.read_results()['run']["start_time_seconds"], 0.2)
        self.assertEqual(self.read_results()['run']["end_time_seconds"], 0.4)

    def test_fractional_start_skips_earlier_frame_and_sampling_starts_in_range(self):
        self.assertEqual(self.run_video("--start-time", "0.05", "--every-n-frames", "2"), 0)
        self.assertEqual([row['frame_index'] for row in self.read_frames()], [1, 3])

    def test_end_only(self):
        self.assertEqual(self.run_video("--end-time", "0.2"), 0)
        self.assertEqual(self.read_results()['summary']["processed_frames"], 2)
        self.assertEqual(self.read_results()['summary']["stop_reason"], "end_time")

    def test_range_combines_with_max_frames(self):
        self.assertEqual(self.run_video("--start-time", "0.1", "--end-time", "0.5",
                                        "--every-n-frames", "2", "--max-frames", "1"), 0)
        self.assertEqual([row['frame_index'] for row in self.read_frames()], [1])
        self.assertEqual(self.read_results()['summary']["stop_reason"], "max_frames")

    def test_empty_range_beyond_eof_has_no_results(self):
        self.assertEqual(self.run_video("--start-time", "10"), 1)
        self.assertEqual(self.read_results()['summary']["processed_frames"], 0)
        self.assertEqual(self.read_frames(), [])

    def test_invalid_ranges_rejected(self):
        for start, end in [("0.2", "0.1"), ("0.2", "0.2")]:
            with self.subTest(start=start, end=end), self.assertRaises(SystemExit):
                self.run_video("--start-time", start, "--end-time", end)
        self.assertFalse(self.output.exists())

    def test_timestamp_formats_and_invalid_values(self):
        parser = build_parser()
        for value, expected in [("90.5", 90.5), ("01:30.5", 90.5), ("01:02:03.5", 3723.5)]:
            self.assertEqual(parser.parse_args(["input.avi", "--start-time", value]).start_time, expected)
        for value in ["-1", "nan", "inf", "1:60", "1:2:3:4", "1.5:20", "abc"]:
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["input.avi", "--start-time", value])

    def test_time_range_rejects_camera_and_stream_before_opening(self):
        for source in ["0", "https://example.test/video.mp4"]:
            with patch("railways.run_candidate_video.cv2.VideoCapture") as capture:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main([source, "--end-time", "10"])
                capture.assert_not_called()

    def test_candidate_failure_is_saved_and_later_frames_continue(self):
        def predict(path):
            if Path(path).stem == "frame_00000001":
                raise RuntimeError("test failure")
            return []

        self.assertEqual(self.run_video(process_image=predict), 1)
        self.assertIn("test failure", self.read_frame(1)["error"])
        self.assertEqual(self.read_frame(4)["predictions"], [])
        self.assertEqual(self.read_results()['summary']["failed_frames"], 1)
        self.assertEqual(self.read_results()['summary']["processed_frames"], 5)

    def test_interrupt_retains_completed_frames_and_summary(self):
        def predict(path):
            if Path(path).stem == "frame_00000001":
                raise KeyboardInterrupt
            return []

        self.assertEqual(self.run_video(process_image=predict), 130)
        self.assertEqual(self.read_results()['summary']["stop_reason"], "interrupted")
        self.assertEqual(self.read_results()['summary']["processed_frames"], 1)
        self.assertEqual(len(self.read_frames()), 1)
        self.assertTrue(self.read_results()['finalized'])

    def test_existing_results_are_not_overwritten(self):
        self.assertEqual(self.run_video("--max-frames", "1"), 0)
        before = (self.output / "results.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "must be empty"):
            self.run_video()
        self.assertEqual((self.output / "results.json").read_bytes(), before)

    def test_actual_candidate_loads_and_processes_video(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = main([str(self.video), "-o", str(self.output), "--max-frames", "1", "--workers", "1"])
        self.assertEqual(result, 0)
        self.assertEqual(self.read_frame(0)["predictions"], [])

    def test_array_candidate_does_not_write_temporary_frames(self):
        # process_frame is optional: generated candidates only promise process_image.
        def process_frame(frame):
            return []

        def process_image(path):
            raise AssertionError("array processing must not use the file-based function")

        with (
            patch("railways.run_candidate_video.load_candidate_functions",
                  return_value=(process_image, process_frame)),
            patch("railways.run_candidate_video.tempfile.TemporaryDirectory",
                  side_effect=AssertionError("array processing must not use disk")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = main([str(self.video), "-o", str(self.output), "--workers", "1", "--no-overlays"])
        self.assertEqual(result, 0)
        self.assertEqual(len(self.read_frames()), 5)
        self.assertEqual(sorted(p.name for p in self.output.iterdir()), ['results.json'])
        self.assertEqual(list(self.output.glob("*.jpg")), [])
        self.assertIsNone(self.read_frame(0)["debug_image"])

    def test_parallel_matches_serial_and_keeps_order(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            main([str(self.video), "-o", str(self.output), "--workers", "1"])
            parallel = self.root / "parallel"
            result = main([str(self.video), "-o", str(parallel), "--workers", "2"])
        self.assertEqual(result, 0)
        records = self.read_frames(parallel)
        self.assertEqual([row["frame_index"] for row in records], list(range(5)))
        for row in records:
            expected = self.read_frame(row['frame_index'])
            self.assertEqual(row["predictions"], expected["predictions"])
            self.assertEqual(row["timestamp_seconds"], expected["timestamp_seconds"])

    def test_parallel_respects_range(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = main([str(self.video), "-o", str(self.output), "--workers", "2",
                           "--start-time", "0.1", "--end-time", "0.4", "--every-n-frames", "2"])
        self.assertEqual(result, 0)
        records = self.read_frames()
        self.assertEqual([row["frame_index"] for row in records], [1, 3])
        self.assertEqual(self.read_results()['summary']["stop_reason"], "end_time")

    def test_streaming_keeps_all_file_frames_and_throttles_empty_search(self):
        predict = Mock(return_value=[])
        self.assertEqual(self.run_video('--streaming', '--workers', '4', '--no-overlays', process_image=predict), 0)
        self.assertEqual(predict.call_count, 1)
        self.assertEqual(self.read_results()['summary']['processed_frames'], 5)
        self.assertEqual(self.read_results()['summary']['streaming_modes'], {'search': 1, 'search_wait': 4})
        self.assertFalse(self.read_results()['run']['live_latest_frame'])
        self.assertEqual(self.read_results()['run']['workers'], 1)
        self.assertEqual(self.read_frame(0)['streaming']['mode'], 'search')
        self.assertEqual(self.read_frame(1)['streaming']['mode'], 'search_wait')

    def test_streaming_live_limits_saved_results_not_capture(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 25.0

        def frames(capture, args, is_file, fps, summary, started_at, first_frame):
            self.assertIsNone(args.max_frames)
            for i in range(40):
                summary['decoded_frames'] += 1
                yield np.zeros((120, 160, 3), np.uint8), {
                    'frame_index': i, 'timestamp_seconds': i / 25,
                    'timestamp_source': 'elapsed_monotonic',
                }
            summary['stop_reason'] = 'stream_read_failed'

        with (
            patch('railways.run_candidate_video.cv2.VideoCapture', return_value=capture),
            patch('railways.run_candidate_video.read_frames', side_effect=frames),
            patch('railways.run_candidate_video.load_candidate_functions', return_value=(lambda _: [], lambda _: [])),
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()),
        ):
            result = main(['mock://live', '--streaming', '--max-frames', '1', '--no-overlays',
                           '--output-dir', str(self.output)])
        self.assertEqual(result, 0)
        summary = self.read_results()['summary']
        self.assertEqual(summary['processed_frames'], 1)
        self.assertEqual(summary['stop_reason'], 'max_frames')
        self.assertIn('dropped_pending_frames', summary)
        self.assertTrue(self.read_results()['run']['live_latest_frame'])
        rows = self.read_frames()
        self.assertEqual(len(rows), 1)
        self.assertGreaterEqual(rows[0]['application_latency_seconds'], rows[0]['processing_seconds'])
        capture.release.assert_called_once()


class ConsolidatedWriterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'results.json'
        self.writer = VideoResultsWriter(self.path, {'source': 'test.mp4'})
        self.addCleanup(self.writer.output.close)

    def read(self):
        return json.loads(self.path.read_text())

    def test_valid_json_after_every_append_and_finish(self):
        self.assertEqual(self.read()['frames'], [])
        self.assertFalse(self.read()['finalized'])
        for index in range(20):
            self.writer.append({'frame_index': index, 'predictions': [], 'note': 'рейки'})
            result = self.read()
            self.assertEqual([r['frame_index'] for r in result['frames']], list(range(index + 1)))
            self.assertEqual(result['summary']['processed_frames'], index + 1)
            self.assertFalse(result['finalized'])
        self.writer.finish({'stop_reason': 'end_of_video', 'processed_frames': 20})
        self.assertTrue(self.read()['finalized'])
        self.assertEqual(self.read()['run'], {'source': 'test.mp4'})
        self.assertEqual(self.read()['summary']['stop_reason'], 'end_of_video')

    def test_serialization_error_preserves_previous_records(self):
        self.writer.append({'frame_index': 0})
        previous = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.writer.append({'bad': float('nan')})
        self.assertEqual(self.path.read_bytes(), previous)
        self.writer.finish({'stop_reason': 'error'})
        self.assertEqual(self.read()['frames'], [{'frame_index': 0}])

    def test_interrupt_during_append_rolls_back_incomplete_record(self):
        self.writer.append({'frame_index': 0})
        before = self.path.read_bytes()
        real_write = self.writer._write_footer
        calls = 0

        def interrupt_once(footer):
            nonlocal calls
            calls += 1
            if calls == 1:
                self.writer.output.write(footer[:5])
                raise KeyboardInterrupt
            real_write(footer)

        with patch.object(self.writer, '_write_footer', side_effect=interrupt_once):
            with self.assertRaises(KeyboardInterrupt):
                self.writer.append({'frame_index': 1})
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.writer.count, 1)
        self.writer.finish({'stop_reason': 'interrupted', 'processed_frames': 1})
        self.assertTrue(self.read()['finalized'])

    def test_existing_document_cannot_be_overwritten(self):
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            VideoResultsWriter(self.path, {})
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
