"""Timing telemetry and coarse-to-fine prompt policy; no model/API calls."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from railways import evaluator, orchestrator as loop


class TimingTests(unittest.TestCase):
    def test_latency_percentiles_and_parallel_throughput_are_distinct(self):
        rows = [dict(elapsed=0.1, stage_timings_seconds={'lsd': 0.04}),
                dict(elapsed=0.3)]
        metrics, stages = evaluator.summarize_timings(rows, 0.5)
        self.assertAlmostEqual(metrics['mean_image_seconds'], 0.2)
        self.assertAlmostEqual(metrics['p50_image_seconds'], 0.2)
        self.assertAlmostEqual(metrics['p95_image_seconds'], 0.29)
        self.assertEqual(metrics['max_image_seconds'], 0.3)
        self.assertAlmostEqual(metrics['serial_fps_estimate'], 5)
        self.assertAlmostEqual(metrics['evaluation_fps'], 4)
        self.assertEqual(stages['lsd']['samples'], 1)
        self.assertEqual(stages['lsd']['mean_seconds'], 0.04)

    def test_zero_duration_and_empty_results_are_finite(self):
        for rows in ([], [dict(elapsed=0)]):
            metrics, stages = evaluator.summarize_timings(rows, 0)
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
            self.assertEqual(stages, {})

    def test_optional_invalid_stage_values_are_ignored(self):
        def process(_):
            return []
        self.assertEqual(evaluator.read_stage_timings(process), {})
        process.last_timings = {'lsd': 0.05, 'negative': -1, 'nan': float('nan'),
                                'infinite': float('inf'), 'flag': True, 'text': '2',
                                'bad name': 1, 123: 1, 'nested': {}}
        self.assertEqual(evaluator.read_stage_timings(process), {'lsd': 0.05})
        process.last_timings = []
        self.assertEqual(evaluator.read_stage_timings(process), {})

    def test_worker_clears_stale_stages_even_on_empty_detection(self):
        seen = []
        def process(_):
            seen.append(dict(getattr(process, 'last_timings', {})))
            if len(seen) == 1:
                process.last_timings = {'lsd': 0.025}
            return []
        with (
            patch.object(evaluator, '_WORKER_PROCESS_IMAGE', process),
            patch.object(evaluator, '_WORKER_INIT_ERROR', None),
            patch.object(evaluator, 'read_image_shape', return_value=(48, 64)),
        ):
            first = evaluator.evaluate_sample(('first.png', 'first.png', []))
            second = evaluator.evaluate_sample(('second.png', 'second.png', []))
        self.assertEqual(seen, [{}, {}])
        self.assertIsNone(first['error'])
        self.assertEqual(first['stage_timings_seconds'], {'lsd': 0.025})
        self.assertEqual(second['stage_timings_seconds'], {})
        self.assertEqual(evaluator._format_debug_metrics(first)['stage_timings_seconds'], {'lsd': 0.025})
        first['elapsed'], second['elapsed'] = 0.1, 0.3
        # Exercise the actual evaluator output and orchestrator parser contract.
        output = io.StringIO()
        with (
            patch.object(evaluator, 'load_railway_ground_truth', return_value={'first.png': [], 'second.png': []}),
            patch.object(evaluator, 'evaluate_samples_parallel', return_value=([first, second], 2, 0.5)),
            redirect_stdout(output),
        ):
            evaluator.evaluate(candidate_path='unused.py', val_dir='unused')
        text = output.getvalue()
        self.assertIn('TIMING_STAGES:', text)
        self.assertIn('elapsed_seconds=0.100000', text)
        parsed = loop.parse_metrics(text)
        self.assertEqual(parsed.mean_image_seconds, 0.2)
        self.assertEqual(parsed.p50_image_seconds, 0.2)
        self.assertEqual(parsed.p95_image_seconds, 0.29)
        self.assertEqual(parsed.evaluation_fps, 4)
        self.assertEqual(parsed.serial_fps_estimate, 5)

    def test_stage_telemetry_survives_real_worker_processes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidate = root / 'candidate.py'
            candidate.write_text('import time\n'
                                 'def process_image(path):\n'
                                 '    started = time.perf_counter()\n'
                                 '    process_image.last_timings = {"lsd": time.perf_counter() - started}\n'
                                 '    return []\n')
            samples = []
            for i in range(2):
                path = root / f'{i}.png'
                cv2.imwrite(str(path), np.zeros((48, 64, 3), np.uint8))
                samples.append((path.name, str(path), []))
            with patch.object(evaluator, 'MAX_WORKERS', 2):
                rows, workers, wall = evaluator.evaluate_samples_parallel(samples, str(candidate))
            self.assertEqual(workers, 2)
            self.assertGreater(wall, 0)
            self.assertTrue(all('lsd' in row['stage_timings_seconds'] for row in rows))

    def test_legacy_metrics_still_parse_without_invented_timings(self):
        text = ('METRICS: dataset_iou=0.2 mean_positive_iou=0.3 positive_coverage=0.4 '
                'polygon_precision=0.5 false_positive_rate=0.0 detected_positive_rate=0.6 wall_seconds=1.0')
        result = loop.parse_metrics(text)
        self.assertEqual(result.dataset_iou, 0.2)
        self.assertIsNone(result.mean_image_seconds)
        self.assertIsNone(result.p95_image_seconds)

    def test_speed_does_not_relax_acceptance_gate(self):
        previous = loop.CandidateMetrics(.5, .5, .5, .9, 0, .8, 10, mean_image_seconds=1)
        faster = replace(previous, wall_seconds=1, mean_image_seconds=.1)
        self.assertFalse(loop.retention_decision(faster, previous)[0])
        self.assertTrue(loop.retention_decision(replace(faster, dataset_iou=.51), previous)[0])


class PromptPolicyTests(unittest.TestCase):
    def check_policy(self, prompt):
        for phrase in ('1280 x 720', '1920 x 1080', 'sx = W_native / W_small',
                       'sy = H_native / H_small', 'fit_section', 'ORIGINAL pixels',
                       'process_image.last_timings', 'time.perf_counter()'):
            self.assertIn(phrase, prompt)
        for banned in ('Never downsample before', 'Never solve a timeout by downsampling',
                       'without\ndownsampling', 'Preserve full-resolution processing'):
            self.assertNotIn(banned, prompt)

    def test_main_and_architect_share_resolution_and_timing_policy(self):
        self.check_policy(loop.SYSTEM_PROMPT)
        payload = loop.build_prompt_payload('', '', '', '', '', '', '', '', 'test')
        self.check_policy(payload)
        metrics = loop.CandidateMetrics(.2, .2, .3, .5, 0, .5, 1)
        record = loop.CandidateRecord('code', metrics, 1, 'test', 'feedback', None)
        archive = loop.ParetoArchive()
        archive.add(record)
        with patch.object(loop, 'call_codex_cli', return_value='plan') as generate:
            loop.request_architect_guidance(record, archive, 'feedback')
        self.check_policy(generate.call_args.args[0])

    def test_repairs_share_policy_and_save_attempt_timing(self):
        with tempfile.TemporaryDirectory() as folder:
            prefix = str(Path(folder) / 'iteration')
            with (
                patch.object(loop, 'MAX_REPAIR_ATTEMPTS', 1),
                patch.object(loop, 'evaluate_candidate_source', side_effect=[(False, '', 'timeout'), (True, '', '')]),
                patch.object(loop, 'call_codex_cli', return_value='def process_image(path): return []') as generate,
                redirect_stdout(io.StringIO()),
            ):
                result = loop.evaluate_with_repairs('def process_image(path): return []', prefix,
                    strategy_name='test', strategy_instruction='test', task='test',
                    retained_metrics=None, parent_montage=None)
            self.assertTrue(result.success)
            self.check_policy(generate.call_args.args[0])
            for path in (Path(prefix + '.timing.json'), Path(prefix + '_repair_01.timing.json')):
                data = json.loads(path.read_text())
                self.assertGreaterEqual(data['evaluation_attempt_wall_seconds'], 0)


if __name__ == '__main__':
    unittest.main()
