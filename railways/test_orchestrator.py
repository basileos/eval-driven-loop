"""Search policy regression tests; no generation calls are made."""
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
from railways import evaluator, orchestrator as loop


def metrics(iou=0.2, fpr=0.0):
    return loop.CandidateMetrics(iou, iou, 0.5 if iou else 0, iou, fpr, 1.0 if iou else 0, 1.0)


def evaluation(value):
    return True, "METRICS: " + " ".join(f"{k}={v}" for k, v in asdict(value).items()), ""


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        limit = patch.object(loop, "MAX_FALSE_POSITIVE_RATE", 0.1)
        limit.start()
        self.addCleanup(limit.stop)

    def test_acceptance_matrix(self):
        cases = [
            (metrics(0.9, 1/3), metrics(0.13, 2/3), False),
            (metrics(0.3, 0.05), metrics(0.2, 0), True),
            (metrics(0.3, 4/41), metrics(0.2, 0), True),
            (metrics(0.3, 5/41), metrics(0.2, 0), False),
            (metrics(0.3, 0.1), metrics(0.2, 0.5), True),
            (metrics(0.2, 0), metrics(0.2, 0.1), False),
            (metrics(0.2), metrics(0.2), False),
            (metrics(0.19), metrics(0.2, 0.1), False),
            (metrics(0), None, False),
            (metrics(0.9, 0.5), None, False),
            (metrics(0.2), None, True),
        ]
        for candidate, retained, expected in cases:
            with self.subTest(candidate=candidate, retained=retained):
                self.assertEqual(loop.retention_decision(candidate, retained)[0], expected)

    def test_empty_feedback_keeps_negative_rejection(self):
        context = loop.ContextManager()
        context.record(1, 0, "UNCHANGED", "", "DIAGNOSIS: EMPTY DETECTOR")
        self.assertIn("returning [] on negative images", context.next_move())
        self.assertNotIn("MUST return at least one", context.next_move())
        self.assertNotIn("unless the REQUIRED NEXT MOVE", loop.SYSTEM_PROMPT)

    def test_payload_uses_fixed_gate_even_after_zero_fpr(self):
        payload = loop.build_prompt_payload("", "", "", "", "", "", "", "", "test", metrics(0.2, 0))
        self.assertIn("FPR <= 0.100000", payload)
        self.assertIn("FPR may increase", payload)
        self.assertIn("strictly improve", payload)
        self.assertNotIn("temporarily overrides", payload)

    def test_rejection_feedback_targets_false_positives(self):
        context = loop.ContextManager()
        context.record(1, 0.9, "ARCHIVED", "", "FPR_REJECTED: above ceiling")
        self.assertIn("false-positive acceptance gate", context.next_move())

    def run_search(self, attempts):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate_script.py"
            initial = "def process_image(path):\n    return []\n"
            candidate.write_text(initial)
            codes = [f"# attempt {index}\n{initial}" for index in range(len(attempts))]
            output = io.StringIO()
            with (
                patch.object(loop, "CANDIDATE_FILE", str(candidate)),
                patch.object(loop, "RUNS_DIR", str(Path(directory) / "runs")),
                patch.object(loop, "MAX_ITERATIONS", len(attempts)),
                patch.object(loop, "execute_evaluator", return_value=evaluation(metrics(0.13, 2/3))),
                patch.object(loop, "call_codex_cli", side_effect=codes) as generate,
                patch.object(loop, "evaluate_candidate_source", side_effect=[evaluation(m) for m in attempts]),
                redirect_stdout(output),
            ):
                loop.run_orchestrator("test")
            run = next((Path(directory) / "runs").iterdir())
            self.assertEqual((run / "baseline.py").read_text(), initial)
            self.assertEqual(len(list(run.glob("iteration_*.py"))), generate.call_count)
            self.assertEqual(len(list(run.glob("iteration_*.stdout.txt"))), generate.call_count)
            decisions = sorted(run.glob("*.decision.json"))
            self.assertEqual(len(decisions), generate.call_count)
            self.assertIn("accepted", json.loads(decisions[0].read_text()))
            self.assertTrue((run / "archive.json").exists())
            return candidate.read_text(), initial, codes, generate.call_count, output.getvalue()

    def test_rejected_target_score_does_not_stop_or_replace_file(self):
        retained, _, codes, calls, output = self.run_search([metrics(0.9, 1/3), metrics(0.3, 0)])
        self.assertEqual(calls, 2)
        self.assertEqual(retained, codes[1].strip())
        self.assertNotIn("Target score achieved", output)
        self.assertIn("exceeds acceptance ceiling", output)

    def test_only_rejected_attempts_preserve_baseline(self):
        retained, initial, _, _, output = self.run_search([metrics(0.9, 1/3)])
        self.assertEqual(retained, initial)
        self.assertIn("FPR GATE NOT MET", output)

    def test_eligible_target_stops_search(self):
        retained, _, codes, calls, output = self.run_search([metrics(0.9, 0), metrics(0.95, 0)])
        self.assertEqual(calls, 1)
        self.assertEqual(retained, codes[0].strip())
        self.assertIn("Target score achieved", output)


class SearchRuntimeTests(unittest.TestCase):
    def test_timeout_feedback_takes_priority_over_stagnation(self):
        context = loop.ContextManager()
        for iteration in (1, 2):
            context.record(iteration, 0, "CRASH", "", "EVALUATION_TIMEOUT")
        self.assertIn("computational complexity", context.next_move())
        self.assertIn("native resolution", context.next_move())

    def test_failed_final_branch_still_resumes_from_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.py"
            initial = "def process_image(path):\n    return []\n"
            candidate.write_text(initial)
            codes = [f"# attempt {i}\n{initial}" for i in range(3)]
            with (
                patch.object(loop, "CANDIDATE_FILE", str(candidate)),
                patch.object(loop, "RUNS_DIR", str(Path(directory) / "runs")),
                patch.object(loop, "MAX_ITERATIONS", 3),
                patch.object(loop, "MAX_STAGNANT_ITERATIONS", 1),
                patch.object(loop, "MAX_REPAIR_ATTEMPTS", 0),
                patch.object(loop, "BRANCH_STRATEGIES", [("test_branch", "test")]),
                patch.object(loop, "request_architect_guidance", return_value="test"),
                patch.object(loop, "execute_evaluator", return_value=evaluation(metrics(.1, .8))),
                patch.object(loop, "call_codex_cli", side_effect=codes) as generate,
                patch.object(loop, "evaluate_candidate_source", side_effect=[
                    evaluation(metrics(.2, .7)), (False, "", "EVALUATION_TIMEOUT"),
                    evaluation(metrics(.2, .7)),
                ]),
                redirect_stdout(io.StringIO()) as output,
            ):
                loop.run_orchestrator("test")
            self.assertEqual(generate.call_count, 3)
            prompt = generate.call_args_list[2].args[0]
            self.assertIn(codes[0].strip(), prompt)
            self.assertNotIn(codes[1].strip(), prompt)
            self.assertIn("balanced search parent (iteration 1;", output.getvalue())
            self.assertEqual(candidate.read_text(), initial)

    def test_parent_selects_iteration_seven_not_baseline_or_collapsed_nine(self):
        archive = loop.ParetoArchive()
        for iteration, iou, positive_iou, fpr in (
            (0, .062484, .2567, .829268),
            (7, .086087, .2146, .829268),
            (9, .000340, .0007, .634146),
        ):
            archive.add(loop.CandidateRecord(
                str(iteration), replace(metrics(iou, fpr), mean_positive_iou=positive_iou),
                iteration, "test", "", None,
            ))
        self.assertEqual(archive.best_positive().iteration, 0)
        self.assertEqual(archive.best_search_parent().iteration, 7)
        self.assertFalse(loop.retention_decision(archive.best_search_parent().metrics, archive.best_positive().metrics)[0])

    def test_all_branches_avoid_high_fpr_baseline_using_latest_run_scores(self):
        archive = loop.ParetoArchive()
        for iteration, iou, positive_iou, fpr in (
            (0, .062484, .256720, .829268),
            (2, .215121, .198705, .073171),
            (3, .269147, .254550, .487805),
            (4, .187950, .135892, 0),
        ):
            archive.add(loop.CandidateRecord(
                str(iteration), replace(metrics(iou, fpr), mean_positive_iou=positive_iou),
                iteration, "test", "", None,
            ))
        self.assertEqual(archive.best_positive().iteration, 0)
        for strategy, _ in loop.BRANCH_STRATEGIES:
            with self.subTest(strategy=strategy):
                self.assertEqual(archive.parent_for(strategy).iteration, 4)

    def test_branch_can_use_baseline_when_it_is_the_only_parent(self):
        archive = loop.ParetoArchive()
        baseline = loop.CandidateRecord("baseline", metrics(.2, .8), 0, "baseline", "", None)
        archive.add(baseline)
        for strategy, _ in loop.BRANCH_STRATEGIES:
            self.assertIs(archive.parent_for(strategy), baseline)

    def test_outer_deadline_tracks_dataset_deadline(self):
        self.assertEqual(loop.EVAL_TIMEOUT, evaluator.POOL_TIMEOUT_SECONDS + 60)

    def test_outer_timeout_stops_group_and_preserves_output(self):
        process = Mock()
        process.communicate.side_effect = [subprocess.TimeoutExpired("test", 2), ("partial output", "partial error")]
        with (
            patch.object(loop.subprocess, "Popen", return_value=process) as launch,
            patch.object(loop, "terminate_process_group") as stop,
            patch.object(loop, "Spinner", return_value=nullcontext()),
        ):
            success, stdout, stderr = loop.execute_evaluator(timeout=2)
        self.assertFalse(success)
        self.assertEqual(stdout, "partial output")
        self.assertIn("partial error", stderr)
        self.assertIn("EVALUATION_TIMEOUT", stderr)
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        stop.assert_called_once_with(process)

    def test_pool_timeout_reports_completed_and_queued_work(self):
        pool = Mock()
        ready = Mock()
        ready.ready.return_value = True
        ready.get.return_value = {"image": "done.png", "error": None}
        pending = Mock()
        pending.ready.return_value = False
        pool.apply_async.side_effect = [ready, pending]
        with (
            patch.object(evaluator.mp, "get_context") as context,
            patch.object(evaluator.time, "perf_counter", side_effect=[0, 2]),
            patch.object(evaluator, "POOL_TIMEOUT_SECONDS", 1),
        ):
            context.return_value.Pool.return_value = pool
            with self.assertRaisesRegex(TimeoutError, "Completed 1/2 images.*1 queued or running"):
                evaluator.evaluate_samples_parallel([("done.png", "", []), ("pending.png", "", [])])
        pool.terminate.assert_called_once()
        pool.join.assert_called_once()

    def test_failed_source_and_logs_survive_run(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.py"
            initial = "def process_image(path):\n    return []\n"
            candidate.write_text(initial)
            bad_code = "def process_image(path):\n    return nine\n"
            with (
                patch.object(loop, "CANDIDATE_FILE", str(candidate)),
                patch.object(loop, "RUNS_DIR", str(Path(directory) / "runs")),
                patch.object(loop, "MAX_ITERATIONS", 1),
                patch.object(loop, "MAX_REPAIR_ATTEMPTS", 0),
                patch.object(loop, "execute_evaluator", side_effect=[evaluation(metrics()), (False, "partial", "NameError: nine")]),
                patch.object(loop, "call_codex_cli", return_value=bad_code),
                redirect_stdout(io.StringIO()),
            ):
                loop.run_orchestrator("test")
            run = next((Path(directory) / "runs").iterdir())
            self.assertEqual((run / "iteration_01_local_refinement.py").read_text(), bad_code.strip())
            self.assertEqual((run / "iteration_01_local_refinement.stderr.txt").read_text(), "NameError: nine")
            self.assertEqual(candidate.read_text(), initial)


class ConservativeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.candidate = self.root / "candidate_script.py"
        self.run = self.root / "run"
        self.run.mkdir()
        for name, value in (("CANDIDATE_FILE", str(self.candidate)), ("MAX_FALSE_POSITIVE_RATE", .1)):
            setting = patch.object(loop, name, value)
            setting.start()
            self.addCleanup(setting.stop)
        self.saved = Path(loop.conservative_candidate_path())

    def record(self, iou, fpr, iteration=0):
        return loop.CandidateRecord(
            f"# version {iteration}\ndef process_image(path):\n    return []\n",
            metrics(iou, fpr), iteration, "test", "", None,
        )

    def test_only_positive_iou_zero_fpr_can_be_saved(self):
        with redirect_stdout(io.StringIO()):
            for iou, fpr in ((0, 0), (.3, .001)):
                self.assertIsNone(loop.preserve_conservative_candidate(self.record(iou, fpr), None, str(self.run)))
                self.assertFalse(self.saved.exists())

    def test_conservative_improves_independently_without_overwriting_main(self):
        self.candidate.write_text("# main detector")
        first = self.record(.2, 0)
        better = self.record(.28, 0, 1)
        with redirect_stdout(io.StringIO()):
            best = loop.preserve_conservative_candidate(first, None, str(self.run))
            best = loop.preserve_conservative_candidate(better, best, str(self.run))
            for rejected in (self.record(.27, 0, 2), self.record(.28, 0, 3), self.record(.4, .01, 4)):
                self.assertIs(loop.preserve_conservative_candidate(rejected, best, str(self.run)), better)
        self.assertEqual(self.saved.read_text(), better.code)
        self.assertEqual(self.candidate.read_text(), "# main detector")
        metadata = json.loads((self.run / "conservative_best.metrics.json").read_text())
        self.assertEqual(metadata["metrics"]["dataset_iou"], .28)
        self.assertEqual(metadata["code_hash"], better.code_hash)

    def test_identical_baseline_reuses_evaluation(self):
        baseline = self.record(.25, 0)
        self.saved.write_text(baseline.code)
        with patch.object(loop, "execute_evaluator") as evaluate, redirect_stdout(io.StringIO()):
            self.assertIs(loop.load_conservative_candidate(baseline, str(self.run)), baseline)
        evaluate.assert_not_called()
        self.assertEqual((self.run / "conservative_baseline.py").read_text(), baseline.code)

    def test_existing_alternative_revalidated_across_runs(self):
        saved = self.record(.28, 0, 1)
        baseline = self.record(.3, .05, 2)
        self.saved.write_text(saved.code)
        with (
            patch.object(loop, "execute_evaluator", return_value=evaluation(saved.metrics)) as evaluate,
            redirect_stdout(io.StringIO()),
        ):
            reloaded = loop.load_conservative_candidate(baseline, str(self.run))
            self.assertEqual(reloaded.metrics.dataset_iou, .28)
            best = loop.preserve_conservative_candidate(self.record(.27, 0, 3), reloaded, str(self.run))
        self.assertIs(best, reloaded)
        evaluate.assert_called_once()
        self.assertEqual(self.saved.read_text(), saved.code)

    def test_changed_dataset_or_failed_revalidation_does_not_claim_zero_fpr(self):
        saved = self.record(.28, 0, 1)
        self.saved.write_text(saved.code)
        for result in (evaluation(metrics(.28, .05)), evaluation(metrics(0, 0)), (False, "", "timeout")):
            with (
                self.subTest(result=result),
                patch.object(loop, "execute_evaluator", return_value=result),
                redirect_stdout(io.StringIO()),
            ):
                self.assertIsNone(loop.load_conservative_candidate(None, str(self.run)))
                self.assertEqual(self.saved.read_text(), saved.code)

    def test_search_keeps_baseline_then_better_zero_fpr_despite_main_rejection(self):
        baseline = self.record(.25, 0)
        versions = [self.record(.3, 4/41, 1), self.record(.28, 0, 2), self.record(.27, 0, 3)]
        self.candidate.write_text(baseline.code)

        def generate(*args, **kwargs):
            # The zero-FPR baseline is saved before any potentially replacing generation.
            if generate.index == 0:
                self.assertEqual(self.saved.read_text(), baseline.code)
            record = versions[generate.index]
            generate.index += 1
            return record.code

        generate.index = 0
        with (
            patch.object(loop, "RUNS_DIR", str(self.root / "runs")),
            patch.object(loop, "MAX_ITERATIONS", 3),
            patch.object(loop, "execute_evaluator", return_value=evaluation(baseline.metrics)),
            patch.object(loop, "call_codex_cli", side_effect=generate),
            patch.object(loop, "evaluate_candidate_source", side_effect=[evaluation(v.metrics) for v in versions]),
            redirect_stdout(io.StringIO()),
        ):
            loop.run_orchestrator("test")
        self.assertEqual(self.candidate.read_text(), versions[0].code.strip())
        self.assertEqual(self.saved.read_text(), versions[1].code.strip())


class RepairTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prefix = str(self.root / "iteration_06_segment_graph_spline")
        self.source = "def process_image(path):\n    return undefined_name\n"
        self.fixed = "def process_image(path):\n    return []\n"
        setting = patch.object(loop, "MAX_REPAIR_ATTEMPTS", 1)
        setting.start()
        self.addCleanup(setting.stop)

    def run_attempt(self, code=None):
        with redirect_stdout(io.StringIO()):
            return loop.evaluate_with_repairs(
                self.source if code is None else code, self.prefix,
                strategy_name="segment_graph_spline", strategy_instruction="Fit splines to fragment chains.",
                task="Detect railways", retained_metrics=metrics(.215, .073),
                parent_montage="parent.json",
            )

    def test_runtime_error_repaired_with_same_source_architecture_and_gate(self):
        with (
            patch.object(loop, "evaluate_candidate_source", side_effect=[
                (False, "partial", "cv2.error: m == src2.rows"), evaluation(metrics(.3, .05)),
            ]) as evaluate,
            patch.object(loop, "call_codex_cli", return_value=self.fixed) as generate,
        ):
            result = self.run_attempt()
        self.assertTrue(result.success)
        self.assertEqual(result.code, self.fixed.strip())
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(generate.call_count, 1)
        prompt = generate.call_args.args[0]
        for expected in (self.source, "m == src2.rows", "segment_graph_spline", "Fit splines", "FPR <= 0.100000", "Preserve its architecture"):
            self.assertIn(expected, prompt)
        self.assertEqual(generate.call_args.kwargs["model"], loop.CODEX_MODEL)
        self.assertEqual(generate.call_args.kwargs["diagnostic_montage"], "parent.json")
        self.assertEqual(Path(self.prefix + ".py").read_text(), self.source)
        self.assertIn("m == src2.rows", Path(self.prefix + ".stderr.txt").read_text())
        self.assertEqual(Path(result.artifact_prefix + ".py").read_text(), self.fixed.strip())
        self.assertTrue(result.artifact_prefix.endswith("_repair_01"))
        self.assertEqual(evaluate.call_args.kwargs["diagnostic_montage_path"], result.artifact_prefix + ".json")

    def test_syntax_error_is_repairable(self):
        with (
            patch.object(loop, "execute_evaluator", return_value=evaluation(metrics(.3))) as evaluate,
            patch.object(loop, "call_codex_cli", return_value=self.fixed) as generate,
        ):
            result = self.run_attempt("def process_image(:")
        self.assertTrue(result.success)
        self.assertEqual(evaluate.call_count, 1)
        self.assertIn("SyntaxError", generate.call_args.args[0])
        self.assertIn("def process_image(:", generate.call_args.args[0])

    def test_repeated_errors_stop_at_repair_budget(self):
        with (
            patch.object(loop, "evaluate_candidate_source", return_value=(False, "", "EVALUATION_TIMEOUT")) as evaluate,
            patch.object(loop, "call_codex_cli", return_value=self.fixed) as generate,
        ):
            result = self.run_attempt()
        self.assertFalse(result.success)
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(generate.call_count, 1)

    def test_successful_but_ineligible_result_does_not_trigger_repair(self):
        with (
            patch.object(loop, "evaluate_candidate_source", return_value=evaluation(metrics(.4, .9))),
            patch.object(loop, "call_codex_cli") as generate,
        ):
            result = self.run_attempt()
        self.assertTrue(result.success)
        generate.assert_not_called()

    def test_long_source_reaches_evaluation_without_length_repair(self):
        source = "# comment\n" * 10000 + self.fixed
        with (
            patch.object(loop, "execute_evaluator", return_value=evaluation(metrics())) as evaluate,
            patch.object(loop, "call_codex_cli") as generate,
            patch.dict(loop.os.environ, {"RAILWAYS_MAX_CANDIDATE_LINES": "50"}),
        ):
            result = self.run_attempt(source)
            self.assertTrue(result.success)
            self.assertEqual(result.code, source)
            self.assertEqual(evaluate.call_count, 1)
            generate.assert_not_called()
        self.assertEqual(Path(self.prefix + ".py").read_text(), source)
        self.assertFalse(hasattr(loop, "MAX_CANDIDATE_LINES"))

    def test_long_repair_is_evaluated_and_prompts_have_no_line_limit(self):
        repaired = "# repaired stage\n" * 1200 + self.fixed
        with (
            patch.object(loop, "evaluate_candidate_source", side_effect=[
                (False, "", "NameError: undefined_name"), evaluation(metrics()),
            ]) as evaluate,
            patch.object(loop, "call_codex_cli", return_value=repaired) as generate,
        ):
            result = self.run_attempt()
        self.assertTrue(result.success)
        self.assertEqual(result.code, repaired.strip())
        self.assertEqual(evaluate.call_count, 2)
        payload = loop.build_prompt_payload("", "", "", "", "", "", "", "", "test")
        for prompt in (loop.SYSTEM_PROMPT, payload, generate.call_args.args[0]):
            self.assertNotRegex(prompt, r"(?i)line (?:cap|limit)|(?:at most|no more than) .*lines|\d+ lines")

    def test_generation_failure_saved_and_propagated_without_retry(self):
        with (
            patch.object(loop, "evaluate_candidate_source", return_value=(False, "", "crash")),
            patch.object(loop, "call_codex_cli", side_effect=RuntimeError("model unavailable")) as generate,
        ):
            with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                self.run_attempt()
        self.assertEqual(generate.call_count, 1)
        self.assertIn("model unavailable", Path(self.prefix + "_repair_01.generation_error.txt").read_text())

    def test_repaired_candidate_only_retained_after_acceptance_gate(self):
        for fpr, accepted in ((.05, True), (.8, False)):
            with self.subTest(fpr=fpr):
                candidate = self.root / "candidate.py"
                initial = "# retained baseline\n" + self.fixed
                candidate.write_text(initial)
                with (
                    patch.object(loop, "CANDIDATE_FILE", str(candidate)),
                    patch.object(loop, "RUNS_DIR", str(self.root / "runs")),
                    patch.object(loop, "MAX_ITERATIONS", 1),
                    patch.object(loop, "execute_evaluator", return_value=evaluation(metrics(.2, .1))),
                    patch.object(loop, "call_codex_cli", side_effect=[self.source, self.fixed]),
                    patch.object(loop, "evaluate_candidate_source", side_effect=[
                        (False, "", "NameError: undefined_name"), evaluation(metrics(.3, fpr)),
                    ]),
                    redirect_stdout(io.StringIO()),
                ):
                    loop.run_orchestrator("test")
                self.assertEqual(candidate.read_text(), self.fixed.strip() if accepted else initial)
        decisions = list((self.root / "runs").glob("*/*_repair_01.decision.json"))
        self.assertEqual(len(decisions), 2)
        self.assertEqual({json.loads(path.read_text())["accepted"] for path in decisions}, {False, True})

    def test_branch_repair_finishes_before_next_branch_and_keeps_iteration(self):
        candidate = self.root / "candidate.py"
        candidate.write_text(self.fixed)
        codes = [f"# version {index}\n{self.fixed}" for index in range(4)]
        with (
            patch.object(loop, "CANDIDATE_FILE", str(candidate)),
            patch.object(loop, "RUNS_DIR", str(self.root / "runs")),
            patch.object(loop, "MAX_ITERATIONS", 3),
            patch.object(loop, "MAX_STAGNANT_ITERATIONS", 1),
            patch.object(loop, "BRANCH_STRATEGIES", [("segment_graph_spline", "spline"), ("multiscale_path", "path")]),
            patch.object(loop, "request_architect_guidance", return_value="guidance"),
            patch.object(loop, "execute_evaluator", return_value=evaluation(metrics(.2, .1))),
            patch.object(loop, "call_codex_cli", side_effect=codes) as generate,
            patch.object(loop, "evaluate_candidate_source", side_effect=[
                evaluation(metrics(.18, .09)), (False, "", "cv2.error: matrix shape"),
                evaluation(metrics(.3, .05)), evaluation(metrics(.29, .09)),
            ]),
            redirect_stdout(io.StringIO()) as output,
        ):
            loop.run_orchestrator("test")
        self.assertEqual([call.kwargs["role"] for call in generate.call_args_list], [
            "local_refinement implementer", "segment_graph_spline implementer",
            "segment_graph_spline repair 1", "multiscale_path implementer",
        ])
        self.assertIn(codes[1].strip(), generate.call_args_list[2].args[0])
        self.assertIn("Iteration 3/3 [DIVERGE:multiscale_path from iteration 2]", output.getvalue())
        self.assertEqual(candidate.read_text(), codes[2].strip())
        run = next((self.root / "runs").iterdir())
        self.assertTrue((run / "iteration_02_segment_graph_spline_repair_01.decision.json").exists())


if __name__ == "__main__":
    unittest.main()
