import os
import re
import signal
import subprocess
import sys
import tempfile
import itertools
import json
import hashlib
import math
import threading
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass

if __package__:
    from .evaluation_config import POOL_TIMEOUT_SECONDS, PROCESS_TIMEOUT_SECONDS
else:
    from evaluation_config import POOL_TIMEOUT_SECONDS, PROCESS_TIMEOUT_SECONDS

# Configuration
MAX_ITERATIONS = 10
TARGET_SCORE = 0.8
MAX_FALSE_POSITIVE_RATE = float(os.environ.get("RAILWAYS_MAX_FALSE_POSITIVE_RATE", "0.10"))
if not math.isfinite(MAX_FALSE_POSITIVE_RATE) or not 0.0 <= MAX_FALSE_POSITIVE_RATE <= 1.0:
    raise ValueError("RAILWAYS_MAX_FALSE_POSITIVE_RATE must be between 0 and 1.")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_FILE = os.path.join(BASE_DIR, "candidate_script.py")
EVALUATOR_FILE = os.path.join(BASE_DIR, "evaluator.py")
RUNS_DIR = os.environ.get("RAILWAYS_RUNS_DIR", os.path.join(BASE_DIR, "runs"))
CODEX_TIMEOUT = max(1, int(os.environ.get("RAILWAYS_CODEX_TIMEOUT", "600")))
CODEX_MODEL = os.environ.get("RAILWAYS_CODEX_MODEL", "gpt-5.6-luna")
CODEX_REASONING_EFFORT = os.environ.get(
    "RAILWAYS_CODEX_REASONING_EFFORT", "low"
)
ARCHITECT_MODEL = os.environ.get(
    "RAILWAYS_ARCHITECT_MODEL", "gpt-5.6-sol"
)
ARCHITECT_REASONING_EFFORT = os.environ.get(
    "RAILWAYS_ARCHITECT_REASONING_EFFORT", "medium"
)
ARCHITECT_TIMEOUT = max(
    1, int(os.environ.get("RAILWAYS_ARCHITECT_TIMEOUT", "600"))
)
MAX_CANDIDATE_LINES = max(
    50, int(os.environ.get("RAILWAYS_MAX_CANDIDATE_LINES", "1000"))
)
MAX_REPAIR_ATTEMPTS = max(0, int(os.environ.get("RAILWAYS_MAX_REPAIR_ATTEMPTS", "1")))
MAX_STAGNANT_ITERATIONS = max(
    1, int(os.environ.get("RAILWAYS_MAX_STAGNANT_ITERATIONS", "3"))
)
MAX_DISRUPTION_ROUNDS = max(
    0, int(os.environ.get("RAILWAYS_MAX_DISRUPTION_ROUNDS", "1"))
)
PARETO_ARCHIVE_SIZE = max(
    2, int(os.environ.get("RAILWAYS_PARETO_ARCHIVE_SIZE", "4"))
)
EVAL_TIMEOUT = PROCESS_TIMEOUT_SECONDS
MAX_LOG_BYTES = 9000  # Preserve geometry diagnostics in the next prompt

SYSTEM_PROMPT = """You are an expert Computer Vision engineer.
Write Python code implementing strictly the function:
`process_image(img_path: str) -> list[dict[str, object]]`

Rules:
1. Detect acceptable railway tracks in aerial/drone imagery.
2. Return one dictionary per track, pairing the two rail boundary polylines:
   [{"left_rail": [[x1, y1], [x2, y2], ...],
     "right_rail": [[x1, y1], [x2, y2], ...]}, ...]
3. Each rail must contain at least two finite pixel coordinates within the image bounds.
4. The evaluator closes the area between each rail pair and scores dataset-level filled polygon IoU: total intersection pixels / total union pixels.
5. Return [] if the image cannot be loaded or no acceptable railway is detected. Never force a corridor merely because an image decoded successfully. Require railway evidence and reject unrelated parallel roads, field boundaries, and vegetation edges.
6. Return ONLY runnable Python code inside ```python ... ``` fences.
7. Do not include CLI handlers or an `if __name__ == "__main__"` block.
8. Process images at their original pixel resolution. Never downsample before line, edge, texture, or rail detection: thin rails lose their evidence. Native-resolution overlapping tiles are allowed if coordinates are mapped back correctly and duplicate tracks are merged. Optimize runtime without reducing spatial resolution.
9. Keep process_image deterministic and independent across images; evaluation runs images concurrently in separate processes.
10. Do not inspect the filesystem, run commands, edit files, or use tools. All required context is in the task payload.
11. Respect the configured line cap in the task payload. Keep Python concise and simplify oversized code instead of extending it.
12. This environment uses OpenCV 5. Normalize HoughLinesP output with `np.asarray(lines).reshape(-1, 4)`; do not assume the OpenCV 4-only `(N, 1, 4)` shape.
13. Do not wrap process_image in `except Exception: return []` or another blanket error handler. Return [] for an explicit image-decode failure, but let unexpected programming errors raise so the evaluator can report them.
14. The target rails can curve and may be partly occluded. A disruption branch must produce piecewise or curved polylines rather than reducing every track to one extended straight segment.
15. Reject undersized detections: the shorter predicted rail and the overall corridor span must each be at least 5% of the full image diagonal. Tiny parallel fragments are clutter, not an acceptable railway.
16. Bound runtime at native resolution: prune clutter fragments before pairwise matching, use spatial indexing or vectorized operations, and avoid rescanning the entire image for every segment pair. Never solve a timeout by downsampling.
17. Preserve meaningful per-track confidence metadata when available: optional "confidence" in [0, 1] and "confidence_method" identifying its calculation. This is an uncalibrated evidence score, not a probability of correctness. For "verified_patch_fraction_v1", use the number of final verification sections passing the railway-evidence check divided by the number checked. If changing the calculation, change the method name; omit confidence rather than inventing a constant score. Confidence is diagnostic only and does not change polygon-IoU scoring.
"""

ARCHITECT_SYSTEM_PROMPT = """You are the architecture planner for an eval-driven computer-vision search.
Do not write implementation code. Diagnose why the current detector plateaus and give concise, concrete implementation guidance for four deliberately different classical-OpenCV branches. Respect the supplied runtime and output contract. Focus on representation changes, not threshold tweaks. Return plain text under 900 words."""

TASK_DESCRIPTION = (
    "Detect acceptable railway tracks in aerial/drone imagery using classical OpenCV methods and return paired left/right rail polylines for polygon-IoU evaluation."
)

BRANCH_STRATEGIES = (
    (
        "segment_graph_spline",
        "Build a graph of compatible LSD/Hough fragments, chain separated and "
        "occluded evidence, fit smooth piecewise curves or splines, and infer a "
        "paired corridor. Do not select only one straight segment pair.",
    ),
    (
        "corridor_segmentation",
        "Replace line-pair selection with multiscale color/texture corridor "
        "segmentation, elongated-component cleanup, skeleton or center-path "
        "extraction, and two smooth boundaries.",
    ),
    (
        "multiscale_path",
        "Construct a multiscale ridge/edge-orientation cost field and recover "
        "long smooth paths with dynamic programming or graph search, allowing "
        "curvature and short occlusions before forming the paired rails.",
    ),
    (
        "lsd_piecewise_repair",
        "Reuse useful parts of the current LSD detector but group many local "
        "segments into a piecewise curved railway, correct overextension and "
        "offset using the visual diagnostics, enforce the 5%-diagonal minimum "
        "corridor size, and add conservative negative gating.",
    ),
)


@dataclass(frozen=True)
class CandidateMetrics:
    dataset_iou: float
    mean_positive_iou: float
    positive_coverage: float
    polygon_precision: float
    false_positive_rate: float
    detected_positive_rate: float
    wall_seconds: float

    def summary(self) -> str:
        return (
            f"IoU={self.dataset_iou:.6f}, posIoU={self.mean_positive_iou:.4f}, "
            f"coverage={self.positive_coverage:.4f}, "
            f"precision={self.polygon_precision:.4f}, "
            f"FPR={self.false_positive_rate:.3f}"
        )


@dataclass
class CandidateRecord:
    code: str
    metrics: CandidateMetrics
    iteration: int
    strategy: str
    feedback: str
    diagnostic_montage: str | None

    @property
    def code_hash(self) -> str:
        return hashlib.sha256(self.code.encode("utf-8")).hexdigest()[:12]


def retention_decision(
    candidate: CandidateMetrics,
    retained: CandidateMetrics | None,
) -> tuple[bool, str]:
    """Use a fixed FPR ceiling and require strictly better dataset IoU."""
    if candidate.dataset_iou <= 0:
        return False, "zero IoU; an empty or non-overlapping detector cannot be retained"
    ceiling = MAX_FALSE_POSITIVE_RATE
    if candidate.false_positive_rate > ceiling:
        return False, (
            f"FPR {candidate.false_positive_rate:.6f} exceeds acceptance ceiling "
            f"{ceiling:.6f} (fixed configured maximum)"
        )
    if retained is None or candidate.dataset_iou > retained.dataset_iou:
        return True, "IoU improved and the false-positive gate passed"
    return False, "requires strictly higher dataset IoU"


def acceptance_guidance(retained: CandidateMetrics | None) -> str:
    ceiling = MAX_FALSE_POSITIVE_RATE
    baseline = (
        f"Retained candidate: {retained.summary()}. " if retained else "No valid retained baseline. "
    )
    return (
        baseline
        + f"File replacement requires positive dataset IoU and FPR <= {ceiling:.6f}. "
        "FPR is the fraction of negative images with any detection, not polygon area. "
        "Dataset IoU must strictly improve over the retained candidate; equal or lower IoU is not accepted. "
        "FPR may increase relative to the retained candidate as long as it stays within the fixed ceiling. "
        "The highest-IoU positive-overlap zero-FPR candidate is saved separately as a conservative alternative. "
        "Candidates outside the gate "
        "may inform further search but cannot replace the retained file or meet the target. "
        "Do not return an empty detector or force predictions to evade these criteria."
    )


class ParetoArchive:
    """Keep non-dominated geometry/classification candidates during one run."""

    def __init__(self, max_size: int = PARETO_ARCHIVE_SIZE):
        self.max_size = max_size
        self.records: list[CandidateRecord] = []

    @staticmethod
    def _dominates(left: CandidateRecord, right: CandidateRecord) -> bool:
        a = left.metrics
        b = right.metrics
        no_worse = (
            a.dataset_iou >= b.dataset_iou
            and a.mean_positive_iou >= b.mean_positive_iou
            and a.false_positive_rate <= b.false_positive_rate
        )
        strictly_better = (
            a.dataset_iou > b.dataset_iou
            or a.mean_positive_iou > b.mean_positive_iou
            or a.false_positive_rate < b.false_positive_rate
        )
        return no_worse and strictly_better

    def add(self, record: CandidateRecord) -> bool:
        self.records = [
            existing
            for existing in self.records
            if existing.code_hash != record.code_hash
        ]
        if any(self._dominates(existing, record) for existing in self.records):
            return False

        self.records = [
            existing
            for existing in self.records
            if not self._dominates(record, existing)
        ]
        self.records.append(record)
        self._trim()
        return any(item.code_hash == record.code_hash for item in self.records)

    def _trim(self) -> None:
        if len(self.records) <= self.max_size:
            return

        leaders = [self.best_score(), self.best_search_parent(), self.best_positive()]
        leaders.append(
            max(
                self.records,
                key=lambda record: (
                    record.metrics.polygon_precision,
                    record.metrics.dataset_iou,
                ),
            )
        )

        retained = []
        seen = set()
        for record in leaders + sorted(
            self.records,
            key=lambda record: record.metrics.dataset_iou,
            reverse=True,
        ):
            if record.code_hash in seen:
                continue
            seen.add(record.code_hash)
            retained.append(record)
            if len(retained) == self.max_size:
                break
        self.records = retained

    def best_score(self) -> CandidateRecord:
        return max(self.records, key=lambda record: record.metrics.dataset_iou)

    def best_positive(self) -> CandidateRecord:
        return max(
            self.records,
            key=lambda record: (
                record.metrics.mean_positive_iou,
                record.metrics.dataset_iou,
            ),
        )

    def best_search_parent(self) -> CandidateRecord:
        """Prefer lower FPR without choosing an almost-empty detector.

        This is a search heuristic, not the deployment acceptance gate. Keep at
        least half of both archive-leading dataset IoU and positive-image IoU.
        """
        best_iou = self.best_score().metrics.dataset_iou
        best_positive = self.best_positive().metrics.mean_positive_iou
        viable = [
            record for record in self.records
            if record.metrics.dataset_iou > 0
            and record.metrics.dataset_iou >= 0.5 * best_iou
            and record.metrics.mean_positive_iou >= 0.5 * best_positive
        ]
        return min(
            viable,
            key=lambda record: (record.metrics.false_positive_rate, -record.metrics.dataset_iou),
        ) if viable else self.best_score()

    def parent_for(self, strategy: str) -> CandidateRecord:
        # Architecture diversity comes from branch instructions, not from
        # seeding some branches with a high-FPR positive-IoU leader.
        return self.best_search_parent()

    def render(self) -> str:
        if not self.records:
            return "Pareto archive is empty."
        lines = [
            "| Source | Strategy | Dataset IoU | Positive IoU | Coverage | "
            "Precision | FPR |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for record in sorted(
            self.records,
            key=lambda item: item.metrics.dataset_iou,
            reverse=True,
        ):
            source = "baseline" if record.iteration == 0 else str(record.iteration)
            metrics = record.metrics
            lines.append(
                f"| {source} | {record.strategy} | {metrics.dataset_iou:.6f} | "
                f"{metrics.mean_positive_iou:.4f} | "
                f"{metrics.positive_coverage:.4f} | "
                f"{metrics.polygon_precision:.4f} | "
                f"{metrics.false_positive_rate:.3f} |"
            )
        return "\n".join(lines)


def build_reference_montage(output_path: str) -> str | None:
    """Write a manifest of separate native-resolution originals and annotations.

    The legacy function name is retained, but no montage or resizing is used.
    """
    try:
        import cv2
        import numpy as np

        val_dir = os.path.join(BASE_DIR, "val_dataset")
        positives = []
        negatives = []

        for filename in sorted(os.listdir(val_dir)):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue

            image_path = os.path.join(val_dir, filename)
            annotation_path = os.path.splitext(image_path)[0] + ".json"
            tracks = []
            if os.path.exists(annotation_path):
                with open(annotation_path, "r") as annotation_file:
                    tracks = json.load(annotation_file).get("tracks", [])

            (positives if tracks else negatives).append(
                (filename, image_path, tracks)
            )

        if not positives:
            return None

        # Balance classes and bound image count, without reducing resolution.
        samples = [positives[index] for index in sorted({0, len(positives) - 1})]
        if negatives:
            samples.extend(negatives[index] for index in sorted({0, len(negatives) - 1}))
        samples = samples[:4]

        attachments = []
        output_dir = os.path.splitext(output_path)[0] + "_images"
        os.makedirs(output_dir, exist_ok=True)
        for filename, image_path, tracks in samples:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            panel = image.copy()
            panel_height, panel_width = image.shape[:2]
            scale_x = scale_y = 1.0

            for track in tracks:
                rails = []
                for rail_name in ("left_rail", "right_rail"):
                    rail = [
                        [round(point[0] * scale_x), round(point[1] * scale_y)]
                        for point in track.get(rail_name, [])
                    ]
                    if len(rail) >= 2:
                        rails.append(np.asarray(rail, dtype=np.int32))
                if len(rails) == 2:
                    overlay = panel.copy()
                    polygon = np.concatenate((rails[0], rails[1][::-1]))
                    cv2.fillPoly(overlay, [polygon], (0, 215, 255))
                    panel = cv2.addWeighted(overlay, 0.22, panel, 0.78, 0.0)
                    cv2.polylines(panel, [rails[0]], False, (0, 0, 255), 3)
                    cv2.polylines(panel, [rails[1]], False, (255, 80, 0), 3)

            label = f"{'POSITIVE TARGET' if tracks else 'NEGATIVE'}: {filename}"
            cv2.rectangle(panel, (0, 0), (panel_width, 28), (0, 0, 0), -1)
            cv2.putText(
                panel,
                label,
                (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            panel_path = os.path.abspath(os.path.join(output_dir, f"{len(attachments):02d}.png"))
            if not cv2.imwrite(panel_path, panel, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError(f"Could not write {panel_path}")
            attachments.extend([
                {"path": os.path.abspath(image_path), "label": f"Original {label}"},
                {"path": panel_path, "label": f"Annotated {label}"},
            ])

        if not attachments:
            return None
        with open(output_path, "w") as output:
            json.dump({"images": attachments}, output, indent=2)
        return output_path
    except Exception as error:
        print(f"Warning: Could not prepare full-resolution references: {error}", file=sys.stderr)
        return None


def load_model_context_images(
    reference_path: str | None,
    diagnostic_path: str | None,
) -> list[dict[str, str]]:
    """Use parent diagnostics, or references when no diagnostics exist.

    Both bundles contain originals and annotated copies. Sending both repeats
    their role and previously ballooned the payload to hundreds of megabytes.
    """
    images = []
    seen = set()
    for manifest_path in (diagnostic_path or reference_path,):
        if not manifest_path:
            continue
        with open(manifest_path) as manifest:
            entries = json.load(manifest)["images"]
        for entry in entries:
            path = os.path.abspath(entry["path"])
            if path not in seen:
                if not os.path.isfile(path):
                    raise FileNotFoundError(path)
                images.append({"path": path, "label": entry["label"]})
                seen.add(path)
    return images


class Spinner:
    """Displays a terminal spinner with an elapsed time counter in a background thread."""

    def __init__(self, message: str = "Waiting for Codex CLI..."):
        self.message = message
        self.stop_running = False
        self.thread = None

    def _spin(self):
        spinner = itertools.cycle(
            ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        )
        start_time = time.time()
        while not self.stop_running:
            elapsed = int(time.time() - start_time)
            sys.stdout.write(f"\r{self.message} {next(spinner)} ({elapsed}s)")
            sys.stdout.flush()
            time.sleep(0.1)
        # Clear the spinner line upon exit
        sys.stdout.write("\r" + " " * (len(self.message) + 20) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        self.stop_running = False
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_running = True
        if self.thread:
            self.thread.join()


class ContextManager:
    """Maintains a lightweight iteration history to prevent context window bloat."""

    def __init__(self):
        self.history = []

    def record(
        self,
        iteration: int,
        score: float,
        status: str,
        notes: str,
        feedback: str,
    ):
        self.history.append({
            "iter": iteration,
            "score": score,
            "status": status,
            "notes": notes,
            "feedback": feedback,
        })

    def get_best_score(self) -> float:
        return max([e["score"] for e in self.history], default=0.0)

    def render_scoreboard(self) -> str:
        if not self.history:
            return "No prior iterations."
        lines = ["| Iter | Score | Status | Notes |", "|---|---|---|---|"]
        for e in self.history:
            lines.append(
                f"| {e['iter']} | {e['score']:.4f} | {e['status']} | {e['notes']} |"
            )
        return "\n".join(lines)

    def get_latest_feedback(self) -> str:
        if not self.history:
            return "Initial iteration. Generate baseline solution."
        return self.history[-1]["feedback"]

    def non_improvement_streak(self) -> int:
        streak = 0
        for entry in reversed(self.history):
            if entry["status"] in {"UNCHANGED", "REGRESSED", "CRASH", "REJECTED", "ARCHIVED"}:
                streak += 1
            else:
                break
        return streak

    def next_move(self) -> str:
        feedback = self.get_latest_feedback()
        if "EVALUATION_TIMEOUT" in feedback:
            return (
                "The last candidate exceeded the dataset runtime budget. Reduce "
                "computational complexity, bound segment-pair searches and avoid "
                "repeated full-image operations. Keep native resolution and genuine "
                "railway detection; do not bypass evaluation by returning []."
            )
        if "EMPTY DETECTOR" in feedback:
            return (
                "The last attempt missed all positive railway examples. Diagnose "
                "which evidence extraction or rejection stage removed real tracks, "
                "and recover supported rail pairs on positive examples. Continue "
                "returning [] on negative images and whenever railway evidence is "
                "insufficient. Do not add an unconditional detection fallback."
            )
        if "FPR_REJECTED" in feedback:
            return (
                "The previous candidate failed the false-positive acceptance gate. "
                "Use the negative-image diagnostics to reject unrelated parallel "
                "structures while preserving real railway overlap. Higher IoU alone "
                "cannot compensate for failing the FPR ceiling."
            )
        if self.non_improvement_streak() >= 2:
            return (
                "The score is stagnant. Do not make another minor threshold tweak "
                "or return functionally equivalent code; replace the weakest stage "
                "with a materially different detection or geometry strategy."
            )
        return (
            "Use the per-image diagnostics to make one concrete, measurable change "
            "to railway discrimination, detection recall, or polygon localization "
            "while respecting the false-positive acceptance gate."
        )


def extract_code(llm_response: str) -> str:
    """Extracts raw code from Markdown python blocks."""
    match = re.search(r"```python\n(.*?)```", llm_response, re.DOTALL)
    return match.group(1).strip() if match else llm_response.strip()


def truncate_log(log_text: str, max_bytes: int = MAX_LOG_BYTES) -> str:
    """Prevents large stack traces or verbose outputs from blowing up prompt tokens."""
    if len(log_text) > max_bytes:
        return log_text[:max_bytes] + "\n... [Log Truncated]"
    return log_text if log_text else "No output details."


def terminate_process_group(process: subprocess.Popen, grace_seconds: float = 2.0):
    """Terminate a subprocess and all descendants started in its process group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return

    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    # The group leader can exit before its workers. Kill remaining descendants
    # too, otherwise their inherited stdout/stderr pipes can keep communicate open.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def call_codex_cli(
    prompt: str,
    system_prompt: str = "",
    timeout: int = CODEX_TIMEOUT,
    *,
    model: str = CODEX_MODEL,
    reasoning_effort: str = CODEX_REASONING_EFFORT,
    diagnostic_montage: str | None = None,
    role: str = "implementer",
) -> str:
    """Executes Codex CLI non-interactively in ephemeral mode."""
    full_prompt = (
        f"System Instructions:\n{system_prompt}\n\nTask Payload:\n{prompt}"
    )

    cmd = [
        "codex",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "--model",
        model,
        "--config",
        f'model_reasoning_effort="{reasoning_effort}"',
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "-",
    ]

    with tempfile.TemporaryDirectory(prefix="railways-codex-") as generation_dir:
        cmd[2:2] = ["--cd", generation_dir]
        reference_path = None
        if not diagnostic_montage:
            reference_path = build_reference_montage(
                os.path.join(generation_dir, "railway_reference.json")
            )
        context_images = load_model_context_images(reference_path, diagnostic_montage)
        if context_images:
            full_prompt += "\n\nNative-resolution attachments (originals and overlays are separate):\n"
            for index, entry in enumerate(context_images, start=1):
                cmd[-1:-1] = ["--image", entry["path"]]
                full_prompt += f"Image {index}: {entry['label']}\n"
        image_bytes = sum(os.path.getsize(entry["path"]) for entry in context_images)
        print(
            f"GENERATION: {role} model={model} reasoning={reasoning_effort} "
            f"timeout={timeout}s images={len(context_images)} "
            f"image_files={image_bytes / (1024 ** 2):.1f} MiB (native resolution)",
            flush=True,
        )
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=generation_dir,
            start_new_session=True,
        )

        try:
            message = (
                f"Running {role} via {model}/{reasoning_effort} "
                f"(timeout {timeout}s)"
            )
            with Spinner(message):
                stdout, stderr = process.communicate(
                    input=full_prompt, timeout=timeout
                )
        except subprocess.TimeoutExpired as error:
            terminate_process_group(process)
            detail = error.stderr or error.output or ""
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", errors="replace")
            if detail.strip():
                print(f"Last CLI output before timeout:\n{detail[-2000:]}", file=sys.stderr)
            raise RuntimeError(
                f"Codex CLI exceeded its {timeout}s limit; its process group was "
                "stopped. Adjust RAILWAYS_CODEX_TIMEOUT (or "
                "RAILWAYS_ARCHITECT_TIMEOUT for architecture calls) if needed."
            ) from error
        except KeyboardInterrupt:
            terminate_process_group(process)
            raise

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode,
                cmd,
                output=stdout,
                stderr=stderr,
            )

    return stdout.strip()


def execute_evaluator(
    timeout: float | None = EVAL_TIMEOUT,
    candidate_path: str = CANDIDATE_FILE,
    diagnostic_montage_path: str | None = None,
) -> tuple[bool, str, str]:
    """Runs evaluator.py with visual progress spinner and optional timeout."""
    with Spinner("Running evaluation suite"):
        command = [sys.executable, EVALUATOR_FILE, candidate_path]
        if diagnostic_montage_path:
            command.extend([
                os.path.join(BASE_DIR, "val_dataset"),
                diagnostic_montage_path,
            ])
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=BASE_DIR,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_process_group(process)
            stdout, stderr = process.communicate()
            return False, stdout.strip(), (
                f"EVALUATION_TIMEOUT: evaluator process exceeded {timeout}s "
                f"(dataset budget {POOL_TIMEOUT_SECONDS}s); process group stopped.\n{stderr}"
            ).strip()
        except BaseException:
            terminate_process_group(process)
            raise
    return process.returncode == 0, stdout.strip(), stderr.strip()


def build_prompt_payload(
    scoreboard: str,
    archive: str,
    active_code: str,
    diagnostics: str,
    next_move: str,
    strategy_name: str,
    strategy_instruction: str,
    architect_guidance: str,
    task: str,
    retained_metrics: CandidateMetrics | None = None,
) -> str:
    """Constructs a fixed-size prompt payload for O(1) context scaling."""
    code_block = (
        f"```python\n{active_code}\n```"
        if active_code
        else "[No code written yet]"
    )

    return f"""Task Goal: {task}
Target Score: {TARGET_SCORE}
Runtime budget: {POOL_TIMEOUT_SECONDS:g} seconds for the entire validation dataset, not per image.

### FILE ACCEPTANCE CRITERIA
{acceptance_guidance(retained_metrics)}

### ITERATION SCOREBOARD
{scoreboard}

### PARETO ARCHIVE
{archive}

### ACTIVE SEARCH STRATEGY: {strategy_name}
{strategy_instruction}

### ARCHITECT GUIDANCE
{architect_guidance or "No architecture escalation has run yet."}

### CURRENT CANDIDATE CODE
{code_block}

### LATEST EVALUATION FEEDBACK
{diagnostics}

### REQUIRED NEXT MOVE
{next_move}

Instructions:
Review the scoreboard trend, current code, and diagnostic feedback.
Follow REQUIRED NEXT MOVE while preserving evidence-based rejection of non-railway images.
Implement the active search strategy materially; do not collapse it into another threshold-only variant of the current straight-line detector.
Increase Dataset Polygon IoU toward {TARGET_SCORE} subject to FILE ACCEPTANCE CRITERIA. Use positive IoU, coverage, and precision as diagnostic feedback; false-positive rate is also a hard acceptance constraint.
The attachments are separate native-resolution original frames and annotated copies, identified by the attachment index below. Ground truth uses red/blue rails with yellow fill; predictions use cyan/magenta rails with green fill. Inspect originals for rail evidence unobscured by overlays. Use coverage, precision, area ratio, and normalized centroid offset to correct geometry. Preserve full-resolution processing in the generated detector.
Do not inspect files or run tools. Respond immediately from the supplied code and feedback.
Return at most {MAX_CANDIDATE_LINES} lines; simplify the implementation when necessary.
Return ONLY full updated Python code inside ```python ``` fences."""


def parse_score(output: str) -> float | None:
    match = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", output)
    return float(match.group(1)) if match else None


def parse_metrics(output: str) -> CandidateMetrics | None:
    match = re.search(r"^METRICS:\s*(.+)$", output, re.MULTILINE)
    if not match:
        return None
    values = {
        key: float(value)
        for key, value in re.findall(
            r"([a-z_]+)=([0-9]+(?:\.[0-9]+)?)", match.group(1)
        )
    }
    required = {
        "dataset_iou",
        "mean_positive_iou",
        "positive_coverage",
        "polygon_precision",
        "false_positive_rate",
        "detected_positive_rate",
        "wall_seconds",
    }
    if not required.issubset(values):
        return None
    return CandidateMetrics(**{key: values[key] for key in required})


def print_evaluator_diagnostics(output: str) -> None:
    """Show the concise evaluator facts that explain a score."""
    for line in output.splitlines():
        if line.startswith(("SUMMARY:", "METRICS:", "DIAGNOSIS:")):
            print(line)


def save_candidate(code: str) -> None:
    """Replace candidate_script.py only after its source compiles."""
    compile(code, CANDIDATE_FILE, "exec")
    with open(CANDIDATE_FILE, "w") as candidate_file:
        candidate_file.write(code)


def evaluate_candidate_source(
    code: str,
    timeout: float | None = EVAL_TIMEOUT,
    diagnostic_montage_path: str | None = None,
    candidate_path: str | None = None,
) -> tuple[bool, str, str]:
    """Evaluate generated source without replacing the retained candidate."""
    compile(code, CANDIDATE_FILE, "exec")
    if candidate_path:
        write_artifact(candidate_path, code)
        return execute_evaluator(timeout, candidate_path, diagnostic_montage_path)
    with tempfile.TemporaryDirectory(prefix="railways-attempt-") as attempt_dir:
        attempt_path = os.path.join(attempt_dir, "candidate_script.py")
        with open(attempt_path, "w") as attempt_file:
            attempt_file.write(code)
        return execute_evaluator(
            timeout=timeout,
            candidate_path=attempt_path,
            diagnostic_montage_path=diagnostic_montage_path,
        )


def write_artifact(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as artifact:
        artifact.write(text)


def conservative_candidate_path() -> str:
    return os.path.join(os.path.dirname(CANDIDATE_FILE), "candidate_conservative.py")


def is_conservative_candidate(record: CandidateRecord) -> bool:
    return record.metrics.dataset_iou > 0 and record.metrics.false_positive_rate == 0


def load_conservative_candidate(
    baseline: CandidateRecord | None, search_dir: str,
) -> CandidateRecord | None:
    """Re-score the saved alternative on today's dataset, not stale run metrics."""
    path = conservative_candidate_path()
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as source:
        code = source.read()
    snapshot = os.path.join(search_dir, "conservative_baseline")
    write_artifact(snapshot + ".py", code)
    if baseline is not None and code == baseline.code:
        record = baseline
    else:
        print("\n--- Revalidating conservative alternative ---")
        success, stdout, stderr = execute_evaluator(candidate_path=snapshot + ".py")
        write_artifact(snapshot + ".stdout.txt", stdout)
        write_artifact(snapshot + ".stderr.txt", stderr)
        metrics = parse_metrics(stdout) if success else None
        if metrics is None:
            print("CONSERVATIVE UNVERIFIED: previous source preserved, but evaluation failed.")
            return None
        record = CandidateRecord(code, metrics, 0, "conservative_baseline", truncate_log(stdout), None)
    if not is_conservative_candidate(record):
        print("CONSERVATIVE NO LONGER QUALIFIES: previous source preserved; not zero-FPR with positive IoU on this dataset.")
        return None
    print(f"CONSERVATIVE VERIFIED: {record.metrics.summary()}")
    return record


def preserve_conservative_candidate(
    record: CandidateRecord,
    conservative: CandidateRecord | None,
    search_dir: str,
) -> CandidateRecord | None:
    """Keep the best nonempty zero-FPR detector independently of main retention."""
    if not is_conservative_candidate(record):
        return conservative
    if conservative is not None and record.metrics.dataset_iou <= conservative.metrics.dataset_iou:
        return conservative
    path = conservative_candidate_path()
    compile(record.code, path, "exec")
    # Versioned run artifacts survive later replacements of the stable filename.
    write_artifact(os.path.join(search_dir, "conservative_best.py"), record.code)
    write_artifact(os.path.join(search_dir, "conservative_best.metrics.json"), json.dumps({
        "metrics": asdict(record.metrics), "code_hash": record.code_hash,
        "iteration": record.iteration, "strategy": record.strategy,
    }, indent=2))
    write_artifact(path, record.code)
    print(f"CONSERVATIVE SAVED: {path}; {record.metrics.summary()}")
    return record


@dataclass
class AttemptEvaluation:
    code: str
    success: bool
    stdout: str
    stderr: str
    artifact_prefix: str


def evaluate_with_repairs(
    code: str,
    attempt_prefix: str,
    *,
    strategy_name: str,
    strategy_instruction: str,
    task: str,
    retained_metrics: CandidateMetrics | None,
    parent_montage: str | None,
) -> AttemptEvaluation:
    """Give a failed implementation bounded repairs within its iteration.

    Each version has separate artifacts. Successful evaluation exits immediately;
    score/FPR rejection is not a programming failure and never triggers a repair.
    Generation failures propagate to stop the search, not retry account errors.
    """
    version_prefix = attempt_prefix
    for repair_index in range(MAX_REPAIR_ATTEMPTS + 1):
        write_artifact(version_prefix + ".py", code)
        line_count = len(code.splitlines())
        if line_count > MAX_CANDIDATE_LINES:
            success, stdout, stderr = False, "", (
                f"LINE_LIMIT: generated {line_count} lines; configured limit is {MAX_CANDIDATE_LINES}."
            )
        else:
            try:
                success, stdout, stderr = evaluate_candidate_source(
                    code, candidate_path=version_prefix + ".py",
                    diagnostic_montage_path=version_prefix + ".json",
                )
            except SyntaxError as error:
                success, stdout, stderr = False, "", "".join(
                    traceback.format_exception_only(type(error), error)
                ).strip()
        write_artifact(version_prefix + ".stdout.txt", stdout)
        write_artifact(version_prefix + ".stderr.txt", stderr)
        result = AttemptEvaluation(code, success, stdout, stderr, version_prefix)
        if success or repair_index == MAX_REPAIR_ATTEMPTS:
            return result

        error_log = truncate_log(stderr or stdout)
        print(f"ATTEMPT FAILED; repairing {strategy_name}:\n{error_log}")
        next_prefix = f"{attempt_prefix}_repair_{repair_index + 1:02d}"
        repair_prompt = f"""Task: {task}
Architecture: {strategy_name}
Architecture instructions: {strategy_instruction}

Repair this failed implementation in place. Preserve its architecture and railway
discrimination logic; do not restart from the baseline or substitute another branch.
Correct the concrete failure below. For timeouts, reduce complexity without
downsampling. For excessive length, simplify without dropping detection stages.
Never hide errors with blanket exception handlers or unconditional empty output.
Evaluation will rerun on the entire original-resolution validation dataset.
Runtime budget: {POOL_TIMEOUT_SECONDS:g} seconds for the entire dataset.
Return no more than {MAX_CANDIDATE_LINES} lines, including comments and blank lines.
File acceptance criteria: {acceptance_guidance(retained_metrics)}

Failure from the current source:
{error_log}

Current failing source (repair THIS version):
```python
{code}
```
Return ONLY the complete repaired Python source inside ```python ``` fences.
"""
        write_artifact(next_prefix + ".prompt.txt", repair_prompt)
        print(f"REPAIR {repair_index + 1}/{MAX_REPAIR_ATTEMPTS}: {strategy_name} (same iteration)")
        try:
            response = call_codex_cli(
                repair_prompt, SYSTEM_PROMPT, model=CODEX_MODEL,
                reasoning_effort=CODEX_REASONING_EFFORT,
                diagnostic_montage=parent_montage,
                role=f"{strategy_name} repair {repair_index + 1}",
            )
        except (RuntimeError, subprocess.CalledProcessError) as error:
            write_artifact(next_prefix + ".generation_error.txt", str(getattr(error, "stderr", None) or error))
            raise
        write_artifact(next_prefix + ".response.txt", response)
        code = extract_code(response)
        version_prefix = next_prefix
    raise AssertionError("Unreachable repair loop exit")


def request_architect_guidance(
    best: CandidateRecord,
    archive: ParetoArchive,
    latest_feedback: str,
) -> str:
    branch_list = "\n".join(
        f"- {name}: {description}" for name, description in BRANCH_STRATEGIES
    )
    prompt = f"""The railway detector has plateaued.

Best metrics: {best.metrics.summary()}

File acceptance criteria: {acceptance_guidance(best.metrics)}

Pareto archive:
{archive.render()}

Current best implementation:
```python
{best.code}
```

Latest evaluator feedback:
{latest_feedback}

Required independent branches:
{branch_list}

For each branch, identify the representation change, core OpenCV operations, how to recover long curved/occluded rails, and how to control false positives and runtime. Process original-resolution pixels; never downsample railway evidence. Use the separate full-resolution original frames and prediction-vs-target overlays as evidence. Do not return code."""

    try:
        guidance = call_codex_cli(
            prompt,
            ARCHITECT_SYSTEM_PROMPT,
            timeout=ARCHITECT_TIMEOUT,
            model=ARCHITECT_MODEL,
            reasoning_effort=ARCHITECT_REASONING_EFFORT,
            diagnostic_montage=best.diagnostic_montage,
            role="plateau architect",
        )
        return guidance[:9000]
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ARCHITECT FALLBACK: {error}")
        return (
            "The architecture call was unavailable. Follow the active static "
            "branch specification exactly and replace the straight-pair "
            "representation with its requested alternative."
        )


def run_orchestrator(task_description: str):
    ctx = ContextManager()
    archive = ParetoArchive()

    best_code = ""
    if os.path.exists(CANDIDATE_FILE):
        with open(CANDIDATE_FILE, "r") as candidate_file:
            best_code = candidate_file.read()

    working_code = best_code
    working_feedback = "Initial iteration. Generate a baseline solution."
    working_montage = None
    best_record: CandidateRecord | None = None
    conservative_record: CandidateRecord | None = None
    stagnant_attempts = 0
    disruption_rounds = 0
    branch_queue: list[tuple[str, str]] = []
    architect_guidance = ""
    is_branch = False

    os.makedirs(RUNS_DIR, exist_ok=True)
    with nullcontext(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S-"), dir=RUNS_DIR)) as search_dir:
        print(f"RUN ARTIFACTS: {search_dir}")
        print(f"EVALUATION BUDGET: dataset={POOL_TIMEOUT_SECONDS:g}s; process={EVAL_TIMEOUT:g}s")
        print(f"CANDIDATE LIMITS: lines={MAX_CANDIDATE_LINES}; repair attempts={MAX_REPAIR_ATTEMPTS} per iteration")
        write_artifact(os.path.join(search_dir, "baseline.py"), best_code)
        if best_code:
            print("\n--- Baseline ---")
            baseline_montage = os.path.join(search_dir, "baseline.json")
            success, stdout, stderr = execute_evaluator(
                diagnostic_montage_path=baseline_montage
            )
            write_artifact(os.path.join(search_dir, "baseline.stdout.txt"), stdout)
            write_artifact(os.path.join(search_dir, "baseline.stderr.txt"), stderr)
            baseline_metrics = parse_metrics(stdout) if success else None
            if success and baseline_metrics is not None:
                baseline_feedback = truncate_log(stdout)
                best_record = CandidateRecord(
                    code=best_code,
                    metrics=baseline_metrics,
                    iteration=0,
                    strategy="baseline",
                    feedback=baseline_feedback,
                    diagnostic_montage=(
                        baseline_montage
                        if os.path.exists(baseline_montage)
                        else None
                    ),
                )
                archive.add(best_record)
                working_feedback = baseline_feedback
                working_montage = best_record.diagnostic_montage
                print(f"BASELINE: {baseline_metrics.summary()}")
                if baseline_metrics.false_positive_rate > MAX_FALSE_POSITIVE_RATE:
                    print(
                        "BASELINE FAILS FPR GATE: preserving it until a replacement "
                        f"passes FPR <= {MAX_FALSE_POSITIVE_RATE:.6f} with strictly higher IoU."
                    )
                ctx.record(
                    iteration=0,
                    score=baseline_metrics.dataset_iou,
                    status="BASELINE",
                    notes=baseline_metrics.summary(),
                    feedback=(
                        "Baseline evaluation succeeded.\n"
                        f"Evaluator Output:\n{baseline_feedback}"
                    ),
                )
            else:
                baseline_error = truncate_log(stderr or stdout)
                print(f"BASELINE FAILED:\n{baseline_error}")
                working_feedback = baseline_error
                ctx.record(
                    iteration=0,
                    score=0.0,
                    status="CRASH",
                    notes="Initial candidate failed evaluation",
                    feedback=(
                        "Baseline evaluation failed. Repair or replace the current "
                        f"candidate.\n{baseline_error}"
                    ),
                )

        conservative_record = load_conservative_candidate(best_record, search_dir)
        if best_record is not None:
            conservative_record = preserve_conservative_candidate(best_record, conservative_record, search_dir)

        for iteration in range(1, MAX_ITERATIONS + 1):
            # Resume after the final branch even if that branch timed out or
            # failed compilation (those paths continue before the loop's end).
            if is_branch and not branch_queue and archive.records:
                seed = archive.best_search_parent()
                working_code = seed.code
                working_feedback = seed.feedback
                working_montage = seed.diagnostic_montage
                stagnant_attempts = 0
                print(
                    "DISRUPTION ROUND COMPLETE: continuing from balanced search "
                    f"parent (iteration {seed.iteration}; {seed.metrics.summary()})."
                )
            if (
                not branch_queue
                and stagnant_attempts >= MAX_STAGNANT_ITERATIONS
            ):
                if disruption_rounds >= MAX_DISRUPTION_ROUNDS or best_record is None:
                    print(
                        f"STOPPED: no global improvement for "
                        f"{MAX_STAGNANT_ITERATIONS} consecutive local attempts "
                        "and the disruption budget is exhausted."
                    )
                    break

                disruption_rounds += 1
                print(
                    f"\n=== DISRUPTION ROUND {disruption_rounds}/"
                    f"{MAX_DISRUPTION_ROUNDS}: architecture reset ==="
                )
                architect_guidance = request_architect_guidance(
                    best_record, archive, working_feedback
                )
                branch_queue = list(BRANCH_STRATEGIES)
                stagnant_attempts = 0

            is_branch = bool(branch_queue)
            if is_branch:
                strategy_name, strategy_instruction = branch_queue.pop(0)
                if archive.records:
                    parent = archive.parent_for(strategy_name)
                    parent_code = parent.code
                    parent_feedback = parent.feedback
                    parent_montage = parent.diagnostic_montage
                    parent_label = (
                        "baseline"
                        if parent.iteration == 0
                        else f"iteration {parent.iteration}"
                    )
                else:
                    parent_code = working_code
                    parent_feedback = working_feedback
                    parent_montage = working_montage
                    parent_label = "working candidate"
                mode = f"DIVERGE:{strategy_name} from {parent_label}"
            else:
                strategy_name = "local_refinement"
                strategy_instruction = (
                    "Repair the current working implementation using the visual "
                    "and geometry diagnostics. Make a structural correction when "
                    "several images have zero overlap; do not only tune thresholds."
                )
                parent_code = working_code
                parent_feedback = working_feedback
                parent_montage = working_montage
                mode = "EXPLOIT:local_refinement"

            print(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} [{mode}] ---")
            prompt_payload = build_prompt_payload(
                scoreboard=ctx.render_scoreboard(),
                archive=archive.render(),
                active_code=parent_code,
                diagnostics=parent_feedback,
                next_move=ctx.next_move(),
                strategy_name=strategy_name,
                strategy_instruction=strategy_instruction,
                architect_guidance=architect_guidance,
                task=task_description,
                retained_metrics=best_record.metrics if best_record else None,
            )
            attempt_prefix = os.path.join(search_dir, f"iteration_{iteration:02d}_{strategy_name}")
            write_artifact(attempt_prefix + ".prompt.txt", prompt_payload)

            try:
                llm_output = call_codex_cli(
                    prompt_payload,
                    SYSTEM_PROMPT,
                    model=CODEX_MODEL,
                    reasoning_effort=CODEX_REASONING_EFFORT,
                    diagnostic_montage=parent_montage,
                    role=f"{strategy_name} implementer",
                )
            except subprocess.CalledProcessError as error:
                write_artifact(attempt_prefix + ".generation_error.txt", str(error.stderr))
                print(f"Codex CLI Execution Error:\n{error.stderr}")
                break
            except RuntimeError as error:
                write_artifact(attempt_prefix + ".generation_error.txt", str(error))
                print(f"STOPPED: {error}")
                print("The best successfully evaluated candidate was preserved.")
                break

            code = extract_code(llm_output)
            write_artifact(attempt_prefix + ".response.txt", llm_output)
            try:
                attempt = evaluate_with_repairs(
                    code, attempt_prefix,
                    strategy_name=strategy_name,
                    strategy_instruction=strategy_instruction,
                    task=task_description,
                    retained_metrics=best_record.metrics if best_record else None,
                    parent_montage=parent_montage,
                )
            except (RuntimeError, subprocess.CalledProcessError) as error:
                print(f"STOPPED: repair generation failed: {getattr(error, 'stderr', None) or error}")
                print("The retained candidate was preserved.")
                break

            code, success = attempt.code, attempt.success
            stdout, stderr = attempt.stdout, attempt.stderr
            attempt_prefix = attempt.artifact_prefix
            attempt_montage = attempt_prefix + ".json"
            working_code = code
            if not success:
                err_log = truncate_log(stderr)
                timed_out = "EVALUATION_TIMEOUT" in err_log
                print(f"FAILED: {'Evaluation timeout' if timed_out else 'Candidate evaluation error'}\n{err_log}")
                working_feedback = (
                    f"Execution Failed with Error:\n{err_log}\n"
                    + ("Reduce computational complexity at native resolution. Bound fragment counts "
                       "before pairwise matching and avoid full-image operations inside segment loops. "
                       "Do not downsample or replace detection with unconditional []."
                       if timed_out else "Fix the programming error before changing detection behavior.")
                )
                working_montage = None
                ctx.record(
                    iteration=iteration,
                    score=0.0,
                    status="CRASH",
                    notes="Syntax or Runtime Exception",
                    feedback=working_feedback,
                )
                stagnant_attempts += 1
                continue

            metrics = parse_metrics(stdout)
            if metrics is None:
                print("FAILED: evaluator.py did not emit a complete METRICS line.")
                working_feedback = (
                    "Evaluator output contract failure.\n"
                    f"Evaluator Output:\n{truncate_log(stdout)}"
                )
                ctx.record(
                    iteration=iteration,
                    score=0.0,
                    status="CRASH",
                    notes="Evaluator output contract failure",
                    feedback=working_feedback,
                )
                stagnant_attempts += 1
                continue

            eval_summary = truncate_log(stdout)
            working_feedback = eval_summary
            working_montage = (
                attempt_montage if os.path.exists(attempt_montage) else None
            )
            record = CandidateRecord(
                code=code,
                metrics=metrics,
                iteration=iteration,
                strategy=strategy_name,
                feedback=eval_summary,
                diagnostic_montage=working_montage,
            )
            archived = archive.add(record)
            conservative_record = preserve_conservative_candidate(record, conservative_record, search_dir)

            previous_best = (
                best_record.metrics.dataset_iou if best_record is not None else 0.0
            )
            accepted, decision_reason = retention_decision(
                metrics, best_record.metrics if best_record else None
            )
            write_artifact(attempt_prefix + ".decision.json", json.dumps({
                "metrics": asdict(metrics), "accepted": accepted,
                "archived": archived, "reason": decision_reason,
            }, indent=2))
            if accepted:
                status = "IMPROVED"
                best_record = record
                best_code = code
                save_candidate(best_code)
                retention_note = f"retained: {decision_reason}"
                stagnant_attempts = 0
            elif archived:
                status = "ARCHIVED"
                retention_note = (
                    f"kept as a Pareto alternate; retained file unchanged: {decision_reason}"
                )
                stagnant_attempts += 1
            elif metrics.dataset_iou == previous_best:
                status = "UNCHANGED"
                retention_note = f"discarded; retained file unchanged: {decision_reason}"
                stagnant_attempts += 1
            else:
                status = "REJECTED"
                retention_note = f"discarded; retained file unchanged: {decision_reason}"
                stagnant_attempts += 1

            if not accepted:
                rejection = "FPR_REJECTED" if decision_reason.startswith("FPR ") else "ACCEPTANCE_REJECTED"
                working_feedback = f"{rejection}: {decision_reason}\n{eval_summary}"
                record.feedback = working_feedback

            print(
                f"SUCCESS: {metrics.summary()} (Previous Best: "
                f"{previous_best:.6f}) — {retention_note}"
            )
            print_evaluator_diagnostics(stdout)
            ctx.record(
                iteration=iteration,
                score=metrics.dataset_iou,
                status=status,
                notes=f"{strategy_name}; {retention_note}",
                feedback=(
                    f"Execution Succeeded. {metrics.summary()}. The attempt was "
                    f"{retention_note}.\nEvaluator Output:\n{working_feedback}"
                ),
            )

            if accepted and metrics.dataset_iou >= TARGET_SCORE:
                print(
                    f"\nTarget score achieved ({metrics.dataset_iou:.4f} >= "
                    f"{TARGET_SCORE}) with FPR={metrics.false_positive_rate:.6f} "
                    f"<= {MAX_FALSE_POSITIVE_RATE:.6f}! Optimization complete."
                )
                break

        write_artifact(os.path.join(search_dir, "archive.json"), json.dumps([
            {"iteration": record.iteration, "strategy": record.strategy,
             "metrics": asdict(record.metrics), "code_hash": record.code_hash}
            for record in archive.records
        ], indent=2))
        if conservative_record is not None:
            print(f"\nCONSERVATIVE: {conservative_record.metrics.summary()}; saved separately at {conservative_candidate_path()}")
        if best_record is not None:
            source = (
                "baseline"
                if best_record.iteration == 0
                else f"iteration {best_record.iteration}"
            )
            print(
                f"\nBEST: {best_record.metrics.summary()} from {source}; "
                "candidate preserved."
            )
            if best_record.metrics.false_positive_rate > MAX_FALSE_POSITIVE_RATE:
                print("FPR GATE NOT MET: original baseline remains; no acceptable replacement was found.")
            print("\nFINAL PARETO ARCHIVE:")
            print(archive.render())


if __name__ == "__main__":
    run_orchestrator(TASK_DESCRIPTION)
