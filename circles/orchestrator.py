import os
import re
import subprocess
import sys
import itertools
import sys
import threading
import time

# Configuration
MAX_ITERATIONS = 10
TARGET_SCORE = 0.95
CANDIDATE_FILE = "candidate_script.py"
EVALUATOR_FILE = "evaluator.py"
# TIMEOUT_SECONDS = 15
# CODEX_TIMEOUT_SECONDS = 180
# Configuration
CODEX_TIMEOUT: int | None = (
    None  # Set to None for unlimited execution time, or e.g. 600 (10 mins)
)
EVAL_TIMEOUT: int | None = (
    60  # Set to None for large validation sets, or e.g. 120
)
MAX_LOG_BYTES = 1200  # Cap stderr/stdout size to protect token context

SYSTEM_PROMPT = """
You are an expert Computer Vision engineer.
Write Python code implementing strictly the function:
`process_image(img_path: str) -> list[list[float]]`

Rules:
1. Return ONLY runnable Python code inside standard ```python ... ``` code fences.
2. DO NOT include `if __name__ == '__main__':` or external CLI handlers.
3. Execution Speed: Must process a 600x600 image in under 500ms. Avoid dense parameter sweeps or brute-force nested loops.
4. Export `process_image` as the core entry point returning bounding boxes [[x1, y1, x2, y2], ...].
5. Return an empty list [] if no targets are found or image loading fails.
"""


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


def extract_code(llm_response: str) -> str:
    """Extracts raw code from Markdown python blocks."""
    match = re.search(r"```python\n(.*?)```", llm_response, re.DOTALL)
    return match.group(1).strip() if match else llm_response.strip()


def truncate_log(log_text: str, max_bytes: int = MAX_LOG_BYTES) -> str:
    """Prevents large stack traces or verbose outputs from blowing up prompt tokens."""
    if len(log_text) > max_bytes:
        return log_text[:max_bytes] + "\n... [Log Truncated]"
    return log_text if log_text else "No output details."


def call_codex_cli(prompt: str, system_prompt: str = "", timeout: int | None = CODEX_TIMEOUT,) -> str:
    """Executes Codex CLI non-interactively in ephemeral mode."""
    full_prompt = (
        f"System Instructions:\n{system_prompt}\n\nTask Payload:\n{prompt}"
    )

    cmd = [
        "codex",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "-",
    ]

    try:
        with Spinner("Generating code via Codex CLI"):
            result = subprocess.run(
                cmd,
                input=full_prompt,  # Pipes prompt directly into stdin
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout
            )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Codex CLI timed out after {timeout} seconds during execution."
        )


def execute_evaluator(timeout: int | None = EVAL_TIMEOUT) -> tuple[bool, str, str]:
    """Runs evaluator.py with visual progress spinner and optional timeout."""
    try:
        with Spinner("Running evaluation suite"):
            result = subprocess.run(
                [sys.executable, EVALUATOR_FILE],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        return (
            result.returncode == 0,
            result.stdout.strip(),
            result.stderr.strip(),
        )
    except subprocess.TimeoutExpired:
        return False, "", f"Execution timed out after {timeout} seconds."


def build_prompt_payload(
    scoreboard: str, active_code: str, diagnostics: str, task: str
) -> str:
    """Constructs a fixed-size prompt payload for O(1) context scaling."""
    code_block = (
        f"```python\n{active_code}\n```"
        if active_code
        else "[No code written yet]"
    )

    return f"""Task Goal: {task}
Target Score: {TARGET_SCORE}

### ITERATION SCOREBOARD
{scoreboard}

### CURRENT CANDIDATE CODE
{code_block}

### LATEST EVALUATION FEEDBACK
{diagnostics}

Instructions:
Review the scoreboard trend, current code, and diagnostic feedback.
Fix bugs, refine image processing parameters, or add edge-case guards to increase Mean IoU score toward {TARGET_SCORE}.
Return ONLY full updated Python code inside ```python ``` fences."""


def run_orchestrator(task_description: str):
    ctx = ContextManager()

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} ---")

        # 1. Load current active candidate code from disk (if available)
        current_code = ""
        if os.path.exists(CANDIDATE_FILE):
            with open(CANDIDATE_FILE, "r") as f:
                current_code = f.read()

        # 2. Build static-size prompt payload
        prompt_payload = build_prompt_payload(
            scoreboard=ctx.render_scoreboard(),
            active_code=current_code,
            diagnostics=ctx.get_latest_feedback(),
            task=task_description,
        )

        # 3. Call Codex CLI
        try:
            llm_output = call_codex_cli(prompt_payload, SYSTEM_PROMPT)
        except subprocess.CalledProcessError as e:
            print(f"Codex CLI Execution Error:\n{e.stderr}")
            break

        code = extract_code(llm_output)

        # 4. Save generated code turn
        with open(CANDIDATE_FILE, "w") as f:
            f.write(code)

        # 5. Run evaluation script
        success, stdout, stderr = execute_evaluator()

        # 6. Parse metrics & record context feedback
        if not success:
            err_log = truncate_log(stderr)
            print(f"FAILED: Runtime Error in evaluator.py")
            ctx.record(
                iteration=iteration,
                score=0.0,
                status="CRASH",
                notes="Syntax or Runtime Exception",
                feedback=f"Execution Failed with Error:\n{err_log}\nFix the bug.",
            )
        else:
            score_match = re.search(r"SCORE:\s*([0-9.]+)", stdout)
            score = float(score_match.group(1)) if score_match else 0.0
            prev_best = ctx.get_best_score()

            if score > prev_best:
                status = "IMPROVED"
            elif score < prev_best:
                status = "REGRESSED"
            else:
                status = "UNCHANGED"

            print(
                f"SUCCESS: Score = {score:.4f} (Previous Best: {prev_best:.4f})"
            )

            eval_summary = truncate_log(stdout)
            ctx.record(
                iteration=iteration,
                score=score,
                status=status,
                notes=f"Measured Mean IoU: {score:.4f}",
                feedback=f"Execution Succeeded.\nScore: {score:.4f} / Target: {TARGET_SCORE}\nEvaluator Output:\n{eval_summary}",
            )

            # Check exit condition
            if score >= TARGET_SCORE:
                print(
                    f"\nTarget score achieved ({score:.4f} >= {TARGET_SCORE})! Optimization complete."
                )
                break


if __name__ == "__main__":
    task = "Detect circles in images using OpenCV/NumPy heuristics and return bounding boxes [x1, y1, x2, y2]."
    run_orchestrator(task)