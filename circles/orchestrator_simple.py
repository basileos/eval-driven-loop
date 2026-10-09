import os
import re
import subprocess
import sys

# Configuration
MAX_ITERATIONS = 5
TARGET_SCORE = 0.99
CANDIDATE_FILE = "candidate_script.py"
TIMEOUT_SECONDS = 10

# SYSTEM_PROMPT = """
# You are an expert Computer Vision engineer.
# Write Python code implementing `process_dataset(dataset_path: str) -> float`.
# Return ONLY runnable code wrapped in ```python ... ``` code fences.
# Do not introduce external system calls outside OpenCV, NumPy, or PIL.
# """

SYSTEM_PROMPT = """
You are an expert Computer Vision engineer.
Write Python code implementing ONLY the function:
`process_image(img_path: str) -> list[list[float]]`

Rules:
1. Return ONLY runnable Python code inside standard ```python ``` fences.
2. Do not introduce external system calls outside OpenCV, NumPy, or PIL.
3. DO NOT include a `if __name__ == '__main__':` block.
4. DO NOT include extra dataset loader or benchmarking functions like `process_dataset`.
5. Export `process_image` as the sole entry point.
"""



def extract_code(llm_response: str) -> str:
    """Extracts raw code from Markdown code fences."""
    match = re.search(r"```python\n(.*?)```", llm_response, re.DOTALL)
    return match.group(1).strip() if match else llm_response.strip()


def call_claude_cli(prompt: str, system_prompt: str = "") -> str:
    """Calls Claude CLI headlessly using print mode (-p)."""
    # -p: Runs in non-interactive mode and outputs result to stdout
    # --bare: Disables workspace hooks/skills for lightweight CI/script execution
    # --append-system-prompt: Injects task constraints on top of base model instructions
    cmd = ["claude", "-p", prompt, "--bare"]
    if system_prompt:
        cmd.extend(["--append-system-prompt", system_prompt])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def call_custom_codex_cli(prompt: str, system_prompt: str = "") -> str:
    """Calls codex CLI in non-interactive/ephemeral mode."""
    # Prepend system instructions directly into the prompt payload
    if system_prompt:
        full_prompt = f"System Instructions:\n{system_prompt}\n\nTask:\n{prompt}"
    else:
        full_prompt = prompt

    # cmd = ["codex", "exec", "--ephemeral", "--skip-git-repo-check", full_prompt]
    cmd = ["codex", "exec", "--skip-git-repo-check", full_prompt]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def execute_candidate(timeout: int = TIMEOUT_SECONDS) -> tuple[bool, str, str]:
    """Runs generated script safely via subprocess."""
    try:
        # result = subprocess.run(
        #     [sys.executable, CANDIDATE_FILE],
        #     capture_output=True,
        #     text=True,
        #     timeout=timeout,
        # )
        result = subprocess.run(
            [sys.executable, "evaluator.py"],  # Executes evaluator.py in a subprocess
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


def run_orchestrator(task_description: str):
    history = [f"Task: {task_description}\nTarget score: {TARGET_SCORE}"]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} ---")

        # 1. Compile current interaction thread into prompt context
        full_prompt = "\n\n".join(history)

        # 2. Query Claude via CLI
        try:
            llm_output = call_custom_codex_cli(full_prompt, system_prompt=SYSTEM_PROMPT)
        except subprocess.CalledProcessError as e:
            print(f"Codex CLI Error: {e.stderr}")
            break

        code = extract_code(llm_output)

        # 3. Write candidate code to disk
        with open(CANDIDATE_FILE, "w") as f:
            f.write(code)

        # 4. Execute candidate script
        success, stdout, stderr = execute_candidate()

        # 5. Evaluate execution results
        if not success:
            feedback = f"Execution failed with error:\n{stderr}\nFix the bug."
            score = 0.0
            print(f"FAILED: {stderr[:150]}...")
        else:
            score_match = re.search(r"SCORE:\s*([0-9.]+)", stdout)
            score = float(score_match.group(1)) if score_match else 0.0
            feedback = f"Execution succeeded. Score: {score:.4f} / {TARGET_SCORE}."
            print(f"SUCCESS: Score = {score:.4f}")

        # 6. Check exit condition
        if score >= TARGET_SCORE:
            print(f"\nTarget score achieved ({score:.4f})! Loop complete.")
            break

        # 7. Append turn context for the next CLI invocation
        history.append(f"Assistant Output:\n{llm_output}")
        history.append(f"Execution Feedback:\n{feedback}\nRefine the code.")


if __name__ == "__main__":
    task = "Detect circles in images using OpenCV HoughTransforms and output 'SCORE: <value>'."
    run_orchestrator(task)
