import os
import re
import subprocess
import sys
from openai import OpenAI

# Configuration
MAX_ITERATIONS = 5
TARGET_SCORE = 0.85
CANDIDATE_FILE = "candidate_script.py"
TIMEOUT_SECONDS = 10

client = OpenAI()  # Standard SDK, expects OPENAI_API_KEY environment variable

SYSTEM_PROMPT = """
You are an expert Computer Vision engineer.
Write Python code implementing `process_dataset(dataset_path: str) -> float`.
Return ONLY runnable code wrapped in ```python ... ``` code fences.
Do not introduce external system calls outside OpenCV, NumPy, or PIL.
"""


def extract_code(llm_response: str) -> str:
    """Extracts raw code from Markdown code fences."""
    match = re.search(r"```python\n(.*?)```", llm_response, re.DOTALL)
    return match.group(1).strip() if match else llm_response.strip()


def execute_candidate(timeout: int = TIMEOUT_SECONDS) -> tuple[bool, str, str]:
    """Runs generated script safely via subprocess."""
    try:
        result = subprocess.run(
            [sys.executable, CANDIDATE_FILE],
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
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Task: {task_description}\nTarget score: {TARGET_SCORE}",
        },
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} ---")

        # 1. Query LLM
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=0.2
        )
        llm_output = response.choices[0].message.content
        code = extract_code(llm_output)

        # 2. Write candidate to disk
        with open(CANDIDATE_FILE, "w") as f:
            f.write(code)

        # 3. Execute candidate code safely
        success, stdout, stderr = execute_candidate()

        # 4. Evaluate execution logs and compute metrics
        if not success:
            feedback = (
                f"Execution failed with error:\n{stderr}\nFix the bug."
            )
            score = 0.0
            print(f"FAILED: {stderr[:150]}...")
        else:
            # Parse score output from stdout (e.g. stdout formatted as "SCORE: 0.78")
            score_match = re.search(r"SCORE:\s*([0-9.]+)", stdout)
            score = float(score_match.group(1)) if score_match else 0.0
            feedback = f"Execution succeeded. Measured metric score: {score:.4f} / {TARGET_SCORE}."
            print(f"SUCCESS: Score = {score:.4f}")

        # 5. Check termination condition
        if score >= TARGET_SCORE:
            print(f"\nTarget metric achieved ({score:.4f})! Stopping loop.")
            break

        # 6. Append interaction context for the next iteration turn
        messages.append({"role": "assistant", "content": llm_output})
        messages.append(
            {
                "role": "user",
                "content": f"{feedback}\nRefine the code to improve performance.",
            }
        )


if __name__ == "__main__":
    task = "Detect circles in images using OpenCV HoughTransforms and return mean IoU score in stdout as 'SCORE: <value>'."
    run_orchestrator(task)

