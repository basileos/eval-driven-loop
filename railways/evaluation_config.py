"""Shared evaluation deadline (seconds for the entire validation dataset)."""
import math
import os

POOL_TIMEOUT_SECONDS = float(os.environ.get("RAILWAYS_EVAL_TIMEOUT", "180"))
if not math.isfinite(POOL_TIMEOUT_SECONDS) or POOL_TIMEOUT_SECONDS <= 0:
    raise ValueError("RAILWAYS_EVAL_TIMEOUT must be a finite positive number.")

# Allow startup, worker shutdown, scoring and native-resolution diagnostic output.
PROCESS_TIMEOUT_SECONDS = POOL_TIMEOUT_SECONDS + 60
