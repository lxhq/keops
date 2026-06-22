#!/usr/bin/env python3
"""Stage 0 remote KeOps runner placeholder.

This script is intentionally blocked until the remote machine paths and CUDA
environment are known. Once those values are fixed, it should mirror
run_keops_local_3080ti.py and keep the same output schema:

- summary.csv
- correctness.csv
- machine_inventory.txt
- commands.sh
- logs/
- outputs/
"""

from __future__ import annotations

from pathlib import Path


REMOTE_CONFIG_READY = False

KEOPS_ROOT = Path("TODO_REMOTE_KEOPS_ROOT")
HELPER = KEOPS_ROOT / "scripts/keops_exact_kde.py"
PYTHON_BIN = "python3"
CUDA_HOME = Path("TODO_REMOTE_CUDA_HOME")

DATA_ROOT = Path("TODO_REMOTE_DATASET_ROOT")
TMP_RESULT_ROOT = Path("TODO_REMOTE_TMP_RESULT_ROOT") / "stage0/keops"
PERSISTENT_RESULT_ROOT = Path("TODO_REMOTE_PERSISTENT_RESULT_ROOT") / "stage0"
BASIC_GROUND_TRUTH_ROOT = PERSISTENT_RESULT_ROOT / "TODO_REMOTE_BASIC_GROUND_TRUTH_RELATIVE_PATH"

MACHINE = "remote-tbd"
METHOD = "KeOps"
ENGINE = "keops"
BACKEND = "GPU"
PRECISION = "FP64"
DTYPE = "float64"
EXPECTED_TIMING_SCOPE = "in_memory_query_pipeline"
SVM_SCOTT_B = "0.1"
KDV_SCOTT_B = "1"
SVM_BATCH_SIZE = 4096
KDV_BATCH_SIZE = 8192
SMOKE_TIMEOUT_SECONDS = 600
SVM_TIMEOUT_SECONDS = 3600
KDV_TIMEOUT_SECONDS = 21600


def main() -> int:
    if not REMOTE_CONFIG_READY:
        print(
            "[ERROR] Remote KeOps Stage 0 runner is not configured yet.\n\n"
            "Fill these values first:\n"
            "  REMOTE_CONFIG_READY = True\n"
            "  KEOPS_ROOT\n"
            "  CUDA_HOME\n"
            "  DATA_ROOT\n"
            "  TMP_RESULT_ROOT\n"
            "  PERSISTENT_RESULT_ROOT\n"
            "  BASIC_GROUND_TRUTH_ROOT\n"
            "  MACHINE\n\n"
            "The finalized remote runner must collect the same fields as "
            "run_keops_local_3080ti.py, including runtime_s/qps from the "
            "helper timing output, machine inventory, commands.sh, summary.csv, "
            "correctness.csv, timeout_seconds, logs, and temporary outputs.",
            flush=True,
        )
        return 1

    print("[ERROR] Remote runner body is pending until remote details are fixed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
