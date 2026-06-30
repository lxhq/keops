#!/usr/bin/env python3
"""Run the remote H100 Stage 1 KeOps batch-size sweep.

This is a KeOps-specific helper, not the Stage 1 top-level runner. The
top-level remote runner lives in the Basic repo and invokes this helper for the
KeOps batch-size sweep.
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

from keops_stage1_common import (
    BATCH_SIZES,
    EXPECTED_TIMING_SCOPE,
    PRECISIONS,
    SvmWorkload,
    collect_generated_kernel_rows,
    correctness_stats,
    ground_truth_path,
    keops_command,
    matrix_shape,
    parse_metrics,
    query_batches,
    require_files,
    run_command,
    run_env,
    run_shell_capture,
    shell_join,
    write_csv,
    write_inventory,
    write_text,
)


MACHINE = "remote-h100"
KEOPS_ROOT = Path("/home/lxheq/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/baselines/keops-stage1")
HELPER = KEOPS_ROOT / "scripts/keops_exact_kde.py"
PYTHON_BIN = sys.executable
CUDA_HOME = Path("/usr/local/cuda-12.4")

DATA_ROOT = Path("/home/lxheq/Documents/workspace/dataset/GPU-accelerated_Kernel_Density_Computation")
TMP_ROOT = Path("/home/lxheq/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/tmp-results/stage1/remote-h100/keops")
GROUND_TRUTH_ROOT = Path("/home/lxheq/Documents/workspace/dataset/GPU-accelerated_Kernel_Density_Computation/exact/experiments/stage0/ground_truth")

TIMEOUT_SECONDS = 3600

WORKLOADS = (
    SvmWorkload(
        "svm_susy",
        "SUSY",
        DATA_ROOT / "susy/SUSY_X.data",
        DATA_ROOT / "susy/SUSY_qSet.data",
    ),
    SvmWorkload(
        "svm_home",
        "Home",
        DATA_ROOT / "home/HT_Sensor_dataset_X.data",
        DATA_ROOT / "home/HT_Sensor_dataset_qSet.data",
    ),
    SvmWorkload(
        "svm_miniboone",
        "MiniBooNE",
        DATA_ROOT / "miniboone/MiniBooNE_X.data",
        DATA_ROOT / "miniboone/MiniBooNE_qSet.data",
    ),
)


def runtime_row(
    *,
    timestamp: str,
    git_branch: str,
    git_commit: str,
    workload: SvmWorkload,
    precision,
    batch_size: int,
    output_path: Path,
    log_path: Path,
    command: list[str],
    return_code: str,
    timeout_seconds: int,
    cli_wall_seconds: float,
    status: str,
    metrics: dict[str, str],
) -> dict[str, str]:
    data_rows, dim = matrix_shape(workload.data_path)
    query_rows, _ = matrix_shape(workload.query_path)
    return {
        "timestamp": timestamp,
        "machine": MACHINE,
        "workload": workload.workload,
        "dataset": workload.dataset,
        "precision": precision.precision,
        "dtype": precision.dtype,
        "query_rows": str(query_rows),
        "data_rows": str(data_rows),
        "dim": str(dim),
        "batch_size": str(batch_size),
        "query_batches": str(query_batches(query_rows, batch_size)),
        "runtime_s": metrics.get("execution_seconds", ""),
        "qps": metrics.get("qps", ""),
        "timing_scope": metrics.get("timing_scope", ""),
        "return_code": return_code,
        "timeout_seconds": str(timeout_seconds),
        "cli_wall_seconds": f"{cli_wall_seconds:.9f}",
        "output_path": str(output_path),
        "log_path": str(log_path),
        "command": shell_join(command),
        "status": status,
        "git_branch": git_branch,
        "git_commit": git_commit,
    }


def correctness_row(
    *,
    timestamp: str,
    git_branch: str,
    git_commit: str,
    workload: SvmWorkload,
    precision,
    batch_size: int,
    output_path: Path,
    reference_path: Path,
    run_status: str,
) -> dict[str, str]:
    stats = correctness_stats(output_path, reference_path)
    status = stats["status"] if run_status == "ok" else run_status
    return {
        "timestamp": timestamp,
        "machine": MACHINE,
        "workload": workload.workload,
        "precision": precision.precision,
        "dtype": precision.dtype,
        "batch_size": str(batch_size),
        "abs_p01": stats["abs_p01"],
        "abs_p25": stats["abs_p25"],
        "abs_p50": stats["abs_p50"],
        "abs_p75": stats["abs_p75"],
        "abs_p99": stats["abs_p99"],
        "max_abs_err": stats["max_abs_err"],
        "rel_p01": stats["rel_p01"],
        "rel_p25": stats["rel_p25"],
        "rel_p50": stats["rel_p50"],
        "rel_p75": stats["rel_p75"],
        "rel_p99": stats["rel_p99"],
        "max_rel_err": stats["max_rel_err"],
        "value_count": stats["value_count"],
        "reference_output": str(reference_path),
        "output_path": str(output_path),
        "status": status,
        "git_branch": git_branch,
        "git_commit": git_commit,
    }


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = TMP_ROOT / f"{MACHINE}_{timestamp}"
    commands_file = run_dir / "commands.sh"
    inventory_file = run_dir / "machine_inventory.txt"
    runtime_file = run_dir / "keops_batch_sweep.csv"
    correctness_file = run_dir / "keops_correctness.csv"
    generated_file = run_dir / "keops_generated_kernels.csv"
    cache_root = run_dir / "keops-cache"

    run_dir.mkdir(parents=True, exist_ok=True)
    write_text(
        commands_file,
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        f"# Generated by {Path(__file__).name} at {timestamp}.\n\n",
    )
    commands_file.chmod(0o755)

    require_files(
        [
            HELPER,
            *[w.data_path for w in WORKLOADS],
            *[w.query_path for w in WORKLOADS],
            *[ground_truth_path(GROUND_TRUTH_ROOT, w) for w in WORKLOADS],
        ]
    )

    env = run_env(KEOPS_ROOT, CUDA_HOME, cache_root)
    git_branch = run_shell_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"], KEOPS_ROOT, env)
    git_commit = run_shell_capture(["git", "rev-parse", "HEAD"], KEOPS_ROOT, env)
    write_inventory(
        inventory_file,
        machine=MACHINE,
        timestamp=timestamp,
        keops_root=KEOPS_ROOT,
        helper=HELPER,
        data_root=DATA_ROOT,
        tmp_root=TMP_ROOT,
        ground_truth_root=GROUND_TRUTH_ROOT,
        cuda_home=CUDA_HOME,
        python_bin=PYTHON_BIN,
        git_branch=git_branch,
        git_commit=git_commit,
        env=env,
    )

    runtime_rows: list[dict[str, str]] = []
    correctness_rows: list[dict[str, str]] = []
    generated_done: set[tuple[str, str]] = set()
    generated_rows: list[dict[str, str]] = []

    for workload in WORKLOADS:
        reference_path = ground_truth_path(GROUND_TRUTH_ROOT, workload)
        for precision in PRECISIONS:
            precision_cache = cache_root / workload.workload / precision.precision.lower()
            precision_env = run_env(KEOPS_ROOT, CUDA_HOME, precision_cache)
            for batch_size in BATCH_SIZES:
                output_path = (
                    run_dir
                    / "outputs"
                    / workload.workload
                    / precision.precision.lower()
                    / f"{workload.workload}_{precision.precision.lower()}_batch{batch_size}.out"
                )
                log_path = (
                    run_dir
                    / "logs"
                    / workload.workload
                    / precision.precision.lower()
                    / f"{workload.workload}_{precision.precision.lower()}_batch{batch_size}.log"
                )
                command = keops_command(
                    PYTHON_BIN,
                    HELPER,
                    KEOPS_ROOT,
                    workload,
                    precision,
                    batch_size,
                    output_path,
                )
                print(
                    f"[RUN] {workload.workload} {precision.precision} batch={batch_size}",
                    flush=True,
                )
                return_code, wall_seconds, log_text, timed_out = run_command(
                    command,
                    log_path,
                    commands_file,
                    KEOPS_ROOT,
                    precision_env,
                    TIMEOUT_SECONDS,
                )
                metrics = parse_metrics(log_text)
                status = "timeout" if timed_out else ("ok" if return_code == "0" else f"failed({return_code})")
                query_rows, _ = matrix_shape(workload.query_path)
                if status == "ok" and metrics.get("timing_scope") != EXPECTED_TIMING_SCOPE:
                    status = "missing_or_bad_timing"
                if status == "ok" and metrics.get("query_count") != str(query_rows):
                    status = "bad_query_count"

                runtime_rows.append(
                    runtime_row(
                        timestamp=timestamp,
                        git_branch=git_branch,
                        git_commit=git_commit,
                        workload=workload,
                        precision=precision,
                        batch_size=batch_size,
                        output_path=output_path,
                        log_path=log_path,
                        command=command,
                        return_code=return_code,
                        timeout_seconds=TIMEOUT_SECONDS,
                        cli_wall_seconds=wall_seconds,
                        status=status,
                        metrics=metrics,
                    )
                )
                correctness_rows.append(
                    correctness_row(
                        timestamp=timestamp,
                        git_branch=git_branch,
                        git_commit=git_commit,
                        workload=workload,
                        precision=precision,
                        batch_size=batch_size,
                        output_path=output_path,
                        reference_path=reference_path,
                        run_status=status,
                    )
                )

                key = (workload.workload, precision.precision)
                if status == "ok" and key not in generated_done:
                    generated_rows.extend(
                        collect_generated_kernel_rows(
                            MACHINE,
                            workload,
                            precision,
                            precision_cache,
                            run_dir,
                        )
                    )
                    generated_done.add(key)

    runtime_fields = [
        "timestamp",
        "machine",
        "workload",
        "dataset",
        "precision",
        "dtype",
        "query_rows",
        "data_rows",
        "dim",
        "batch_size",
        "query_batches",
        "runtime_s",
        "qps",
        "timing_scope",
        "return_code",
        "timeout_seconds",
        "cli_wall_seconds",
        "output_path",
        "log_path",
        "command",
        "status",
        "git_branch",
        "git_commit",
    ]
    correctness_fields = [
        "timestamp",
        "machine",
        "workload",
        "precision",
        "dtype",
        "batch_size",
        "abs_p01",
        "abs_p25",
        "abs_p50",
        "abs_p75",
        "abs_p99",
        "max_abs_err",
        "rel_p01",
        "rel_p25",
        "rel_p50",
        "rel_p75",
        "rel_p99",
        "max_rel_err",
        "value_count",
        "reference_output",
        "output_path",
        "status",
        "git_branch",
        "git_commit",
    ]
    generated_fields = [
        "machine",
        "workload",
        "precision",
        "dtype",
        "dim",
        "cache_root",
        "source_path",
        "nfo_path",
        "copied_source_path",
        "copied_nfo_path",
        "source_sha256",
        "nfo_sha256",
        "status",
    ]

    write_csv(runtime_file, runtime_rows, runtime_fields)
    write_csv(correctness_file, correctness_rows, correctness_fields)
    write_csv(generated_file, generated_rows, generated_fields)

    if cache_root.exists():
        shutil.rmtree(cache_root)

    print(f"[DONE] run_dir={run_dir}")
    print(f"[DONE] runtime={runtime_file}")
    print(f"[DONE] correctness={correctness_file}")
    print(f"[DONE] generated={generated_file}")
    print("[DONE] Copy this run directory back to local persistent storage before shutting down remote.")

    failures = [row for row in runtime_rows if row["status"] not in ("ok", "timeout")]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
