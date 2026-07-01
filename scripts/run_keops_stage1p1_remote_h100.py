#!/usr/bin/env python3
"""Run the Stage 1-p1 KeOps cold/warm timing experiment on remote H100."""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

from keops_stage1_common import (
    EXPECTED_TIMING_SCOPE,
    PRECISIONS,
    SvmWorkload,
    correctness_stats,
    ground_truth_path,
    keops_command,
    matrix_shape,
    parse_metrics,
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
METHOD = "KeOps"
KEOPS_ROOT = Path("/home/lxheq/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/baselines/keops-stage1")
HELPER = KEOPS_ROOT / "scripts/keops_exact_kde.py"
PYTHON_BIN = sys.executable
CUDA_HOME = Path("/usr/local/cuda-12.4")

DATA_ROOT = Path("/home/lxheq/Documents/workspace/dataset/GPU-accelerated_Kernel_Density_Computation")
TMP_ROOT = Path("/home/lxheq/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/tmp-results/stage1-p1/keops")
GROUND_TRUTH_ROOT = Path("/home/lxheq/Documents/workspace/dataset/GPU-accelerated_Kernel_Density_Computation/exact/experiments/stage0/ground_truth")

TIMEOUT_SECONDS = 3600
TIMING_MODES = ("cold_start", "warm")

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


def write_one_query_file(source_query: Path, target_query: Path) -> None:
    with source_query.open("r", encoding="utf-8") as source:
        header = source.readline().split()
        if len(header) != 2:
            raise ValueError(f"Invalid query matrix header: {source_query}")
        _, dim = header
        first_row = source.readline()
        if not first_row:
            raise ValueError(f"Query matrix has no rows: {source_query}")

    target_query.parent.mkdir(parents=True, exist_ok=True)
    target_query.write_text(f"1 {dim}\n{first_row}", encoding="utf-8")


def summary_row(
    *,
    timestamp: str,
    git_branch: str,
    git_commit: str,
    workload: SvmWorkload,
    precision,
    timing_mode: str,
    batch_size: int,
    output_path: Path,
    log_path: Path,
    command: list[str],
    return_code: str,
    timeout_seconds: int,
    cli_wall_seconds: float,
    status: str,
    metrics: dict[str, str],
    correctness: dict[str, str],
) -> dict[str, str]:
    data_rows, dim = matrix_shape(workload.data_path)
    query_rows, _ = matrix_shape(workload.query_path)
    warmup_policy = metrics.get(
        "warmup_policy",
        "full_untimed_keops_call" if timing_mode == "warm" else "existing_cache_fresh_process",
    )
    return {
        "timestamp": timestamp,
        "machine": MACHINE,
        "method": METHOD,
        "workload": workload.workload,
        "dataset": workload.dataset,
        "precision": precision.precision,
        "dtype": precision.dtype,
        "scale": f"scott_diag_b={workload.scale_value}",
        "batch_policy": "single_batch",
        "timing_mode": timing_mode,
        "warmup_policy": warmup_policy,
        "timing_scope": metrics.get("timing_scope", ""),
        "runtime_s": metrics.get("execution_seconds", ""),
        "qps": metrics.get("qps", ""),
        "max_abs_err": correctness["max_abs_err"],
        "max_rel_err": correctness["max_rel_err"],
        "abs_p01": correctness["abs_p01"],
        "abs_p25": correctness["abs_p25"],
        "abs_p50": correctness["abs_p50"],
        "abs_p75": correctness["abs_p75"],
        "abs_p99": correctness["abs_p99"],
        "rel_p01": correctness["rel_p01"],
        "rel_p25": correctness["rel_p25"],
        "rel_p50": correctness["rel_p50"],
        "rel_p75": correctness["rel_p75"],
        "rel_p99": correctness["rel_p99"],
        "value_count": correctness["value_count"],
        "query_rows": str(query_rows),
        "data_rows": str(data_rows),
        "dim": str(dim),
        "batch_size": str(batch_size),
        "batch_count": metrics.get("batch_count", ""),
        "return_code": return_code,
        "timeout_seconds": str(timeout_seconds),
        "cli_wall_seconds": f"{cli_wall_seconds:.9f}",
        "reference_output": str(ground_truth_path(GROUND_TRUTH_ROOT, workload)),
        "output_path": str(output_path),
        "log_path": str(log_path),
        "command": shell_join(command),
        "status": status,
        "git_branch": git_branch,
        "git_commit": git_commit,
    }


def runner_status_row(
    *,
    timestamp: str,
    workload: SvmWorkload,
    precision,
    timing_mode: str,
    output_path: Path,
    log_path: Path,
    command: list[str],
    return_code: str,
    timeout_seconds: int,
    cli_wall_seconds: float,
    status: str,
) -> dict[str, str]:
    return {
        "timestamp": timestamp,
        "machine": MACHINE,
        "method": METHOD,
        "workload": workload.workload,
        "precision": precision.precision,
        "timing_mode": timing_mode,
        "return_code": return_code,
        "timeout_seconds": str(timeout_seconds),
        "cli_wall_seconds": f"{cli_wall_seconds:.9f}",
        "output_path": str(output_path),
        "log_path": str(log_path),
        "command": shell_join(command),
        "status": status,
    }


def write_timing_scope(path: Path) -> None:
    rows = [
        {
            "method": METHOD,
            "timing_mode": "cold_start",
            "timing_scope": EXPECTED_TIMING_SCOPE,
            "includes_host_to_device_transfer": "yes",
            "includes_device_allocation": "yes",
            "includes_jit_cache_load": "yes_if_triggered_by_measured_call",
            "includes_output_transfer": "yes",
            "includes_disk_write": "no",
            "notes": "Fresh Python process; run-local KeOps cache is prepared before accepted rows.",
        },
        {
            "method": METHOD,
            "timing_mode": "warm",
            "timing_scope": EXPECTED_TIMING_SCOPE,
            "includes_host_to_device_transfer": "yes",
            "includes_device_allocation": "yes",
            "includes_jit_cache_load": "no_unless_triggered_again",
            "includes_output_transfer": "yes",
            "includes_disk_write": "no",
            "notes": "One full untimed KeOps call is executed before the measured call in the same process.",
        },
    ]
    write_csv(
        path,
        rows,
        [
            "method",
            "timing_mode",
            "timing_scope",
            "includes_host_to_device_transfer",
            "includes_device_allocation",
            "includes_jit_cache_load",
            "includes_output_transfer",
            "includes_disk_write",
            "notes",
        ],
    )


def write_sha256sums(run_dir: Path) -> None:
    lines: list[str] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(run_dir)}")
    write_text(run_dir / "SHA256SUMS", "\n".join(lines) + "\n")


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = TMP_ROOT / f"{MACHINE}_{timestamp}"
    commands_file = run_dir / "commands.sh"
    inventory_file = run_dir / "machine_inventory.txt"
    timing_scope_file = run_dir / "timing_scope.csv"
    summary_file = run_dir / "summary.csv"
    runner_status_file = run_dir / "runner_status.csv"
    cache_root = run_dir / "keops-cache"
    cache_prepare_root = run_dir / "cache_prepare"

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
    with inventory_file.open("a", encoding="utf-8") as handle:
        handle.write(f"\nkeops_cache_folder: {cache_root}\n")
        handle.write("cache_policy: run-local cache prepared before accepted timing rows\n")

    write_timing_scope(timing_scope_file)

    summary_rows: list[dict[str, str]] = []
    status_rows: list[dict[str, str]] = []

    for workload in WORKLOADS:
        query_rows, _ = matrix_shape(workload.query_path)
        batch_size = query_rows
        reference_path = ground_truth_path(GROUND_TRUTH_ROOT, workload)

        for precision in PRECISIONS:
            prepare_query = cache_prepare_root / workload.workload / f"{workload.workload}_one_query.data"
            prepare_output = (
                cache_prepare_root
                / workload.workload
                / precision.precision.lower()
                / f"{workload.workload}_{precision.precision.lower()}_cache_prepare.out"
            )
            prepare_log = (
                cache_prepare_root
                / workload.workload
                / precision.precision.lower()
                / f"{workload.workload}_{precision.precision.lower()}_cache_prepare.log"
            )
            write_one_query_file(workload.query_path, prepare_query)
            prepare_workload = SvmWorkload(
                workload.workload,
                workload.dataset,
                workload.data_path,
                prepare_query,
                workload.scale_value,
            )
            prepare_command = keops_command(
                PYTHON_BIN,
                HELPER,
                KEOPS_ROOT,
                prepare_workload,
                precision,
                1,
                prepare_output,
                "cold_start",
            )
            print(f"[CACHE] {workload.workload} {precision.precision}", flush=True)
            run_command(
                prepare_command,
                prepare_log,
                commands_file,
                KEOPS_ROOT,
                env,
                TIMEOUT_SECONDS,
            )

            for timing_mode in TIMING_MODES:
                output_path = (
                    run_dir
                    / "outputs"
                    / workload.workload
                    / precision.precision.lower()
                    / timing_mode
                    / f"{workload.workload}_keops_{precision.precision.lower()}_{timing_mode}_single_batch.out"
                )
                log_path = (
                    run_dir
                    / "logs"
                    / workload.workload
                    / precision.precision.lower()
                    / timing_mode
                    / f"{workload.workload}_keops_{precision.precision.lower()}_{timing_mode}_single_batch.log"
                )
                command = keops_command(
                    PYTHON_BIN,
                    HELPER,
                    KEOPS_ROOT,
                    workload,
                    precision,
                    batch_size,
                    output_path,
                    timing_mode,
                )
                print(
                    f"[RUN] {workload.workload} {precision.precision} {timing_mode} "
                    f"batch_size={batch_size}",
                    flush=True,
                )
                return_code, wall_seconds, log_text, timed_out = run_command(
                    command,
                    log_path,
                    commands_file,
                    KEOPS_ROOT,
                    env,
                    TIMEOUT_SECONDS,
                )
                metrics = parse_metrics(log_text)
                status = "timeout" if timed_out else ("ok" if return_code == "0" else f"failed({return_code})")
                if status == "ok" and metrics.get("timing_scope") != EXPECTED_TIMING_SCOPE:
                    status = "missing_or_bad_timing_scope"
                if status == "ok" and metrics.get("timing_mode") != timing_mode:
                    status = "missing_or_bad_timing_mode"
                if status == "ok" and metrics.get("query_count") != str(query_rows):
                    status = "bad_query_count"
                if status == "ok" and metrics.get("batch_count") != "1":
                    status = "not_single_batch"

                correctness = correctness_stats(output_path, reference_path)
                if status != "ok":
                    correctness["status"] = status

                summary_rows.append(
                    summary_row(
                        timestamp=timestamp,
                        git_branch=git_branch,
                        git_commit=git_commit,
                        workload=workload,
                        precision=precision,
                        timing_mode=timing_mode,
                        batch_size=batch_size,
                        output_path=output_path,
                        log_path=log_path,
                        command=command,
                        return_code=return_code,
                        timeout_seconds=TIMEOUT_SECONDS,
                        cli_wall_seconds=wall_seconds,
                        status=status,
                        metrics=metrics,
                        correctness=correctness,
                    )
                )
                status_rows.append(
                    runner_status_row(
                        timestamp=timestamp,
                        workload=workload,
                        precision=precision,
                        timing_mode=timing_mode,
                        output_path=output_path,
                        log_path=log_path,
                        command=command,
                        return_code=return_code,
                        timeout_seconds=TIMEOUT_SECONDS,
                        cli_wall_seconds=wall_seconds,
                        status=status,
                    )
                )

    summary_fields = [
        "timestamp",
        "machine",
        "method",
        "workload",
        "dataset",
        "precision",
        "dtype",
        "scale",
        "batch_policy",
        "timing_mode",
        "warmup_policy",
        "timing_scope",
        "runtime_s",
        "qps",
        "max_abs_err",
        "max_rel_err",
        "abs_p01",
        "abs_p25",
        "abs_p50",
        "abs_p75",
        "abs_p99",
        "rel_p01",
        "rel_p25",
        "rel_p50",
        "rel_p75",
        "rel_p99",
        "value_count",
        "query_rows",
        "data_rows",
        "dim",
        "batch_size",
        "batch_count",
        "return_code",
        "timeout_seconds",
        "cli_wall_seconds",
        "reference_output",
        "output_path",
        "log_path",
        "command",
        "status",
        "git_branch",
        "git_commit",
    ]
    status_fields = [
        "timestamp",
        "machine",
        "method",
        "workload",
        "precision",
        "timing_mode",
        "return_code",
        "timeout_seconds",
        "cli_wall_seconds",
        "output_path",
        "log_path",
        "command",
        "status",
    ]

    write_csv(summary_file, summary_rows, summary_fields)
    write_csv(runner_status_file, status_rows, status_fields)

    if cache_root.exists():
        shutil.rmtree(cache_root)
    if cache_prepare_root.exists():
        shutil.rmtree(cache_prepare_root)

    write_sha256sums(run_dir)

    print(f"[DONE] run_dir={run_dir}")
    print(f"[DONE] summary={summary_file}")
    print(f"[DONE] runner_status={runner_status_file}")
    print(f"[DONE] timing_scope={timing_scope_file}")
    print("[DONE] Copy this run directory back to local persistent storage before shutting down remote.")

    failures = [row for row in summary_rows if row["status"] not in ("ok", "timeout")]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
