#!/usr/bin/env python3
"""Shared utilities for Stage 1 KeOps helper scripts."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import re
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PrecisionConfig:
    precision: str
    dtype: str


@dataclass(frozen=True)
class SvmWorkload:
    workload: str
    dataset: str
    data_path: Path
    query_path: Path
    scale_value: str = "1"


@dataclass
class RunRecord:
    workload: SvmWorkload
    precision: PrecisionConfig
    batch_size: int
    output_path: Path
    log_path: Path
    command: list[str]
    return_code: str
    timeout_seconds: int
    cli_wall_seconds: float
    timing_scope: str
    execution_seconds: str
    query_count_from_binary: str
    qps: str
    status: str


PRECISIONS = (
    PrecisionConfig("FP64", "float64"),
    PrecisionConfig("FP32", "float32"),
)

BATCH_SIZES = (10000, 8192, 4096, 2048, 1024, 512)
EXPECTED_TIMING_SCOPE = "in_memory_query_pipeline"
BACKEND = "GPU"
ENGINE = "keops"
METHOD = "KeOps"
METHOD_TOKEN = "keops"


def run_env(keops_root: Path, cuda_home: Path, cache_root: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    py_paths = [str(keops_root / "pykeops"), str(keops_root / "keopscore")]
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)
    env["CUDA_HOME"] = str(cuda_home)
    env["PATH"] = f"{cuda_home / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{cuda_home / 'lib64'}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
        env["KEOPS_CACHE_FOLDER"] = str(cache_root)
    return env


def shell_join(command: Iterable[str]) -> str:
    return shlex.join([str(part) for part in command])


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_command(commands_file: Path, keops_root: Path, command: list[str], timeout_seconds: int) -> None:
    with commands_file.open("a", encoding="utf-8") as handle:
        handle.write(f"cd {shlex.quote(str(keops_root))}\n")
        handle.write(f"# timeout_seconds: {timeout_seconds}\n")
        handle.write(shell_join(command))
        handle.write("\n\n")


def timeout_output_to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_command(
    command: list[str],
    log_path: Path,
    commands_file: Path,
    keops_root: Path,
    env: dict[str, str],
    timeout_seconds: int,
) -> tuple[str, float, str, bool]:
    append_command(commands_file, keops_root, command, timeout_seconds)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=keops_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        wall_seconds = time.perf_counter() - start
        partial_output = timeout_output_to_text(exc.stdout or exc.output)
        log_text = (
            partial_output
            + f"\n[TIMEOUT] Command exceeded timeout_seconds={timeout_seconds}.\n"
            + f"[TIMEOUT] cli_wall_seconds={wall_seconds:.9f}\n"
        )
        write_text(log_path, log_text)
        return "timeout", wall_seconds, log_text, True

    wall_seconds = time.perf_counter() - start
    write_text(log_path, completed.stdout)
    return str(completed.returncode), wall_seconds, completed.stdout, False


def run_shell_capture(command: list[str], keops_root: Path, env: dict[str, str]) -> str:
    completed = subprocess.run(
        command,
        cwd=keops_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout.rstrip()


def parse_metrics(log_text: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for key in ("timing_scope", "execution_seconds", "query_count", "qps"):
        match = re.search(rf"^{key}:\s*(.+?)\s*$", log_text, re.MULTILINE)
        if match:
            metrics[key] = match.group(1)
    return metrics


def matrix_shape(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline().split()
    if len(first) < 2:
        raise ValueError(f"Invalid matrix header: {path}")
    return int(first[0]), int(first[1])


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def iter_floats(path: Path) -> Iterable[float]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            for token in line.split():
                yield float(token)


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def correctness_stats(output_path: Path, reference_path: Path) -> dict[str, str]:
    if not output_path.is_file() or not reference_path.is_file():
        return {
            "value_count": "",
            "abs_p01": "",
            "abs_p25": "",
            "abs_p50": "",
            "abs_p75": "",
            "abs_p99": "",
            "max_abs_err": "",
            "rel_p01": "",
            "rel_p25": "",
            "rel_p50": "",
            "rel_p75": "",
            "rel_p99": "",
            "max_rel_err": "",
            "status": "missing_output",
        }

    abs_errors: list[float] = []
    rel_errors: list[float] = []
    output_iter = iter_floats(output_path)
    reference_iter = iter_floats(reference_path)

    while True:
        sentinel = object()
        output_value = next(output_iter, sentinel)
        reference_value = next(reference_iter, sentinel)
        if output_value is sentinel and reference_value is sentinel:
            break
        if output_value is sentinel or reference_value is sentinel:
            return {
                "value_count": str(len(abs_errors)),
                "abs_p01": "",
                "abs_p25": "",
                "abs_p50": "",
                "abs_p75": "",
                "abs_p99": "",
                "max_abs_err": "",
                "rel_p01": "",
                "rel_p25": "",
                "rel_p50": "",
                "rel_p75": "",
                "rel_p99": "",
                "max_rel_err": "",
                "status": "count_mismatch",
            }

        diff = abs(float(output_value) - float(reference_value))
        abs_errors.append(diff)
        if reference_value != 0.0:
            rel_errors.append(diff / abs(float(reference_value)))
        else:
            rel_errors.append(0.0 if output_value == 0.0 else float("inf"))

    abs_sorted = sorted(abs_errors)
    rel_sorted = sorted(rel_errors)
    return {
        "value_count": str(len(abs_errors)),
        "abs_p01": f"{percentile(abs_sorted, 1):.12g}",
        "abs_p25": f"{percentile(abs_sorted, 25):.12g}",
        "abs_p50": f"{percentile(abs_sorted, 50):.12g}",
        "abs_p75": f"{percentile(abs_sorted, 75):.12g}",
        "abs_p99": f"{percentile(abs_sorted, 99):.12g}",
        "max_abs_err": f"{max(abs_sorted) if abs_sorted else float('nan'):.12g}",
        "rel_p01": f"{percentile(rel_sorted, 1):.12g}",
        "rel_p25": f"{percentile(rel_sorted, 25):.12g}",
        "rel_p50": f"{percentile(rel_sorted, 50):.12g}",
        "rel_p75": f"{percentile(rel_sorted, 75):.12g}",
        "rel_p99": f"{percentile(rel_sorted, 99):.12g}",
        "max_rel_err": f"{max(rel_sorted) if rel_sorted else float('nan'):.12g}",
        "status": "ok",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_generated_kernel_rows(
    machine: str,
    workload: SvmWorkload,
    precision: PrecisionConfig,
    cache_root: Path,
    output_root: Path,
) -> list[dict[str, str]]:
    copied_root = output_root / "generated-kernels" / workload.workload / precision.precision.lower()
    copied_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    cu_files = sorted(cache_root.rglob("*.cu"))
    if not cu_files:
        rows.append({
            "machine": machine,
            "workload": workload.workload,
            "precision": precision.precision,
            "dtype": precision.dtype,
            "dim": str(matrix_shape(workload.data_path)[1]),
            "cache_root": str(cache_root),
            "source_path": "",
            "nfo_path": "",
            "copied_source_path": "",
            "copied_nfo_path": "",
            "source_sha256": "",
            "nfo_sha256": "",
            "status": "missing_generated_source",
        })
        return rows

    for source_path in cu_files:
        nfo_path = source_path.with_suffix(".nfo")
        copied_source = copied_root / source_path.name
        copied_source.write_bytes(source_path.read_bytes())
        copied_nfo = copied_root / nfo_path.name if nfo_path.is_file() else None
        if copied_nfo is not None:
            copied_nfo.write_bytes(nfo_path.read_bytes())

        rows.append({
            "machine": machine,
            "workload": workload.workload,
            "precision": precision.precision,
            "dtype": precision.dtype,
            "dim": str(matrix_shape(workload.data_path)[1]),
            "cache_root": str(cache_root),
            "source_path": str(source_path),
            "nfo_path": str(nfo_path) if nfo_path.is_file() else "",
            "copied_source_path": str(copied_source),
            "copied_nfo_path": str(copied_nfo) if copied_nfo is not None else "",
            "source_sha256": sha256_file(copied_source),
            "nfo_sha256": sha256_file(copied_nfo) if copied_nfo is not None else "",
            "status": "ok" if copied_nfo is not None else "missing_nfo",
        })
    return rows


def query_batches(query_rows: int, batch_size: int) -> int:
    return (query_rows + batch_size - 1) // batch_size


def keops_command(
    python_bin: str,
    helper: Path,
    keops_root: Path,
    workload: SvmWorkload,
    precision: PrecisionConfig,
    batch_size: int,
    output_path: Path,
) -> list[str]:
    return [
        python_bin,
        str(helper),
        "svm",
        str(workload.query_path),
        str(workload.data_path),
        str(output_path),
        "--scott-diag",
        workload.scale_value,
        "--batch-size",
        str(batch_size),
        "--engine",
        ENGINE,
        "--backend",
        BACKEND,
        "--dtype",
        precision.dtype,
        "--local-keops-root",
        str(keops_root),
    ]


def output_name(workload: SvmWorkload, precision: PrecisionConfig, batch_size: int) -> str:
    safe_b = workload.scale_value.replace(".", "p")
    return (
        f"{workload.workload}_keops_{precision.precision.lower()}"
        f"_batch{batch_size}_scott_diag_b{safe_b}.out"
    )


def ground_truth_path(ground_truth_root: Path, workload: SvmWorkload) -> Path:
    safe_b = workload.scale_value.replace(".", "p")
    return ground_truth_root / f"{workload.workload}_basic-scan_fp64_scott_diag_b{safe_b}.out"


def write_inventory(
    path: Path,
    *,
    machine: str,
    timestamp: str,
    keops_root: Path,
    helper: Path,
    data_root: Path,
    tmp_root: Path,
    ground_truth_root: Path,
    cuda_home: Path,
    python_bin: str,
    git_branch: str,
    git_commit: str,
    env: dict[str, str],
) -> None:
    def capture(label: str, command: list[str]) -> str:
        return f"{label}:\n{run_shell_capture(command, keops_root, env)}\n"

    content = [
        f"timestamp: {timestamp}",
        f"machine: {machine}",
        f"hostname: {socket.gethostname()}",
        f"keops_root: {keops_root}",
        f"helper: {helper}",
        f"data_root: {data_root}",
        f"tmp_root: {tmp_root}",
        f"ground_truth_root: {ground_truth_root}",
        f"method: {METHOD}",
        f"engine: {ENGINE}",
        f"backend: {BACKEND}",
        f"cuda_home: {cuda_home}",
        f"timing_scope: {EXPECTED_TIMING_SCOPE}",
        f"git_branch: {git_branch}",
        f"git_commit: {git_commit}",
        "",
        capture("git_status", ["git", "status", "--short"]),
        capture("uname", ["uname", "-a"]),
        capture("lscpu", ["bash", "-lc", "lscpu | sed -n '1,40p'"]),
        capture(
            "nvidia-smi",
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
        ),
        capture("nvcc", ["nvcc", "--version"]),
        capture("gcc", ["bash", "-lc", "gcc --version | head -1"]),
        capture("g++", ["bash", "-lc", "g++ --version | head -1"]),
        capture("cmake", ["bash", "-lc", "cmake --version | head -1"]),
        capture("python", [python_bin, "--version"]),
    ]
    write_text(path, "\n".join(content))
