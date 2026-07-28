#!/usr/bin/env python3
"""Run resumable Stage 3 KeOps FP64 exact-KDE baselines on the remote H100."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, TextIO


MACHINE = "remote-h100"
METHOD = "KeOps"
WORKSPACE_ROOT = Path(
    "/home/lxheq/Documents/workspace/GPU-accelerated_Kernel_Density_Exact"
)
KEOPS_ROOT = WORKSPACE_ROOT / "baselines/keops-stage3"
EXACT_STAGE3_ROOT = WORKSPACE_ROOT / "GPU-kernel-density-exact-stage3"
HELPER = KEOPS_ROOT / "scripts/keops_exact_kde.py"
PYTHON_BIN = WORKSPACE_ROOT / "venvs/kde-baselines/bin/python"
CUDA_HOME = Path("/usr/local/cuda-12.4")
DATA_ROOT = Path(
    "/home/lxheq/Documents/workspace/dataset/"
    "GPU-accelerated_Kernel_Density_Computation"
)
GROUND_TRUTH_ROOT = DATA_ROOT / "exact/experiments/stage3/ground_truth"
GROUND_TRUTH_MANIFEST = GROUND_TRUTH_ROOT / "manifest.csv"
RUN_ROOT = (
    WORKSPACE_ROOT / "tmp-results/stage3/baselines/keops/remote-h100"
)
LOG_ROOT = RUN_ROOT / "logs"
RECORD_ROOT = RUN_ROOT / "records"
OUTPUT_ROOT = RUN_ROOT / "plain-outputs"
CACHE_ROOT = RUN_ROOT / "keops-cache"
SUMMARY_PATH = RUN_ROOT / "summary.csv"
COMMANDS_PATH = RUN_ROOT / "commands.sh"
INVENTORY_PATH = RUN_ROOT / "machine_inventory.txt"
CHECKSUM_PATH = RUN_ROOT / "SHA256SUMS"

EXPECTED_BRANCH = "stage3"
EXPECTED_TIMING_SCOPE = "end_to_end_in_memory"
TIMING_MODE = "cold_start"
TIMEOUT_SECONDS = 24 * 60 * 60
SCALE = "scott_diag b=1"
SCALE_ARGS = ("--scott-diag", "1")
PRECISION = "FP64"
DTYPE = "float64"
RELATIVE_TOLERANCE = 1e-5


@dataclass(frozen=True)
class Workload:
    name: str
    data_rows: int
    query_rows: int
    dimensions: int

    @property
    def dataset_dir(self) -> Path:
        return DATA_ROOT / self.name

    @property
    def data_path(self) -> Path:
        return self.dataset_dir / f"{self.name}_X.data.zst"

    @property
    def query_path(self) -> Path:
        return self.dataset_dir / f"{self.name}_qSet.data.zst"

    @property
    def reference_path(self) -> Path:
        return (
            GROUND_TRUTH_ROOT
            / f"{self.name}_basic-scan_fp64_scott_b1.out.zst"
        )

    @property
    def output_path(self) -> Path:
        return OUTPUT_ROOT / f"{self.name}_keops_fp64_scott_b1.out"

    @property
    def log_path(self) -> Path:
        return LOG_ROOT / f"{self.name}.log"

    @property
    def record_path(self) -> Path:
        return RECORD_ROOT / f"{self.name}.json"

    @property
    def cache_path(self) -> Path:
        return CACHE_ROOT / self.name


WORKLOADS = (
    Workload("vk_lsvd", 19_527_601, 100_000, 64),
    Workload("tencent_chinese_100", 12_187_936, 100_000, 100),
    Workload("tencent_english_200", 6_496_681, 100_000, 200),
    Workload("wolt_food_clip_512", 1_620_611, 100_000, 512),
    Workload("fd", 5_902_200, 100_000, 900),
    Workload("ocr", 4_070_000, 100_000, 1_156),
    Workload("epsilon", 400_000, 100_000, 2_000),
)


SUMMARY_FIELDS = [
    "machine",
    "workload",
    "method",
    "precision",
    "scale",
    "data_rows",
    "query_rows",
    "dimensions",
    "timing_mode",
    "timing_scope",
    "cold_execution_seconds",
    "qps",
    "correctness",
    "correctness_status",
    "failure_count",
    "output_count",
    "max_abs_err",
    "mean_abs_err",
    "max_rel_err",
    "mean_rel_err",
    "zero_reference_count",
    "nonzero_output_zero_reference_count",
    "full_command_wall_seconds",
    "batch_size",
    "batch_count",
    "keops_version",
    "data_sha256",
    "query_sha256",
    "reference_sha256",
    "temporary_output_sha256",
    "git_branch",
    "git_commit",
    "main_goal_commit",
    "log_path",
    "command",
    "accepted_at",
    "status",
]


def shell_join(command: Iterable[object]) -> str:
    return shlex.join([str(part) for part in command])


def require_paths(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required paths:\n  " + "\n  ".join(missing)
        )


def git_value(repo: Path, arguments: list[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip() or f"git failed in {repo}"
        )
    return completed.stdout.strip()


def require_clean_stage3() -> tuple[str, str]:
    branch = git_value(KEOPS_ROOT, ["rev-parse", "--abbrev-ref", "HEAD"])
    commit = git_value(KEOPS_ROOT, ["rev-parse", "HEAD"])
    status = git_value(
        KEOPS_ROOT, ["status", "--porcelain", "--untracked-files=all"]
    )
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(
            f"Expected branch {EXPECTED_BRANCH}, found {branch}."
        )
    if status:
        raise RuntimeError(
            "The KeOps Stage 3 worktree must be clean before an accepted run."
        )
    return branch, commit


def main_goal_commit() -> str:
    return git_value(
        EXACT_STAGE3_ROOT, ["rev-parse", "refs/remotes/origin/main"]
    )


def require_h100() -> None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap",
            "--format=csv,noheader",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "nvidia-smi failed")
    first_gpu = (
        completed.stdout.splitlines()[0]
        if completed.stdout.splitlines()
        else ""
    )
    if "H100" not in first_gpu or "9.0" not in first_gpu:
        raise RuntimeError(
            "Expected an H100 with compute capability 9.0; "
            f"found {first_gpu!r}."
        )


def run_env(cache_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    cuda_library_paths = [
        CUDA_HOME / "lib64",
        CUDA_HOME / "targets/x86_64-linux/lib",
    ]
    existing_cuda_paths = [
        str(path) for path in cuda_library_paths if path.exists()
    ]
    environment["PYTHONNOUSERSITE"] = "1"
    environment["CUDA_HOME"] = str(CUDA_HOME)
    environment["PATH"] = os.pathsep.join(
        [
            str(PYTHON_BIN.parent),
            str(CUDA_HOME / "bin"),
            environment.get("PATH", ""),
        ]
    )
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [*existing_cuda_paths, environment.get("LD_LIBRARY_PATH", "")]
    )
    environment["LIBRARY_PATH"] = os.pathsep.join(
        [*existing_cuda_paths, environment.get("LIBRARY_PATH", "")]
    )
    environment["CPLUS_INCLUDE_PATH"] = os.pathsep.join(
        [
            str(CUDA_HOME / "include"),
            environment.get("CPLUS_INCLUDE_PATH", ""),
        ]
    )
    environment["KEOPS_CACHE_FOLDER"] = str(cache_path)

    site_packages = PYTHON_BIN.parents[1] / "lib/python3.10/site-packages"
    library_paths = sorted(site_packages.glob("lib*/lib64"))
    library_paths.extend(sorted((site_packages / "nvidia").glob("*/lib")))
    existing_library_paths = [
        str(path) for path in library_paths if path.exists()
    ]
    if existing_library_paths:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            [
                *existing_library_paths,
                environment.get("LD_LIBRARY_PATH", ""),
            ]
        )
        environment["LIBRARY_PATH"] = os.pathsep.join(
            [
                *existing_library_paths,
                environment.get("LIBRARY_PATH", ""),
            ]
        )
    return environment


def keops_version(environment: dict[str, str]) -> str:
    completed = subprocess.run(
        [
            str(PYTHON_BIN),
            "-c",
            "import pykeops; print(pykeops.__version__)",
        ],
        cwd=KEOPS_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip() or "Could not read the PyKeOps version."
        )
    output_lines = [
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    ]
    if not output_lines:
        raise RuntimeError("PyKeOps did not report a package version.")
    return output_lines[-1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_zstd_header(path: Path) -> tuple[int, int]:
    process = subprocess.Popen(
        ["zstd", "-q", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.stdout is None:
        raise RuntimeError(f"Could not read {path}")
    first_line = process.stdout.readline()
    process.stdout.close()
    process.terminate()
    process.wait()
    if process.stderr is not None:
        process.stderr.close()
    fields = first_line.split()
    if len(fields) != 2:
        raise ValueError(f"Invalid matrix header in {path}")
    return int(fields[0]), int(fields[1])


def load_ground_truth_manifest() -> dict[str, dict[str, str]]:
    with GROUND_TRUTH_MANIFEST.open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    by_workload = {row["workload"]: row for row in rows}
    expected_names = {workload.name for workload in WORKLOADS}
    if len(rows) != len(WORKLOADS) or set(by_workload) != expected_names:
        raise ValueError(
            "The ground-truth manifest workload names do not match the "
            "Stage 3 KeOps workload list."
        )
    for workload in WORKLOADS:
        row = by_workload[workload.name]
        expected = {
            "method": "Basic-Scan",
            "precision": PRECISION,
            "scale": SCALE,
            "data_rows": str(workload.data_rows),
            "query_rows": str(workload.query_rows),
            "dimensions": str(workload.dimensions),
            "output_rows": str(workload.query_rows),
            "status": "ok",
        }
        for name, value in expected.items():
            if row.get(name) != value:
                raise ValueError(
                    f"The Work 3 record for {workload.name} has "
                    f"{name}={row.get(name)!r}; expected {value!r}."
                )
    return by_workload


def validate_inputs(
    ground_truth_manifest: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    checksums: dict[str, dict[str, str]] = {}
    for workload in WORKLOADS:
        require_paths(
            [
                workload.data_path,
                workload.query_path,
                workload.reference_path,
            ]
        )
        actual_data_shape = read_zstd_header(workload.data_path)
        actual_query_shape = read_zstd_header(workload.query_path)
        expected_data_shape = (workload.data_rows, workload.dimensions)
        expected_query_shape = (workload.query_rows, workload.dimensions)
        if actual_data_shape != expected_data_shape:
            raise ValueError(
                f"{workload.data_path} has shape {actual_data_shape}; "
                f"expected {expected_data_shape}."
            )
        if actual_query_shape != expected_query_shape:
            raise ValueError(
                f"{workload.query_path} has shape {actual_query_shape}; "
                f"expected {expected_query_shape}."
            )
        print(f"[stage3-keops] hashing {workload.name} inputs", flush=True)
        actual_checksums = {
            "data": sha256_file(workload.data_path),
            "query": sha256_file(workload.query_path),
            "reference": sha256_file(workload.reference_path),
        }
        manifest_row = ground_truth_manifest[workload.name]
        expected_checksums = {
            "data": manifest_row["data_sha256"],
            "query": manifest_row["query_sha256"],
            "reference": manifest_row["output_zst_sha256"],
        }
        if actual_checksums != expected_checksums:
            raise ValueError(
                f"{workload.name} input or reference checksum differs "
                "from the accepted Work 3 manifest."
            )
        checksums[workload.name] = actual_checksums
    return checksums


def open_vector(path: Path) -> tuple[TextIO, subprocess.Popen[str] | None]:
    if path.name.endswith(".zst"):
        process = subprocess.Popen(
            ["zstd", "-q", "-dc", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if process.stdout is None:
            raise RuntimeError(f"Could not open {path}")
        return process.stdout, process
    return path.open("r", encoding="utf-8"), None


def close_vector(
    handle: TextIO, process: subprocess.Popen[str] | None
) -> None:
    handle.close()
    if process is None:
        return
    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            stderr.strip()
            or f"zstd failed with return code {return_code}"
        )


def iter_floats(path: Path) -> Iterator[float]:
    handle, process = open_vector(path)
    try:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                raise ValueError(f"Blank line {line_number} in {path}")
            for field in fields:
                yield float(field)
    finally:
        close_vector(handle, process)


def compare_values(
    output_values: Iterable[float],
    reference_values: Iterable[float],
    expected_count: int,
) -> dict[str, object]:
    outputs = list(output_values)
    references = list(reference_values)
    output_count = len(outputs)
    reference_count = len(references)
    compared_count = min(output_count, reference_count)
    failure_count = 0
    max_absolute_error = 0.0
    sum_absolute_error = 0.0
    max_relative_error = 0.0
    sum_relative_error = 0.0
    zero_reference_count = 0
    nonzero_output_zero_reference_count = 0

    for output, reference in zip(outputs, references):
        output = float(output)
        reference = float(reference)
        if not math.isfinite(output) or not math.isfinite(reference):
            failure_count += 1
            max_absolute_error = math.inf
            sum_absolute_error = math.inf
            max_relative_error = math.inf
            sum_relative_error = math.inf
            continue

        absolute_error = abs(output - reference)
        max_absolute_error = max(max_absolute_error, absolute_error)
        sum_absolute_error += absolute_error

        if reference == 0.0:
            zero_reference_count += 1
            if output != 0.0:
                failure_count += 1
                nonzero_output_zero_reference_count += 1
                relative_error = math.inf
            else:
                relative_error = 0.0
        else:
            relative_error = absolute_error / abs(reference)
            if relative_error > RELATIVE_TOLERANCE:
                failure_count += 1
        max_relative_error = max(max_relative_error, relative_error)
        sum_relative_error += relative_error

    if (
        output_count != expected_count
        or reference_count != expected_count
        or output_count != reference_count
    ):
        failure_count = max(1, failure_count)
        correctness_status = "count_mismatch"
    else:
        correctness_status = "compared"

    mean_absolute_error = (
        sum_absolute_error / compared_count if compared_count else math.inf
    )
    mean_relative_error = (
        sum_relative_error / compared_count if compared_count else math.inf
    )
    return {
        "correctness": str(
            correctness_status == "compared" and failure_count == 0
        ).lower(),
        "correctness_status": correctness_status,
        "failure_count": failure_count,
        "output_count": output_count,
        "max_abs_err": max_absolute_error,
        "mean_abs_err": mean_absolute_error,
        "max_rel_err": max_relative_error,
        "mean_rel_err": mean_relative_error,
        "zero_reference_count": zero_reference_count,
        "nonzero_output_zero_reference_count": (
            nonzero_output_zero_reference_count
        ),
    }


def compare_output(
    output_path: Path, reference_path: Path, expected_count: int
) -> dict[str, object]:
    require_paths([output_path, reference_path])
    return compare_values(
        iter_floats(output_path),
        iter_floats(reference_path),
        expected_count,
    )


def parse_metrics(log_text: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for key in (
        "timing_scope",
        "timing_mode",
        "warmup_policy",
        "execution_seconds",
        "query_count",
        "batch_size",
        "batch_count",
        "keops_import_seconds",
        "pre_timer_total_seconds",
        "qps",
    ):
        match = re.search(
            rf"^{key}:\s*(.+?)\s*$", log_text, re.MULTILINE
        )
        if match:
            metrics[key] = match.group(1)
    return metrics


def validate_metrics(
    metrics: dict[str, str], workload: Workload
) -> None:
    expected = {
        "timing_scope": EXPECTED_TIMING_SCOPE,
        "timing_mode": TIMING_MODE,
        "warmup_policy": "none",
        "query_count": str(workload.query_rows),
        "batch_size": str(workload.query_rows),
        "batch_count": "1",
    }
    for name, value in expected.items():
        if metrics.get(name) != value:
            raise ValueError(
                f"{workload.name} reported {name}={metrics.get(name)!r}; "
                f"expected {value!r}."
            )
    for name in ("execution_seconds", "qps"):
        value = float(metrics.get(name, "nan"))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{workload.name} reported invalid {name}: {value}"
            )
    execution_seconds = float(metrics["execution_seconds"])
    reported_qps = float(metrics["qps"])
    expected_qps = workload.query_rows / execution_seconds
    if not math.isclose(reported_qps, expected_qps, rel_tol=1e-8):
        raise ValueError(
            f"{workload.name} reported QPS {reported_qps}; "
            f"expected {expected_qps} from its execution time."
        )


def make_command(
    workload: Workload, output_path: Path
) -> list[object]:
    return [
        PYTHON_BIN,
        HELPER,
        "svm",
        workload.query_path,
        workload.data_path,
        output_path,
        *SCALE_ARGS,
        "--batch-size",
        str(workload.query_rows),
        "--engine",
        "keops",
        "--backend",
        "GPU",
        "--dtype",
        DTYPE,
        "--timing-mode",
        TIMING_MODE,
        "--local-keops-root",
        KEOPS_ROOT,
    ]


def append_command(command: list[object]) -> None:
    write_header = not COMMANDS_PATH.exists()
    with COMMANDS_PATH.open("a", encoding="utf-8") as handle:
        if write_header:
            handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        handle.write(f"cd {shlex.quote(str(KEOPS_ROOT))}\n")
        handle.write(
            f"# timeout_seconds: {TIMEOUT_SECONDS}\n"
            f"{shell_join(command)}\n\n"
        )
    COMMANDS_PATH.chmod(0o755)


def run_command(
    command: list[object],
    environment: dict[str, str],
    log_path: Path,
) -> tuple[float, str]:
    append_command(command)
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=KEOPS_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        wall_seconds = time.perf_counter() - start
        partial_output = error.stdout or error.output or ""
        if isinstance(partial_output, bytes):
            partial_output = partial_output.decode(
                "utf-8", errors="replace"
            )
        log_path.write_text(
            f"$ {shell_join(command)}\n"
            f"timeout_seconds: {TIMEOUT_SECONDS}\n"
            f"full_command_wall_seconds: {wall_seconds:.9f}\n\n"
            f"{partial_output}",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Command timed out after {TIMEOUT_SECONDS} seconds; "
            f"see {log_path}."
        ) from error

    wall_seconds = time.perf_counter() - start
    log_text = (
        f"$ {shell_join(command)}\n"
        f"return_code: {completed.returncode}\n"
        f"full_command_wall_seconds: {wall_seconds:.9f}\n\n"
        f"{completed.stdout or ''}"
    )
    log_path.write_text(log_text, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {completed.returncode}; "
            f"see {log_path}."
        )
    return wall_seconds, log_text


def format_number(value: object) -> str:
    number = float(value)
    if math.isinf(number):
        return "inf"
    return f"{number:.17g}"


def write_json_atomic(path: Path, value: dict[str, object]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".partial")
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_record(
    workload: Workload,
    checksums: dict[str, str],
    branch: str,
    commit: str,
    goal_commit: str,
) -> dict[str, object] | None:
    if not workload.record_path.exists():
        return None
    record = json.loads(workload.record_path.read_text(encoding="utf-8"))
    expected: dict[str, object] = {
        "workload": workload.name,
        "git_branch": branch,
        "git_commit": commit,
        "main_goal_commit": goal_commit,
        "data_sha256": checksums["data"],
        "query_sha256": checksums["query"],
        "reference_sha256": checksums["reference"],
        "status": "ok",
    }
    for name, value in expected.items():
        if record.get(name) != value:
            raise RuntimeError(
                f"{workload.record_path} has {name}="
                f"{record.get(name)!r}; expected {value!r}."
            )
    workload.output_path.unlink(missing_ok=True)
    return record


def write_summary(records: list[dict[str, object]]) -> None:
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def write_inventory(
    environment: dict[str, str],
    branch: str,
    commit: str,
    goal_commit: str,
    version: str,
) -> None:
    commands = [
        ["hostname"],
        ["date", "-Is"],
        ["free", "-h"],
        ["df", "-h", str(DATA_ROOT), str(WORKSPACE_ROOT)],
        ["nvidia-smi"],
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,memory.total",
            "--format=csv",
        ],
        [str(PYTHON_BIN), "--version"],
        [
            str(PYTHON_BIN),
            "-c",
            (
                "import numpy, pykeops, zstandard; "
                "print('numpy', numpy.__version__); "
                "print('pykeops', pykeops.__version__); "
                "print('zstandard', zstandard.__version__)"
            ),
        ],
        ["git", "-C", str(KEOPS_ROOT), "status", "--short", "--branch"],
        ["git", "-C", str(KEOPS_ROOT), "rev-parse", "HEAD"],
    ]
    with INVENTORY_PATH.open("w", encoding="utf-8") as handle:
        handle.write(f"machine: {MACHINE}\n")
        handle.write(f"hostname: {socket.gethostname()}\n")
        handle.write(f"keops_root: {KEOPS_ROOT}\n")
        handle.write(f"python_bin: {PYTHON_BIN}\n")
        handle.write(f"data_root: {DATA_ROOT}\n")
        handle.write(f"ground_truth_root: {GROUND_TRUTH_ROOT}\n")
        handle.write(f"run_root: {RUN_ROOT}\n")
        handle.write(f"git_branch: {branch}\n")
        handle.write(f"git_commit: {commit}\n")
        handle.write(f"main_goal_commit: {goal_commit}\n")
        handle.write(f"keops_version: {version}\n")
        handle.write(f"relative_tolerance: {RELATIVE_TOLERANCE:.17g}\n")
        handle.write("absolute_tolerance: none\n")
        handle.write("zero_reference_rule: exact_zero\n")
        handle.write(f"timeout_seconds: {TIMEOUT_SECONDS}\n\n")
        for command in commands:
            handle.write(f"$ {shell_join(command)}\n")
            completed = subprocess.run(
                [str(part) for part in command],
                cwd=KEOPS_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            handle.write(completed.stdout or "")
            handle.write("\n")


def write_sha256sums() -> None:
    rows = []
    for path in sorted(RUN_ROOT.rglob("*")):
        if (
            path.is_file()
            and path != CHECKSUM_PATH
            and not path.is_relative_to(OUTPUT_ROOT)
            and not path.is_relative_to(CACHE_ROOT)
        ):
            rows.append(
                f"{sha256_file(path)}  ./{path.relative_to(RUN_ROOT)}"
            )
    CHECKSUM_PATH.write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def prepare_directories() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def run_workload(
    workload: Workload,
    checksums: dict[str, str],
    branch: str,
    commit: str,
    goal_commit: str,
    version: str,
) -> dict[str, object]:
    workload.output_path.unlink(missing_ok=True)
    shutil.rmtree(workload.cache_path, ignore_errors=True)
    workload.cache_path.mkdir(parents=True)
    environment = run_env(workload.cache_path)
    command = make_command(workload, workload.output_path)
    print(
        f"[stage3-keops] running {workload.name} cold_start", flush=True
    )
    wall_seconds, log_text = run_command(
        command, environment, workload.log_path
    )
    metrics = parse_metrics(log_text)
    validate_metrics(metrics, workload)
    correctness = compare_output(
        workload.output_path,
        workload.reference_path,
        workload.query_rows,
    )
    temporary_output_sha256 = sha256_file(workload.output_path)

    record: dict[str, object] = {
        "machine": MACHINE,
        "workload": workload.name,
        "method": METHOD,
        "precision": PRECISION,
        "scale": SCALE,
        "data_rows": workload.data_rows,
        "query_rows": workload.query_rows,
        "dimensions": workload.dimensions,
        "timing_mode": TIMING_MODE,
        "timing_scope": metrics["timing_scope"],
        "cold_execution_seconds": metrics["execution_seconds"],
        "qps": metrics["qps"],
        "correctness": correctness["correctness"],
        "correctness_status": correctness["correctness_status"],
        "failure_count": correctness["failure_count"],
        "output_count": correctness["output_count"],
        "max_abs_err": format_number(correctness["max_abs_err"]),
        "mean_abs_err": format_number(correctness["mean_abs_err"]),
        "max_rel_err": format_number(correctness["max_rel_err"]),
        "mean_rel_err": format_number(correctness["mean_rel_err"]),
        "zero_reference_count": correctness["zero_reference_count"],
        "nonzero_output_zero_reference_count": correctness[
            "nonzero_output_zero_reference_count"
        ],
        "full_command_wall_seconds": f"{wall_seconds:.9f}",
        "batch_size": metrics["batch_size"],
        "batch_count": metrics["batch_count"],
        "keops_version": version,
        "data_sha256": checksums["data"],
        "query_sha256": checksums["query"],
        "reference_sha256": checksums["reference"],
        "temporary_output_sha256": temporary_output_sha256,
        "git_branch": branch,
        "git_commit": commit,
        "main_goal_commit": goal_commit,
        "log_path": str(workload.log_path),
        "command": shell_join(command),
        "accepted_at": datetime.now().astimezone().isoformat(),
        "status": "ok",
    }
    write_json_atomic(workload.record_path, record)
    workload.output_path.unlink()
    return record


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError(
            "This runner uses hard-coded Stage 3 paths and accepts no arguments."
        )

    require_paths(
        [
            KEOPS_ROOT,
            EXACT_STAGE3_ROOT,
            HELPER,
            PYTHON_BIN,
            CUDA_HOME,
            DATA_ROOT,
            GROUND_TRUTH_ROOT,
            GROUND_TRUTH_MANIFEST,
            Path("/usr/bin/zstd"),
        ]
    )
    branch, commit = require_clean_stage3()
    goal_commit = main_goal_commit()
    require_h100()
    prepare_directories()

    inventory_environment = run_env(CACHE_ROOT / "inventory")
    version = keops_version(inventory_environment)
    write_inventory(
        inventory_environment, branch, commit, goal_commit, version
    )
    ground_truth_manifest = load_ground_truth_manifest()
    checksums = validate_inputs(ground_truth_manifest)

    records: list[dict[str, object]] = []
    for workload in WORKLOADS:
        existing_record = load_record(
            workload,
            checksums[workload.name],
            branch,
            commit,
            goal_commit,
        )
        if existing_record is not None:
            print(
                f"[stage3-keops] accepted; skipping {workload.name}",
                flush=True,
            )
            records.append(existing_record)
        else:
            records.append(
                run_workload(
                    workload,
                    checksums[workload.name],
                    branch,
                    commit,
                    goal_commit,
                    version,
                )
            )
            print(
                f"[stage3-keops] accepted {workload.name}", flush=True
            )
        write_summary(records)
        write_sha256sums()

    shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
    write_sha256sums()
    print(f"[stage3-keops] run_root={RUN_ROOT}")
    print(f"[stage3-keops] summary={SUMMARY_PATH}")
    print(
        "[stage3-keops] temporary KeOps caches remain until the compact "
        "records are downloaded and verified"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[stage3-keops][ERROR] {error}", file=sys.stderr)
        raise SystemExit(1)
