#!/usr/bin/env python3
"""KeOps adapter for exact raw Gaussian kernel aggregation.

The script mirrors the kernel-scale convention in GPU-kernel-density-exact:
it embeds scalar gamma or diagonal Scott scaling into both data and query
coordinates, then evaluates

    F(q) = sum_x exp(-||q - x||^2)

with PyKeOps.  It supports both point-query SVM/KAQ workloads and 2D KDV grid
workloads.  A small NumPy engine is included only for smoke tests.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import numpy as np

from stage2_dense_io import read_dense_matrix, read_dense_vector


def read_matrix(path: str | Path, dtype: np.dtype) -> np.ndarray:
    return read_dense_matrix(path, dtype)


def read_weights(path: str | Path, expected_rows: int, dtype: np.dtype) -> np.ndarray:
    weights = read_dense_vector(path, dtype)
    if weights.size != expected_rows:
        raise ValueError(
            f"{path}: expected {expected_rows} weights, found {weights.size}"
        )
    return weights


def scalar_gamma_coefficients(data: np.ndarray, gamma: float) -> np.ndarray:
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("--gamma must be finite and non-negative")
    return np.full(data.shape[1], np.sqrt(gamma), dtype=data.dtype)


def scott_diag_coefficients(data: np.ndarray, b: float) -> np.ndarray:
    if not np.isfinite(b) or b <= 0:
        raise ValueError("--scott-diag/--b must be positive")
    rows, dim = data.shape
    n_factor = float(rows) ** (-1.0 / float(dim + 4))
    std = data.std(axis=0, ddof=0)
    if np.any(std <= 0):
        bad = np.nonzero(std <= 0)[0]
        raise ValueError(
            "Scott diagonal scale is undefined for zero-variance dimensions: "
            + ", ".join(str(int(i)) for i in bad)
        )
    h = b * std * n_factor
    gamma = 1.0 / (2.0 * h * h)
    coeff = np.sqrt(gamma).astype(data.dtype, copy=False)
    return coeff


def scale_coefficients(data: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.gamma is not None:
        return scalar_gamma_coefficients(data, args.gamma)
    return scott_diag_coefficients(data, args.scott_diag)


def scale_description(args: argparse.Namespace) -> str:
    if args.gamma is not None:
        return f"gamma={args.gamma}"
    return f"scott_diag_b={args.scott_diag}"


def import_keops(local_keops_root: Optional[str], allow_local: bool):
    if local_keops_root is not None:
        root = Path(local_keops_root).resolve()
    else:
        root = Path(__file__).resolve().parents[1]

    if allow_local:
        pykeops_dir = root / "pykeops"
        keopscore_dir = root / "keopscore"
        if pykeops_dir.exists() and keopscore_dir.exists():
            sys.path.insert(0, str(pykeops_dir))
            sys.path.insert(0, str(keopscore_dir))

            try:
                from pykeops.numpy import LazyTensor

                return LazyTensor
            except Exception as local_error:
                if local_keops_root is not None:
                    raise RuntimeError(
                        "Could not import PyKeOps from the requested local clone. "
                        "Install dependencies with something like:\n"
                        "  python3 -m pip install pybind11 numpy\n"
                        f"Tried local KeOps root: {root}"
                    ) from local_error

    try:
        from pykeops.numpy import LazyTensor

        return LazyTensor
    except Exception as installed_error:
        raise RuntimeError(
            "Could not import PyKeOps. Install dependencies with something like:\n"
            "  python3 -m pip install pybind11 numpy\n"
            "or pass --local-keops-root to the cloned KeOps repository."
        ) from installed_error


def keops_sum(
    data: np.ndarray,
    queries: np.ndarray,
    weights: Optional[np.ndarray],
    backend: str,
    LazyTensor,
) -> np.ndarray:
    q_i = LazyTensor(queries[:, None, :])
    x_j = LazyTensor(data[None, :, :])
    d2 = ((q_i - x_j) ** 2).sum(-1)
    kernel = (-d2).exp()
    if weights is not None:
        w_j = LazyTensor(weights[None, :, None])
        kernel = kernel * w_j

    kwargs = {} if backend == "auto" else {"backend": backend}
    return np.asarray(kernel.sum(axis=1, **kwargs)).reshape(-1)


def numpy_sum(
    data: np.ndarray,
    queries: np.ndarray,
    weights: Optional[np.ndarray],
    data_batch_size: int,
) -> np.ndarray:
    out = np.zeros(queries.shape[0], dtype=np.float64)
    w = None if weights is None else weights.astype(np.float64, copy=False)
    q = queries.astype(np.float64, copy=False)
    x_all = data.astype(np.float64, copy=False)

    for start in range(0, x_all.shape[0], data_batch_size):
        x = x_all[start : start + data_batch_size]
        d2 = ((q[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
        contrib = np.exp(-d2)
        if w is not None:
            contrib *= w[start : start + x.shape[0]][None, :]
        out += contrib.sum(axis=1)
    return out.astype(data.dtype, copy=False)


def compute_sum(
    engine: str,
    data: np.ndarray,
    queries: np.ndarray,
    weights: Optional[np.ndarray],
    backend: str,
    LazyTensor,
    data_batch_size: int,
) -> np.ndarray:
    if engine == "keops":
        try:
            return keops_sum(data, queries, weights, backend, LazyTensor)
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", "")
            if missing.startswith("pykeops_"):
                raise RuntimeError(
                    "PyKeOps imported, but its JIT extension module was not built. "
                    "Install pybind11 (and pip if needed), then rerun:\n"
                    "  python3 -m pip install pybind11\n"
                    f"Missing module: {missing}"
                ) from e
            raise
    if engine == "numpy":
        return numpy_sum(data, queries, weights, data_batch_size)
    raise ValueError(f"Unsupported engine: {engine}")


def batched_ranges(total: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total)


def batch_count(total: int, batch_size: int) -> int:
    return (total + batch_size - 1) // batch_size


def write_vector(path: str | Path, values: Iterable[np.ndarray]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for batch in values:
            for value in np.asarray(batch).reshape(-1):
                f.write(f"{float(value):.17g}\n")


def timed_run(args, compute_output: Callable[[], np.ndarray]) -> Tuple[float, np.ndarray, str, float]:
    warmup_policy = "none"
    warmup_elapsed = 0.0
    if args.timing_mode == "warm":
        warmup_start = time.perf_counter()
        compute_output()
        warmup_elapsed = time.perf_counter() - warmup_start
        warmup_policy = "full_untimed_keops_call" if args.engine == "keops" else "full_untimed_numpy_call"

    start_time = time.perf_counter()
    output = compute_output()
    elapsed = time.perf_counter() - start_time
    return elapsed, output, warmup_policy, warmup_elapsed


def run_svm(
    args,
    data: np.ndarray,
    coeff: np.ndarray,
    LazyTensor,
) -> Tuple[float, int, int, str, float, dict[str, float]]:
    query_read_start = time.perf_counter()
    query = read_matrix(args.query, data.dtype)
    query_read_elapsed = time.perf_counter() - query_read_start
    if query.shape[1] != data.shape[1]:
        raise ValueError(
            f"dimension mismatch: data dim={data.shape[1]}, query dim={query.shape[1]}"
        )

    data_scale_start = time.perf_counter()
    data_scaled = data * coeff
    data_scale_elapsed = time.perf_counter() - data_scale_start
    query_scale_start = time.perf_counter()
    query *= coeff
    query_scale_elapsed = time.perf_counter() - query_scale_start

    weights = None
    weights_read_elapsed = 0.0
    if args.weights is not None:
        weights_read_start = time.perf_counter()
        weights = read_weights(args.weights, data.shape[0], data.dtype)
        weights_read_elapsed = time.perf_counter() - weights_read_start

    data_split_rows = data.shape[0]
    if args.engine == "keops" and args.max_data_scalars > 0:
        data_split_rows = min(
            data.shape[0],
            max(1, args.max_data_scalars // data.shape[1]),
        )
    data_split_count = batch_count(data.shape[0], data_split_rows)

    def compute_output() -> np.ndarray:
        output = np.empty(query.shape[0], dtype=data.dtype)
        for start, end in batched_ranges(query.shape[0], args.batch_size):
            if data_split_count == 1:
                output[start:end] = compute_sum(
                    args.engine,
                    data_scaled,
                    query[start:end],
                    weights,
                    args.backend,
                    LazyTensor,
                    args.data_batch_size,
                )
                continue

            query_output = np.zeros(end - start, dtype=data.dtype)
            for data_start, data_end in batched_ranges(
                data.shape[0], data_split_rows
            ):
                split_weights = (
                    None
                    if weights is None
                    else weights[data_start:data_end]
                )
                query_output += compute_sum(
                    args.engine,
                    data_scaled[data_start:data_end],
                    query[start:end],
                    split_weights,
                    args.backend,
                    LazyTensor,
                    args.data_batch_size,
                )
            output[start:end] = query_output
        return output

    elapsed, output, warmup_policy, warmup_elapsed = timed_run(args, compute_output)
    write_vector(args.output, (output,))
    prep_times = {
        "query_read_seconds": query_read_elapsed,
        "query_construct_seconds": 0.0,
        "data_scale_seconds": data_scale_elapsed,
        "query_scale_seconds": query_scale_elapsed,
        "weights_read_seconds": weights_read_elapsed,
        "data_split_rows": data_split_rows,
        "data_split_count": data_split_count,
    }
    return (
        elapsed,
        query.shape[0],
        batch_count(query.shape[0], args.batch_size),
        warmup_policy,
        warmup_elapsed,
        prep_times,
    )


def run_kdv(
    args,
    data: np.ndarray,
    coeff: np.ndarray,
    LazyTensor,
) -> Tuple[float, int, int, str, float, dict[str, float]]:
    if data.shape[1] != 2:
        raise ValueError(f"KDV mode requires 2D data, found dim={data.shape[1]}")

    data_scale_start = time.perf_counter()
    data_scaled = data * coeff
    data_scale_elapsed = time.perf_counter() - data_scale_start

    query_construct_start = time.perf_counter()
    row_l = float(data[:, 0].min())
    row_u = float(data[:, 0].max())
    col_l = float(data[:, 1].min())
    col_u = float(data[:, 1].max())
    row_incr = 0.0 if args.rows <= 1 else (row_u - row_l) / float(args.rows - 1)
    col_incr = 0.0 if args.cols <= 1 else (col_u - col_l) / float(args.cols - 1)

    weights = None
    weights_read_elapsed = 0.0
    if args.weights is not None:
        weights_read_start = time.perf_counter()
        weights = read_weights(args.weights, data.shape[0], data.dtype)
        weights_read_elapsed = time.perf_counter() - weights_read_start

    total_queries = args.rows * args.cols
    row_coords = row_l + np.arange(args.rows, dtype=data.dtype) * row_incr
    col_coords = col_l + np.arange(args.cols, dtype=data.dtype) * col_incr
    query = np.empty((total_queries, 2), dtype=data.dtype)
    query[:, 0] = np.repeat(row_coords, args.cols)
    query[:, 1] = np.tile(col_coords, args.rows)
    query_construct_elapsed = time.perf_counter() - query_construct_start
    query_scale_start = time.perf_counter()
    query *= coeff
    query_scale_elapsed = time.perf_counter() - query_scale_start

    def compute_output() -> np.ndarray:
        output = np.empty(total_queries, dtype=data.dtype)
        for start, end in batched_ranges(total_queries, args.batch_size):
            output[start:end] = compute_sum(
                args.engine,
                data_scaled,
                query[start:end],
                weights,
                args.backend,
                LazyTensor,
                args.data_batch_size,
            )
        return output

    elapsed, output, warmup_policy, warmup_elapsed = timed_run(args, compute_output)

    grid = output.reshape(args.rows, args.cols)
    with Path(args.output).open("w", encoding="utf-8") as f:
        for row in grid:
            f.write(" ".join(f"{float(v):.17g}" for v in row))
            f.write("\n")
    prep_times = {
        "query_read_seconds": 0.0,
        "query_construct_seconds": query_construct_elapsed,
        "data_scale_seconds": data_scale_elapsed,
        "query_scale_seconds": query_scale_elapsed,
        "weights_read_seconds": weights_read_elapsed,
    }
    return (
        elapsed,
        total_queries,
        batch_count(total_queries, args.batch_size),
        warmup_policy,
        warmup_elapsed,
        prep_times,
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("data", help="Exact KDE .data matrix file")
    parser.add_argument("output", help="Output file")
    scale_group = parser.add_mutually_exclusive_group(required=True)
    scale_group.add_argument("--gamma", type=float, help="Scalar Gaussian scale")
    scale_group.add_argument(
        "--scott-diag",
        "--b",
        dest="scott_diag",
        type=float,
        help="Diagonal Scott multiplier b",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--timing-mode",
        choices=("cold_start", "warm"),
        default="cold_start",
        help=(
            "cold_start measures one pass in this process; warm runs one full "
            "untimed pass before measuring the second pass."
        ),
    )
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--weights", default=None, help="Optional one-value-per-row weights")
    parser.add_argument(
        "--engine",
        choices=("keops", "numpy"),
        default="keops",
        help="Use 'keops' for the baseline; 'numpy' is only for smoke tests.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "GPU", "CPU"),
        default="auto",
        help="PyKeOps reduction backend. Ignored by --engine numpy.",
    )
    parser.add_argument(
        "--data-batch-size",
        type=int,
        default=16384,
        help="Only used by --engine numpy.",
    )
    parser.add_argument(
        "--max-data-scalars",
        type=int,
        default=0,
        help=(
            "For KeOps, split the data reduction so each split contains at "
            "most this many scalar coordinates. Zero disables splitting."
        ),
    )
    parser.add_argument(
        "--local-keops-root",
        default=None,
        help="Path to cloned KeOps repo. Defaults to ../keops from this script.",
    )
    parser.add_argument(
        "--no-local-keops",
        action="store_true",
        help="Only import installed pykeops; do not add the local clone to sys.path.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Deprecated alias for --timing-mode cold_start.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact KeOps baseline for raw Gaussian SVM/KDV tasks."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    svm = subparsers.add_parser("svm", help="Point-query KAQ/SVM-style workload")
    svm.add_argument("query", help="Exact KDE .data query matrix file")
    add_common_args(svm)

    kdv = subparsers.add_parser("kdv", help="2D KDV grid workload")
    kdv.add_argument("--rows", type=int, required=True)
    kdv.add_argument("--cols", type=int, required=True)
    add_common_args(kdv)

    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.data_batch_size <= 0:
        parser.error("--data-batch-size must be positive")
    if args.max_data_scalars < 0:
        parser.error("--max-data-scalars must be non-negative")
    if args.no_warmup:
        args.timing_mode = "cold_start"
    return args


def main() -> int:
    args = parse_args()
    dtype = np.dtype(args.dtype)

    LazyTensor = None
    keops_import_elapsed = 0.0
    if args.engine == "keops":
        import_start = time.perf_counter()
        LazyTensor = import_keops(args.local_keops_root, not args.no_local_keops)
        keops_import_elapsed = time.perf_counter() - import_start

    data_read_start = time.perf_counter()
    data = read_matrix(args.data, dtype)
    data_read_elapsed = time.perf_counter() - data_read_start
    scale_coeff_start = time.perf_counter()
    coeff = scale_coefficients(data, args)
    scale_coeff_elapsed = time.perf_counter() - scale_coeff_start

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "svm":
        elapsed, query_count, batches, warmup_policy, warmup_elapsed, prep_times = run_svm(args, data, coeff, LazyTensor)
    elif args.mode == "kdv":
        elapsed, query_count, batches, warmup_policy, warmup_elapsed, prep_times = run_kdv(args, data, coeff, LazyTensor)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    pre_timer_total = (
        keops_import_elapsed
        + data_read_elapsed
        + scale_coeff_elapsed
        + sum(
            value
            for name, value in prep_times.items()
            if name.endswith("_seconds")
        )
    )
    data_split_rows = int(prep_times.get("data_split_rows", data.shape[0]))
    data_split_count = int(prep_times.get("data_split_count", 1))
    qps = float(query_count) / elapsed if elapsed > 0 else float("inf")
    print(f"Mode: {args.mode}")
    print(f"Engine: {args.engine}")
    print(f"Backend: {args.backend if args.engine == 'keops' else 'n/a'}")
    print(f"Data: size={data.shape[0]}, dim={data.shape[1]}")
    print(f"Kernel scale: {scale_description(args)}")
    print(f"Query count: {query_count}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batch count: {batches}")
    print(f"Data split rows: {data_split_rows}")
    print(f"Data split count: {data_split_count}")
    print(f"Timing mode: {args.timing_mode}")
    print(f"Warmup policy: {warmup_policy}")
    print(f"Warmup call time: {warmup_elapsed:.6f} seconds")
    print(f"Load/preprocess time: {pre_timer_total:.6f} seconds")
    print(f"Elapsed time: {elapsed:.6f} seconds")
    method_label = "KeOps" if args.engine == "keops" else args.engine
    print(f"Method {method_label}: {qps:.6f} Queries/sec")
    print("timing_scope: end_to_end_in_memory")
    print(f"timing_mode: {args.timing_mode}")
    print(f"warmup_policy: {warmup_policy}")
    print(f"warmup_call_seconds: {warmup_elapsed:.9f}")
    print(f"execution_seconds: {elapsed:.9f}")
    print(f"query_count: {query_count}")
    print(f"batch_size: {args.batch_size}")
    print(f"batch_count: {batches}")
    print(f"data_split_rows: {data_split_rows}")
    print(f"data_split_count: {data_split_count}")
    print(f"keops_import_seconds: {keops_import_elapsed:.9f}")
    print(f"data_read_seconds: {data_read_elapsed:.9f}")
    print(f"scale_coeff_seconds: {scale_coeff_elapsed:.9f}")
    print(f"query_read_seconds: {prep_times['query_read_seconds']:.9f}")
    print(f"query_construct_seconds: {prep_times['query_construct_seconds']:.9f}")
    print(f"data_scale_seconds: {prep_times['data_scale_seconds']:.9f}")
    print(f"query_scale_seconds: {prep_times['query_scale_seconds']:.9f}")
    print(f"weights_read_seconds: {prep_times['weights_read_seconds']:.9f}")
    print(f"pre_timer_total_seconds: {pre_timer_total:.9f}")
    print(f"qps: {qps:.9f}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
