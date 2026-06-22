# KeOps Exact KDE Baseline

This directory contains the local KeOps adapter used by Stage 0 baseline
discovery in `GPU-kernel-density-exact`.

The adapter evaluates the exact raw Gaussian sum:

```text
F(q) = sum_i w_i exp(-gamma ||q - p_i||^2)
```

or, in diagonal Scott mode:

```text
F(q) = sum_i w_i exp(-sum_j gamma_j (q_j - p_ij)^2)
```

It uses PyKeOps only as the exact kernel aggregation engine. It does not use
tree pruning, FFT/binning, approximate KDE, or normalized density output.

## Files

| File | Purpose |
| --- | --- |
| `keops_exact_kde.py` | Helper for one exact KeOps SVM or KDV workload. |
| `run_keops_local_3080ti.py` | Local RTX 3080 Ti Stage 0 runner. |
| `run_keops_remote_tbd.py` | Remote placeholder runner. |

A future remote machine should fill in `run_keops_remote_tbd.py` with only the
machine-specific paths and environment defaults.

## Kernel Scale

The runner matches the exact repo convention:

| Mode | CLI |
| --- | --- |
| Scalar gamma | `--gamma <gamma>` |
| Diagonal Scott | `--scott-diag <b>` |

Diagonal Scott uses:

```text
h_j = b sigma_j n^(-1 / (d + 4))
gamma_j = 1 / (2 h_j^2)
```

where `sigma_j` is the population standard deviation of data dimension `j`.
The implementation pre-scales data and query coordinates, then KeOps computes
`exp(-||q' - p_i'||^2)`.

## Local RTX 3080 Ti Suite

Run from this directory or from anywhere:

```bash
python3 /home/ubuntu/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/baselines/keops/scripts/run_keops_local_3080ti.py
```

Default paths:

| Item | Path |
| --- | --- |
| Dataset root | `/home/ubuntu/Documents/workspace/dataset/GPU-accelerated_Kernel_Density_Computation/` |
| Temporary result root | `/home/ubuntu/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/tmp-results/stage0/keops/` |
| Basic-Scan reference root | `/home/ubuntu/Documents/workspace/dataset/GPU-accelerated_Kernel_Density_Computation/exact/experiments/stage0/local-3080ti/basic/basic-scan-ground-truth/` |

Default scale mode is diagonal Scott with separate multipliers for the two
workload families:

| Workload type | Default |
| --- | --- |
| SVM | `--scott-diag 0.1` |
| KDV | `--scott-diag 1` |

The suite records machine inventory, logs, raw outputs, `summary.csv`, and
`correctness.csv` under a timestamped temporary run directory. The runtime
reported by the Python runner is query evaluation time only; file loading,
coordinate scaling, query construction, and result writing are outside the timed
region. The default warmup also keeps first-call PyKeOps JIT compilation outside
the timed region. Persistent promotion is manual after review.

## Five Workloads

| Case | Mode | Data |
| --- | --- | --- |
| `svm_susy` | SVM | `susy/SUSY_X.data`, `susy/SUSY_qSet.data` |
| `svm_home` | SVM | `home/HT_Sensor_dataset_X.data`, `home/HT_Sensor_dataset_qSet.data` |
| `svm_miniboone` | SVM | `miniboone/MiniBooNE_X.data`, `miniboone/MiniBooNE_qSet.data` |
| `kdv_home_visual` | KDV | `home_visualization/HT_Sensor_dataset_vis_X.data` |
| `kdv_susy_visual` | KDV | `susy_visualization/SUSY_vis_X.data` |

## Single Workload Examples

SVM:

```bash
python3 scripts/keops_exact_kde.py \
  svm \
  /home/ubuntu/Documents/workspace/dataset/GPU-accelerated_Kernel_Density_Computation/susy/SUSY_qSet.data \
  /home/ubuntu/Documents/workspace/dataset/GPU-accelerated_Kernel_Density_Computation/susy/SUSY_X.data \
  /tmp/susy_keops.out \
  --scott-diag 1 \
  --backend GPU \
  --dtype float64 \
  --local-keops-root /home/ubuntu/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/baselines/keops
```

KDV:

```bash
python3 scripts/keops_exact_kde.py \
  kdv \
  --rows 256 \
  --cols 256 \
  /home/ubuntu/Documents/workspace/dataset/GPU-accelerated_Kernel_Density_Computation/susy_visualization/SUSY_vis_X.data \
  /tmp/susy_vis_keops.out \
  --scott-diag 1 \
  --backend GPU \
  --dtype float64 \
  --local-keops-root /home/ubuntu/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/baselines/keops
```
