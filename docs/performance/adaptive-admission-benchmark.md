# Bounded DAG admission and optional acceleration: local benchmark

Measured on 2026-10-08 against upstream commit
`2a3af98eacf4fe28b5857ef15adc1f4759d0cb34`. This proposal replaces repeated
pending-node scans with dependency counters and a ready queue, caches external
dataset boundaries once per run, and bounds in-flight futures by worker count.
It preserves dataset release and does not retry failed work.

`AdaptiveThreadRunner` is opt-in: live memory/CPU headroom can reduce admission;
recovery is gradual, critical memory pauses admission with a timeout, and
unavailable telemetry admits one task. `benchmark_workers` is a ceiling supplied
by callers from their own workload measurements. Policies are per execution,
not a global quota. Existing running tasks drain normally.

Optional `accelerated_map`, `standardize_batch`, and `fit_ridge` helpers use
lazy dependencies. Threads suit I/O/native kernels; owned spawn processes suit
pure Python. Borrowed pyaccelerate pools are not closed by the caller. GPU
selection precedes computation and failed kernels are not replayed on CPU.
Dense Ridge is for moderate feature counts, not a sparse NLP solver replacement.

## Reference hardware and runtime

Hardware was inventoried after the benchmark on the same date:

- Virtualized Windows 11 Enterprise x64, version 10.0.26100; hypervisor reported.
- Guest CPU identification: Intel Xeon Gold 6348 @ 2.60 GHz, **4 exposed vCPUs**.
  This does not mean the guest owns all physical cores of that processor.
- Guest RAM: 17,178,677,248 bytes, approximately **16 GiB**.
- Guest disk reports SSD/SAS, 200 GiB; physical backing storage and host
  contention were not measured. Disk is not the main target of this benchmark.
- Python 3.13.14, NumPy 2.5.3, scikit-learn 1.8.0,
  threadpoolctl 3.7, psutil 7.2.2, local pyaccelerate 0.11.2.
- No usable CUDA runtime; GPU acceleration has not been physically validated.

Other workloads were active. The synthetic A/B order was fixed. This is a
reference measurement, not a controlled capacity/SLA certification.

## Results

Seed 42, three timings per scenario, medians. Original runner is loaded from
the pinned upstream revision, including its original dataset release code.

| Scenario | Original/serial | Candidate | Ratio |
| --- | ---: | ---: | ---: |
| Chain of 600 nodes, 4 workers | 3.566 s | 2.069 s | 1.72x |
| 600 independent nodes, 4 workers | 3.475 s | 2.263 s | 1.54x |
| Normalize 40,000 x 32: Python loop vs batches | 1.262 s | 0.00712 s | 177.20x |
| Normalize: NumPy vs pyaccelerate batches | 0.01383 s | 0.00712 s | 1.94x |
| 12 independent Ridge fits, training 12,000 x 128 | 0.2157 s | 0.1369 s | 1.58x |

The wide DAG's peak futures fell from 600 to 4. This is a queue-size
measurement, not an equivalent RSS reduction. The 177x result combines
vectorization with batching; it must not be attributed solely to pyaccelerate.
The first batch invocation took 0.242 s including initialization, so the
steady-state median does not describe cold-start performance.

DAG outputs were checked; normalization uses allclose at 1e-12 and Ridge at
1e-10. Ridge validation has 4,000 synthetic rows and minimum MSE 0.010316;
it demonstrates numerical equivalence, not business/model quality.
Pressure recovery was simulated (0 -> 1 -> 2 -> 3 -> 4), without exhausting
the host's actual memory.

## Tests and reproduction

Original development validation: 518 serial/thread/pipeline/helper tests passed;
66 parallel-runner tests passed with 3 skips. These suites overlap other targeted
checks and their counts must not be combined. Full Kedro suite was not run.
The clean publication worktree was validated again: 518 targeted tests passed
in 9.73 s (73 warnings), covering helpers/admission, serial/thread runners and
pipeline behavior. Ruff passed on all changed Python files.

```powershell
python -m pip install -e '.[acceleration-benchmark]' pytest pytest-mock pytest-cov
python tools/benchmark_adaptive.py --output benchmark.json
python -m pytest tests/runner tests/pipeline -q -o addopts=''
```

The measured optional dependency came from a local pyaccelerate 0.11.2 checkout;
published package versions may differ. Pin and report the actual dependency
revision when reproducing, and avoid interpreting initialization as throughput.

## Cloud interpretation

More CPU/RAM or a suitable GPU can improve absolute time when the workload can
use them. **A larger relative speedup is a hypothesis, not a measured result.**
Both baseline and candidate may improve; memory bandwidth, VM contention,
executor overhead and service latency can limit scaling. Burstable CPU credits
can also distort sustained benchmarks. No cloud run was performed.

Repeat both versions on the same instance with fixed input/seed, worker/BLAS
budgets and comparable load; report instance type, vCPUs, RAM, accelerator,
storage, runtime versions, warm/cold timings and correctness. Add peak RSS and
CPU utilization to assess capacity before increasing concurrency.
