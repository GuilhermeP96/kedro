"""Synthetic offline A/B benchmark; no telemetry upload or business datasets.

Run from the checkout: python tools/benchmark_adaptive.py --output <json path>
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import subprocess
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from kedro.io import DataCatalog, MemoryDataset
from kedro.pipeline import node, pipeline
from kedro.runner import AdaptiveThreadRunner, ResourceSnapshot, ThreadRunner
from kedro.runner.acceleration import accelerated_map, fit_ridge, standardize_batch

UPSTREAM = "2a3af98eacf4fe28b5857ef15adc1f4759d0cb34"
EXPECTED_VALUE = 42


class CountingPool(ThreadPoolExecutor):
    def __init__(self, max_workers):
        super().__init__(max_workers=max_workers)
        self.inflight = self.peak = 0
        self.guard = threading.Lock()

    def submit(self, *args, **kwargs):
        with self.guard:
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
        future = super().submit(*args, **kwargs)

        def finished(_):
            with self.guard:
                self.inflight -= 1

        future.add_done_callback(finished)
        return future


class MeasuredRunner(ThreadRunner):
    def _get_executor(self, max_workers):
        self.pool = CountingPool(max_workers)
        return self.pool


def identity(value):
    return value


def measure(function, repeats=3):
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        timings.append(time.perf_counter() - start)
    return {"seconds": timings, "median_seconds": statistics.median(timings)}, result


def main():  # noqa: PLR0915 - offline experiment phases
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    original = types.ModuleType("kedro.runner.original_benchmark")
    original.__dict__["__name__"] = "kedro.runner.original_benchmark"
    source = subprocess.check_output(  # noqa: S603 - fixed upstream revision
        ["git", "show", f"{UPSTREAM}:kedro/runner/runner.py"],  # noqa: S607
        text=True,
    )
    exec(compile(source, "upstream_runner.py", "exec"), original.__dict__)  # noqa: S102 - fixed upstream baseline

    class OriginalRunner(MeasuredRunner):
        def _run(self, *args, **kwargs):
            return original.AbstractRunner._run(self, *args, **kwargs)

        _release_datasets = staticmethod(original.AbstractRunner._release_datasets)

    report = {
        "upstream": UPSTREAM,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "repeats": 3,
        "benchmarks": {},
    }
    flows = {
        "chain_600": pipeline(
            [node(identity, f"v{i}", f"v{i + 1}", name=f"n{i}") for i in range(600)]
        ),
        "wide_600": pipeline(
            [node(identity, "v0", f"out{i}", name=f"n{i}") for i in range(600)]
        ),
    }
    for name, flow in flows.items():
        values = {}
        for label, runner_cls in [
            ("upstream", OriginalRunner),
            ("bounded", MeasuredRunner),
        ]:
            peaks = []

            def run():
                runner = runner_cls(max_workers=4)
                result = runner.run(flow, DataCatalog({"v0": MemoryDataset(42)}))
                assert all(
                    dataset.load() == EXPECTED_VALUE for dataset in result.values()
                )
                peaks.append(runner.pool.peak)
                return len(result)

            values[label], count = measure(run)
            values[label]["peak_inflight"] = max(peaks)
            values[label]["outputs"] = count
        values["speedup"] = (
            values["upstream"]["median_seconds"] / values["bounded"]["median_seconds"]
        )
        report["benchmarks"][name] = values

    rng = np.random.default_rng(42)
    data = rng.normal(size=(40000, 32))
    mean, scale = data.mean(axis=0), data.std(axis=0)

    def row_loop():
        return np.array(
            [
                [(value - mean[j]) / scale[j] for j, value in enumerate(row)]
                for row in data
            ]
        )

    baseline, expected = measure(row_loop)
    vectorized, actual = measure(lambda: standardize_batch(data, mean, scale)[0])
    batched, batches = measure(
        lambda: list(
            accelerated_map(
                standardize_batch,
                ((part, mean, scale) for part in np.array_split(data, 8)),
                max_workers=4,
            )
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.concatenate([part[0] for part in batches]), expected, rtol=1e-12, atol=1e-12
    )
    report["benchmarks"]["standardize_40000x32"] = {
        "python_loop": baseline,
        "numpy": vectorized,
        "pyaccelerate_batches": batched,
        "equivalent": True,
        "backend": "cpu",
        "speedup_vs_loop": baseline["median_seconds"] / batched["median_seconds"],
    }

    x = rng.normal(size=(16000, 128))
    y = x @ rng.normal(size=128) + rng.normal(size=len(x)) * 0.1
    alphas = np.geomspace(0.01, 100, 12)
    training_x, validation_x = x[:12000], x[12000:]
    training_y, validation_y = y[:12000], y[12000:]
    # Set native BLAS budget once around the experiment, not from worker threads.
    with threadpool_limits(limits=1):
        serial, models = measure(
            lambda: [fit_ridge(training_x, training_y, float(a)) for a in alphas]
        )
        parallel, candidates = measure(
            lambda: list(
                accelerated_map(
                    fit_ridge,
                    ((training_x, training_y, float(a)) for a in alphas),
                    max_workers=4,
                )
            )
        )
    for reference, candidate in zip(models, candidates):
        np.testing.assert_allclose(reference[0], candidate[0], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(reference[1], candidate[1], rtol=1e-10, atol=1e-10)
    scores = [
        float(np.mean((validation_x @ weights + intercept - validation_y) ** 2))
        for weights, intercept, _ in candidates
    ]
    report["benchmarks"]["ridge_12_candidates_12000x128"] = {
        "serial": serial,
        "pyaccelerate_batches": parallel,
        "equivalent": True,
        "backend": "cpu",
        "validation_rows": 4000,
        "best_validation_mse": min(scores),
        "speedup": serial["median_seconds"] / parallel["median_seconds"],
    }
    samples = iter(
        [
            ResourceSnapshot(0, 16 * 1024**3),
            ResourceSnapshot(8 * 1024**3, 16 * 1024**3, 0),
        ]
    )
    healthy = ResourceSnapshot(8 * 1024**3, 16 * 1024**3, 0)
    adaptive = AdaptiveThreadRunner(
        max_workers=4,
        resource_probe=lambda: next(samples, healthy),
        recovery_samples=1,
        poll_interval=0.001,
    )
    adaptive.run(flows["wide_600"], DataCatalog({"v0": MemoryDataset(42)}))
    report["pressure_simulation"] = adaptive.admission_history
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
