"""Admission, failure and DAG regressions for bounded execution."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from kedro.io import DataCatalog, MemoryDataset
from kedro.pipeline import node, pipeline
from kedro.runner import AdaptiveThreadRunner, ResourceSnapshot, ThreadRunner

GIB = 1024**3
HEALTHY = ResourceSnapshot(8 * GIB, 16 * GIB, 10)


def identity(value):
    return value


@pytest.mark.parametrize("runner_type", [ThreadRunner, AdaptiveThreadRunner])
def test_diamond_shared_inputs_release_and_equivalence(runner_type):
    kwargs = (
        {"resource_probe": lambda: HEALTHY}
        if runner_type == AdaptiveThreadRunner
        else {}
    )
    runner = runner_type(max_workers=4, **kwargs)
    flow = pipeline(
        [
            node(identity, "input", "a", name="a"),
            node(identity, "a", "b", name="b"),
            node(identity, "a", "c", name="c"),
            node(lambda b, c: b + c, ["b", "c"], "result", name="join"),
        ]
    )
    catalog = DataCatalog({"input": MemoryDataset(21)})
    result = runner.run(flow, catalog)
    assert result["result"].load() == 42
    assert catalog.load("input") == 21
    for intermediate in ["a", "b", "c"]:
        assert not catalog.exists(intermediate)


def test_bounded_inflight_futures(mocker):
    active = peak = 0
    lock = threading.Lock()

    class CountingPool(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            future = super().submit(*args, **kwargs)

            def finished(_):
                nonlocal active
                with lock:
                    active -= 1

            future.add_done_callback(finished)
            return future

    mocker.patch("kedro.runner.thread_runner.ThreadPoolExecutor", CountingPool)

    def work():
        time.sleep(0.005)
        return 1

    flow = pipeline([node(work, None, f"out_{i}") for i in range(40)])
    result = ThreadRunner(max_workers=3).run(flow, DataCatalog())
    assert len(result) == 40
    assert peak <= 3


@pytest.mark.parametrize(
    "snapshot, expected",
    [
        (HEALTHY, 8),
        (ResourceSnapshot(4 * GIB, 16 * GIB, 10), 4),
        (ResourceSnapshot(GIB, 16 * GIB, 10), 1),
        (ResourceSnapshot(0, 16 * GIB, 10), 0),
        (ResourceSnapshot(), 1),
        (ResourceSnapshot(-1, 16 * GIB), 1),
        (ResourceSnapshot(8 * GIB, 16 * GIB, 99), 4),
    ],
)
def test_pressure_decisions(snapshot, expected):
    runner = AdaptiveThreadRunner(max_workers=8, resource_probe=lambda: snapshot)
    assert runner._get_submission_limit(8) == expected


def test_memory_budget_and_benchmark_ceiling():
    runner = AdaptiveThreadRunner(
        max_workers=16, benchmark_workers=6, memory_per_node_mb=2048
    )
    assert runner._max_workers == 6
    assert runner._decision(HEALTHY, 6)[0] == 3


def test_hysteresis_and_fast_reduction(mocker):
    snapshot = ResourceSnapshot(GIB, 16 * GIB, 10)
    clock = mocker.patch("kedro.runner.adaptive_runner.time.monotonic", return_value=0)
    runner = AdaptiveThreadRunner(max_workers=8, resource_probe=lambda: snapshot)
    assert runner._get_submission_limit(8) == 1
    snapshot = HEALTHY
    for timestamp, expected in [(1, 1), (2, 1), (3, 2)]:
        clock.return_value = timestamp
        assert runner._get_submission_limit(8) == expected
    snapshot = ResourceSnapshot(0, 16 * GIB)
    clock.return_value = 4
    assert runner._get_submission_limit(8) == 0


def test_critical_pause_recovers_without_reexecution():
    samples = iter([ResourceSnapshot(0, 16 * GIB), HEALTHY])
    runner = AdaptiveThreadRunner(
        max_workers=2,
        poll_interval=0.001,
        recovery_samples=1,
        resource_probe=lambda: next(samples, HEALTHY),
    )
    calls = []

    def work():
        calls.append(1)
        return 42

    result = runner.run(pipeline([node(work, None, "out")]), DataCatalog())
    assert result["out"].load() == 42
    assert calls == [1]
    assert [entry["limit"] for entry in runner.admission_history] == [0, 1]


def test_critical_timeout_does_not_execute_nodes():
    runner = AdaptiveThreadRunner(
        max_workers=2,
        poll_interval=0.001,
        pressure_timeout=0.003,
        resource_probe=lambda: ResourceSnapshot(0, 16 * GIB),
    )
    calls = []
    flow = pipeline([node(lambda: calls.append(1), None, "out")])
    with pytest.raises(TimeoutError, match="Critical memory"):
        runner.run(flow, DataCatalog())
    assert calls == []


def test_failure_stops_unsubmitted_work():
    calls = []

    def failure():
        calls.append("failed")
        raise ValueError("intentional")

    flow = pipeline(
        [
            node(failure, None, "a", name="a"),
            node(lambda: calls.append("later"), None, "b", name="b"),
        ]
    )
    with pytest.raises(ValueError, match="intentional"):
        ThreadRunner(max_workers=1).run(flow, DataCatalog())
    assert calls == ["failed"]


def test_reuse_resets_admission_history():
    runner = AdaptiveThreadRunner(max_workers=2, resource_probe=lambda: HEALTHY)
    flow = pipeline([node(lambda: 42, None, "out")])
    for _ in range(2):
        assert runner.run(flow, DataCatalog())["out"].load() == 42
        assert len(runner.admission_history) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"benchmark_workers": 0},
        {"memory_per_node_mb": 0},
        {"reserve_memory_mb": -1},
        {"poll_interval": float("nan")},
        {"pressure_timeout": 0},
        {"recovery_samples": 0},
        {"max_workers": 0},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        AdaptiveThreadRunner(**kwargs)
