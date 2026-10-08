"""Opt-in bounded thread execution with live memory/CPU admission.

Nodes keep ownership of numerical/GPU acceleration. This runner only controls
admission; it never retries a node or changes its datasets or function.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kedro.runner.thread_runner import ThreadRunner

if TYPE_CHECKING:
    from collections.abc import Callable

_LOW_HEADROOM = 0.20
_MODERATE_HEADROOM = 0.30
_HIGH_CPU_PERCENT = 95


@dataclass(frozen=True)
class ResourceSnapshot:
    """Host headroom in bytes; None means telemetry is unavailable."""

    available: int | None = None
    total: int | None = None
    cpu_percent: float | None = None


def _sample_resources() -> ResourceSnapshot:
    try:
        import psutil  # noqa: PLC0415 - optional dependency, lazy telemetry

        memory = psutil.virtual_memory()
        return ResourceSnapshot(memory.available, memory.total, psutil.cpu_percent())
    except (ImportError, OSError, RuntimeError):
        return ResourceSnapshot()


def _recommended_workers() -> int:
    # Import once at the workload boundary, never in each Task/worker.
    try:
        from pyaccelerate.cpu import recommend_workers  # noqa: PLC0415

        return max(1, min(32, recommend_workers(io_bound=True)))
    except (ImportError, OSError, RuntimeError):
        return min(32, (os.cpu_count() or 1) + 4)


class AdaptiveThreadRunner(ThreadRunner):
    """ThreadRunner with a bounded, dynamically adjustable admission window.

    ``benchmark_workers`` is an optional ceiling measured on the caller's own
    workload. It is never inferred from Oracle timings or obtained by executing
    benchmarks automatically. Without telemetry, admission falls back to one.
    Critical pressure pauses new nodes, with a finite timeout when idle. Running
    nodes drain normally. Policies are per run, not a cross-run/global quota.
    """

    def __init__(  # noqa: PLR0913 - explicit resource policy knobs
        self,
        max_workers: int | None = None,
        is_async: bool = False,
        *,
        benchmark_workers: int | None = None,
        memory_per_node_mb: float = 192,
        reserve_memory_mb: float = 1024,
        poll_interval: float = 0.1,
        pressure_timeout: float = 30,
        recovery_samples: int = 3,
        resource_probe: Callable[[], ResourceSnapshot] | None = None,
    ):
        for name, value in (
            ("memory_per_node_mb", memory_per_node_mb),
            ("reserve_memory_mb", reserve_memory_mb),
            ("poll_interval", poll_interval),
            ("pressure_timeout", pressure_timeout),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(recovery_samples, int) or recovery_samples < 1:
            raise ValueError("recovery_samples must be a positive integer")
        if benchmark_workers is not None and (
            not isinstance(benchmark_workers, int) or benchmark_workers < 1
        ):
            raise ValueError("benchmark_workers must be a positive integer")
        ceiling = max_workers if max_workers is not None else _recommended_workers()
        if benchmark_workers is not None:
            ceiling = min(ceiling, benchmark_workers)
        super().__init__(max_workers=ceiling, is_async=is_async)
        self._memory_per_node = max(1, int(memory_per_node_mb * 1024**2))
        self._reserve = int(reserve_memory_mb * 1024**2)
        self._poll_interval = poll_interval
        self._pressure_timeout = pressure_timeout
        self._recovery_samples = recovery_samples
        self._probe = resource_probe or _sample_resources
        self._limit: int | None = None
        self._healthy_samples = 0
        self._next_sample = 0.0
        self._paused_since: float | None = None
        self.admission_history: list[dict[str, int | str]] = []

    def _run(self, *args, **kwargs) -> None:
        self._limit = None
        self._healthy_samples = 0
        self._next_sample = 0.0
        self._paused_since = None
        self.admission_history = []
        super()._run(*args, **kwargs)

    def _get_submission_limit(self, max_workers: int) -> int:
        now = time.monotonic()
        if self._limit is not None and now < self._next_sample:
            return min(max_workers, self._limit)
        self._next_sample = now + self._poll_interval
        sample = self._probe()
        desired, reason = self._decision(sample, max_workers)
        previous = self._limit
        if previous is None or desired <= previous:
            self._limit = desired
            self._healthy_samples = 0
        else:
            self._healthy_samples += 1
            if self._healthy_samples >= self._recovery_samples:
                self._limit = min(desired, previous + 1)
                self._healthy_samples = 0
        if self._limit == 0:
            if self._paused_since is None:
                self._paused_since = now
        else:
            self._paused_since = None
        if previous != self._limit:
            # Bound diagnostic storage, including repeated oscillation.
            self.admission_history.append({"limit": self._limit, "reason": reason})
            del self.admission_history[:-256]
            self._logger.info("Adaptive admission: limit=%s, %s", self._limit, reason)
        return self._limit

    def _decision(self, sample: ResourceSnapshot, ceiling: int) -> tuple[int, str]:
        if (
            sample.available is None
            or sample.total is None
            or sample.available < 0
            or sample.total <= 0
        ):
            return 1, "telemetry unavailable; serial admission"
        reserve = min(self._reserve, max(1, sample.total // 10))
        critical = min(256 * 1024**2, reserve // 4)
        if sample.available < critical:
            return 0, "critical memory; pause new submissions"
        ratio = sample.available / sample.total
        if sample.available < reserve or ratio < _LOW_HEADROOM:
            return 1, "low memory; serial progress"
        memory_limit = max(1, (sample.available - reserve) // self._memory_per_node)
        desired = min(ceiling, memory_limit)
        if ratio < _MODERATE_HEADROOM or (
            sample.cpu_percent is not None and sample.cpu_percent >= _HIGH_CPU_PERCENT
        ):
            return min(desired, max(1, (ceiling + 1) // 2)), "moderate pressure"
        return desired, "headroom available"

    def _wait_for_resources(self) -> None:
        if (
            self._paused_since is not None
            and time.monotonic() - self._paused_since >= self._pressure_timeout
        ):
            raise TimeoutError("Critical memory pressure prevented node admission")
        time.sleep(self._poll_interval)
