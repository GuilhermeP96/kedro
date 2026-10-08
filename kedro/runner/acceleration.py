"""Optional pyaccelerate helpers for data and model nodes.

Dependencies are loaded at use time. GPU selection occurs before computation;
failed user work is never replayed on another backend.
"""

from __future__ import annotations

import logging
import multiprocessing
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Any, TypeVar

from kedro.runner.adaptive_runner import AdaptiveThreadRunner

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

T = TypeVar("T")
_LOGGER = logging.getLogger(__name__)
_MATRIX_DIMENSIONS = 2


def accelerated_map(  # noqa: PLR0913, PLR0912 - bounded generator and pool ownership
    function: Callable[..., T],
    items: Iterable[tuple],
    *,
    max_workers: int = 4,
    backend: str = "thread",
    use_pyaccelerate: bool = True,
    memory_per_task_mb: float = 192,
) -> Iterator[T]:
    """Yield ordered results with a live, bounded submission window.

    Use threads for I/O or native kernels releasing the GIL; processes for pure
    Python CPU work (top-level picklable function, guarded main on Windows).
    The thread backend borrows pyaccelerate's shared pool without closing it.
    Process pools are owned and use spawn. Input is consumed lazily, failures
    cancel pending work and drain running tasks without retrying. Returned
    results remain bounded unless the caller chooses to collect them.
    """
    if backend not in {"thread", "process"}:
        raise ValueError("backend must be 'thread' or 'process'")
    policy = AdaptiveThreadRunner(
        max_workers=max_workers, memory_per_node_mb=memory_per_task_mb
    )
    owned = True
    pool = None
    if use_pyaccelerate:
        try:
            from pyaccelerate.cpu import recommend_workers  # noqa: PLC0415

            max_workers = min(
                max_workers, recommend_workers(io_bound=backend == "thread")
            )
            if backend == "thread":
                from pyaccelerate.threads import get_pool  # noqa: PLC0415

                pool = get_pool()
                owned = False
            _LOGGER.info("pyaccelerate %s batches, ceiling=%d", backend, max_workers)
        except (ImportError, OSError, RuntimeError):
            _LOGGER.info("pyaccelerate unavailable; using stdlib %s", backend)
    if pool is None:
        pool = (
            ProcessPoolExecutor(
                max_workers=max_workers, mp_context=multiprocessing.get_context("spawn")
            )
            if backend == "process"
            else ThreadPoolExecutor(max_workers=max_workers)
        )
    pending = deque()
    source = iter(items)
    exhausted = False
    try:
        while not exhausted or pending:
            limit = policy._get_submission_limit(max_workers)
            while not exhausted and len(pending) < limit:
                try:
                    item = next(source)
                except StopIteration:
                    exhausted = True
                    break
                pending.append(pool.submit(function, *item))
            if pending:
                yield pending.popleft().result()
            elif not exhausted:
                policy._wait_for_resources()
    finally:
        for future in pending:
            future.cancel()
        if owned:
            pool.shutdown(wait=True, cancel_futures=True)
        else:
            wait(pending)


def array_backend(prefer_gpu: bool = False) -> tuple[Any, str]:
    """Select NumPy or pyaccelerate's CUDA module before running a kernel."""
    if prefer_gpu:
        try:
            from pyaccelerate.gpu import cuda  # noqa: PLC0415

            if cuda.is_available():
                return cuda.get_module(), "cuda"
        except ImportError:
            pass
    import numpy as np  # noqa: PLC0415 - optional numerical dependency

    return np, "cpu"


def standardize_batch(data, mean, scale, *, prefer_gpu: bool = False):
    """Apply training-only statistics in float64; return a host array and backend."""
    xp, backend = array_backend(prefer_gpu)
    values = xp.asarray(data, dtype=xp.float64)
    means = xp.asarray(mean, dtype=xp.float64)
    scales = xp.asarray(scale, dtype=xp.float64)
    if (
        values.ndim != _MATRIX_DIMENSIONS
        or means.shape != values.shape[1:]
        or scales.shape != means.shape
    ):
        raise ValueError("Data must be 2D with matching 1D mean/scale")
    if not bool(
        xp.isfinite(values).all() & xp.isfinite(means).all() & xp.isfinite(scales).all()
    ):
        raise ValueError("Non-finite data or statistics")
    result = (values - means) / xp.where(scales == 0, 1.0, scales)
    return (xp.asnumpy(result) if backend == "cuda" else result), backend


def fit_ridge(data, target, alpha: float, *, prefer_gpu: bool = False):
    """Train ridge regression with an unpenalized intercept, CPU/CUDA float64.

    This exact dense solver is suited to moderate feature counts; it is not a
    replacement for sparse/NLP solvers. Backend errors propagate without retry.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    xp, backend = array_backend(prefer_gpu)
    x = xp.asarray(data, dtype=xp.float64)
    y = xp.asarray(target, dtype=xp.float64)
    if (
        x.ndim != _MATRIX_DIMENSIONS
        or y.ndim != 1
        or x.shape[0] != y.shape[0]
        or not x.shape[0]
    ):
        raise ValueError("Expected non-empty 2D features and matching 1D target")
    if not bool(xp.isfinite(x).all() & xp.isfinite(y).all() & xp.isfinite(alpha)):
        raise ValueError("Training inputs must be finite")
    x_mean, y_mean = x.mean(axis=0), y.mean()
    centered, response = x - x_mean, y - y_mean
    gram = centered.T @ centered
    gram += alpha * xp.eye(x.shape[1], dtype=xp.float64)
    coefficients = xp.linalg.solve(gram, centered.T @ response)
    intercept = y_mean - x_mean @ coefficients
    if backend == "cuda":
        coefficients, intercept = xp.asnumpy(coefficients), float(intercept.item())
    return coefficients, float(intercept), backend
