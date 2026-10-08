"""Numerical equivalence and batching contracts."""

import numpy as np
import pytest

from kedro.runner.acceleration import (
    accelerated_map,
    fit_ridge,
    standardize_batch,
)


def square(value):
    return value * value


@pytest.mark.parametrize("use_pyaccelerate", [True, False])
def test_ordered_lazy_batches(use_pyaccelerate):
    if use_pyaccelerate:
        pytest.importorskip("pyaccelerate")
    result = accelerated_map(
        square,
        ((i,) for i in range(15)),
        max_workers=3,
        use_pyaccelerate=use_pyaccelerate,
    )
    assert list(result) == [i * i for i in range(15)]


def test_process_spawn():
    assert list(
        accelerated_map(square, [(2,), (3,)], backend="process", max_workers=2)
    ) == [4, 9]


def test_kernel_failure_not_retried():
    calls = []

    def fail(value):
        calls.append(value)
        raise ValueError("kernel failure")

    with pytest.raises(ValueError, match="kernel failure"):
        list(accelerated_map(fail, [(1,), (2,)], max_workers=1))
    assert calls == [1]


def test_standardization_matches_reference_and_preserves_input():
    x = np.array([[1, 2, 7], [3, 6, 7]], dtype=np.float64)
    original = x.copy()
    mean, scale = x.mean(axis=0), x.std(axis=0)
    actual, backend = standardize_batch(x, mean, scale)
    np.testing.assert_allclose(actual, [[-1, -1, 0], [1, 1, 0]], rtol=1e-12)
    np.testing.assert_array_equal(x, original)
    assert backend == "cpu"


def test_gpu_unavailable_falls_back_before_computation(mocker):
    pytest.importorskip("pyaccelerate")
    mocker.patch("pyaccelerate.gpu.cuda.is_available", return_value=False)
    _, backend = standardize_batch([[1, 2]], [0, 0], [1, 1], prefer_gpu=True)
    assert backend == "cpu"


def test_ridge_matches_independent_augmented_least_squares():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(200, 8))
    y = x @ rng.normal(size=8) + 5 + rng.normal(size=200) * 0.01
    alpha = 0.2
    augmented_x = np.column_stack([x, np.ones(len(x))])
    penalty = np.diag([np.sqrt(alpha)] * 8 + [0])
    reference = np.linalg.lstsq(
        np.vstack([augmented_x, penalty]), np.concatenate([y, np.zeros(9)]), rcond=None
    )[0]
    weights, intercept, backend = fit_ridge(x, y, alpha)
    np.testing.assert_allclose(weights, reference[:-1], rtol=1e-10, atol=1e-10)
    assert intercept == pytest.approx(reference[-1], rel=1e-10)
    assert backend == "cpu"


@pytest.mark.parametrize("alpha", [0, -1, float("nan"), float("inf")])
def test_bad_regularization(alpha):
    with pytest.raises(ValueError):
        fit_ridge([[1]], [1], alpha)


def test_nonfinite_data_fails():
    with pytest.raises(ValueError):
        standardize_batch([[float("nan")]], [0], [1])
