import warnings

import numpy as np
import pytest

from cosmica.utils.vector import normalize


@pytest.mark.parametrize(
    ("vector", "norm_order", "axis", "expected"),
    [
        pytest.param(
            np.zeros(3, dtype=np.int64),
            2,
            None,
            np.zeros(3),
            id="integer-zero-vector",
        ),
        pytest.param(
            np.array([[3.0, 4.0], [0.0, 0.0]]),
            2,
            1,
            np.array([[0.6, 0.8], [0.0, 0.0]]),
            id="mixed-zero-and-nonzero-rows",
        ),
        pytest.param(
            np.array([0.0, 2.0]),
            -np.inf,
            None,
            np.array([0.0, 2.0]),
            id="negative-infinity-order-zero-norm",
        ),
    ],
)
def test_normalize_handles_zero_norm_without_warning(
    vector: np.ndarray,
    norm_order: float,
    axis: int | None,
    expected: np.ndarray,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = normalize(vector, ord=norm_order, axis=axis)

    np.testing.assert_allclose(result, expected)
    assert np.issubdtype(result.dtype, np.floating)
