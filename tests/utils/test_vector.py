import numpy as np
import numpy.typing as npt
import pytest

from cosmica.utils.vector import normalize


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
@pytest.mark.parametrize(
    ("values", "axis", "expected"),
    [
        ([0, 0, 0], None, [0, 0, 0]),
        ([3, 4, 0], None, [0.6, 0.8, 0]),
        ([[0, 0], [3, 4]], -1, [[0, 0], [0.6, 0.8]]),
        ([[0, 3], [0, 4]], 0, [[0, 0.6], [0, 0.8]]),
    ],
)
def test_normalize(
    dtype: npt.DTypeLike,
    values: list,
    axis: int | None,
    expected: list,
):
    vector = np.array(values, dtype=dtype)
    original = vector.copy()

    with np.errstate(divide="raise", invalid="raise"):
        result = normalize(vector, axis=axis)

    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(vector, original)
    assert np.issubdtype(result.dtype, np.floating)
    if np.issubdtype(vector.dtype, np.floating):
        assert result.dtype == vector.dtype
