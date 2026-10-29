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
    """Preserve zero vectors, floating precision, and inputs across supported axes."""
    vector = np.array(values, dtype=dtype)
    original = vector.copy()

    with np.errstate(divide="raise", invalid="raise"):
        result = normalize(vector, axis=axis)

    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(vector, original)
    assert np.issubdtype(result.dtype, np.floating)
    if np.issubdtype(vector.dtype, np.floating):
        assert result.dtype == vector.dtype


@pytest.mark.parametrize("order", [-1.0, -2.0])
@pytest.mark.parametrize("values", [[0.0, 0.0, 0.0], [0.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
def test_normalize_negative_norms(order: float, values: list[float]):
    """Negative norms of vectors containing zero are zero, without division warnings."""
    vector = np.array(values)
    expected = vector if 0.0 in values else vector / np.linalg.norm(vector, ord=order)

    with np.errstate(divide="raise", invalid="raise"):
        result = normalize(vector, ord=order)

    np.testing.assert_allclose(result, expected)
