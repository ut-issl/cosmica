import pytest

from package_name_goes_here import square


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (2, 4),
        (3, 9),
        (4, 16),
        (5, 25),
    ],
)
def test_square(x: int, expected: int) -> None:
    assert square(x) == expected
