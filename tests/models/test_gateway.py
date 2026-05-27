from typing import cast

import numpy as np
import pytest

from cosmica.models import Gateway


def test_gateway_validation_rejects_invalid_latitude() -> None:
    with pytest.raises(AssertionError):
        Gateway(
            id=0,
            latitude=np.deg2rad(120.0),
            longitude=np.deg2rad(0.0),
            minimum_elevation=np.deg2rad(10.0),
        )


def test_gateway_validation_rejects_non_integer_terminals() -> None:
    with pytest.raises(AssertionError):
        Gateway(
            id=0,
            latitude=np.deg2rad(10.0),
            longitude=np.deg2rad(0.0),
            minimum_elevation=np.deg2rad(10.0),
            n_terminals=cast("int", 1.5),
        )
