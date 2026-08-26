import numpy as np
import pytest

from cosmica.models import OpticalCommunicationTerminal


def test_optical_terminal_preserves_body_to_terminal_mounting_transform() -> None:
    dcm_body2terminal = (
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    terminal = OpticalCommunicationTerminal(
        id=1,
        azimuth_min=-np.pi,
        azimuth_max=np.pi,
        elevation_min=-np.pi / 2,
        elevation_max=np.pi / 2,
        angular_velocity_max=1.0,
        dcm_body2terminal=dcm_body2terminal,
    )

    assert terminal.dcm_body2terminal == dcm_body2terminal


def test_optical_terminal_rejects_invalid_mounting_transform() -> None:
    with pytest.raises(ValueError, match="orthonormal, right-handed"):
        OpticalCommunicationTerminal(
            id=1,
            azimuth_min=-np.pi,
            azimuth_max=np.pi,
            elevation_min=-np.pi / 2,
            elevation_max=np.pi / 2,
            angular_velocity_max=1.0,
            dcm_body2terminal=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, -1.0),
            ),
        )


def test_optical_terminal_rejects_wrapping_azimuth_interval() -> None:
    with pytest.raises(ValueError, match="non-wrapping"):
        OpticalCommunicationTerminal(
            id=1,
            azimuth_min=np.deg2rad(170.0),
            azimuth_max=np.deg2rad(-170.0),
            elevation_min=-np.pi / 2,
            elevation_max=np.pi / 2,
            angular_velocity_max=1.0,
        )
