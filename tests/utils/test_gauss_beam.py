"""Tests for Gaussian-beam propagation and aperture-coupling invariants (issue #195)."""

import numpy as np
import pytest

from cosmica.utils.gauss_beam import calc_gauss_beam_radius, calc_gauss_rx_power

WVL = 1550e-9
W0 = 0.04
Z_R = np.pi * W0**2 / WVL


def test_beam_radius_at_waist() -> None:
    assert calc_gauss_beam_radius(W0, WVL, 0.0) == pytest.approx(W0)


def test_beam_radius_at_rayleigh_range() -> None:
    assert calc_gauss_beam_radius(W0, WVL, Z_R) == pytest.approx(W0 * np.sqrt(2))


def test_beam_radius_grows_linearly_in_far_field() -> None:
    far = 1000 * Z_R
    assert calc_gauss_beam_radius(W0, WVL, 2 * far) == pytest.approx(2 * calc_gauss_beam_radius(W0, WVL, far), rel=1e-3)


@pytest.mark.parametrize("distance", [1e3, 1e6, 4e7])
def test_rx_power_within_transmitted_power(distance: float) -> None:
    power = calc_gauss_rx_power(power_tx=1.0, aperture=0.1, distance=distance)
    assert 0.0 <= power <= 1.0


def test_rx_power_increases_with_aperture() -> None:
    small = calc_gauss_rx_power(power_tx=1.0, aperture=0.05, distance=1e6)
    large = calc_gauss_rx_power(power_tx=1.0, aperture=0.5, distance=1e6)
    assert small < large


def test_rx_power_decreases_with_distance() -> None:
    near = calc_gauss_rx_power(power_tx=1.0, aperture=0.1, distance=1e6)
    far = calc_gauss_rx_power(power_tx=1.0, aperture=0.1, distance=1e7)
    assert far < near
