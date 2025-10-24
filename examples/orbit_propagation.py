import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")

with app.setup:
    # Initialization code that runs before all other cells
    from datetime import datetime

    import matplotlib.pyplot as plt
    import numpy as np
    from pymap3d.ecef import ecef2geodetic

    from cosmica.dynamics import CircularSatelliteOrbitPropagator
    from cosmica.models import CircularSatelliteOrbitModel
    from cosmica.utils.coordinates import calc_dcm_eci2ecef
    from cosmica.visualization.equirectangular import draw_countries, draw_lat_lon_grid


@app.cell
def _():
    orbit = CircularSatelliteOrbitModel(
        semi_major_axis=7000e3,
        inclination=np.radians(98.7),
        # NumPy datetime64 does not support timezone info
        epoch=np.datetime64(
            datetime(2025, 1, 1, tzinfo=None),  # noqa: DTZ001
        ),
        phase_at_epoch=np.radians(20.0),
        raan=np.radians(0.0),
    )
    orbit
    return (orbit,)


@app.cell
def _():
    start_time = np.datetime64("2025-01-01T00:00:00")
    end_time = np.datetime64("2025-01-02T00:00:00")
    time_step = np.timedelta64(1, "m")

    time = np.arange(start_time, end_time, time_step)
    return (time,)


@app.cell
def _(orbit, time):
    propagator = CircularSatelliteOrbitPropagator(orbit)

    propagation_result = propagator.propagate(time)
    return (propagation_result,)


@app.cell
def _(propagation_result, time):
    dcm_eci2ecef = calc_dcm_eci2ecef(time)
    sat_pos_ecef = propagation_result.calc_position_ecef(dcm_eci2ecef)
    return (sat_pos_ecef,)


@app.cell
def _(sat_pos_ecef):
    lat, lon, _alt = ecef2geodetic(*sat_pos_ecef.T)
    return lat, lon


@app.cell
def _(lat, lon, time):
    _fig, _ax = plt.subplots(figsize=(15, 8))

    # _ax.set_title(f"Satellite {satellite}")
    draw_countries(ax=_ax)
    draw_lat_lon_grid(ax=_ax)
    # Color by time
    _h = _ax.scatter(lon, lat, label="Satellite", c=(time - time[0]) / np.timedelta64(1, "D"), cmap="viridis")

    # Colorbar
    _cbar = _fig.colorbar(_h, ax=_ax)

    _ax.legend()
    plt.show()


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
