from datetime import datetime, timedelta, timezone

import h5py
import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from firebench import svn
from firebench.standardize import TIME_SERIES
from netCDF4 import Dataset

from firebench_adapter_wrf_sfire import get_weather_stations
from firebench_adapter_wrf_sfire.cli import main
from firebench_adapter_wrf_sfire.wx import _interp_linear_1d_strict


@pytest.mark.parametrize(
    "fmc_options, expected",
    [
        pytest.param({}, [0.2, 0.3, 0.4, 0.5, 0.6], id="default-layer-1"),
        pytest.param({"i_fmc_10h": 0}, [0.1, 0.15, 0.2, 0.25, 0.3], id="custom-layer-0"),
        pytest.param({"i_fmc_10h": None}, [0.1, 0.15, 0.2, 0.25, 0.3], id="3d-field"),
    ],
)
@pytest.mark.parametrize("entrypoint", ["api", "cli"])
def test_weather_station_fmc_layouts(tmp_path, fmc_options, expected, entrypoint):
    wrf_path = tmp_path / "wrfout.nc"
    obs_path = tmp_path / "obs.h5"
    out_path = tmp_path / "model.h5"
    station_path = f"{TIME_SERIES}/station_1"
    fmc_name = svn.FUEL_MOISTURE_CONTENT_10H.value
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    obs_time = np.array([0.0, 30.0, 60.0, 90.0, 120.0])

    with Dataset(wrf_path, "w") as wrf:
        wrf.createDimension("Time", 2)
        wrf.createDimension("south_north", 4)
        wrf.createDimension("west_east", 4)
        dims = ("Time", "south_north", "west_east")
        lat = np.broadcast_to(np.arange(38.0, 42.0)[:, None], (4, 4))
        lon = np.broadcast_to(np.arange(-122.0, -118.0)[None, :], (4, 4))
        for name, values in (("XLAT", lat), ("XLONG", lon), ("FXLAT", lat), ("FXLONG", lon)):
            wrf.createVariable(name, "f8", dims)[:] = np.broadcast_to(values, (2, 4, 4))
        wrf.createVariable("XTIME", "f8", ("Time",))[:] = [0.0, 2.0]

        if fmc_options.get("i_fmc_10h", 1) is None:
            wrf.createVariable("FMC_GC", "f8", dims)[:] = np.broadcast_to(
                np.array([0.1, 0.3])[:, None, None], (2, 4, 4)
            )
        else:
            wrf.createDimension("fuel_layer", 2)
            fmc_dims = ("Time", "fuel_layer", "south_north", "west_east")
            wrf.createVariable("FMC_GC", "f8", fmc_dims)[:] = np.broadcast_to(
                np.array([[0.1, 0.2], [0.3, 0.6]])[:, :, None, None], (2, 2, 4, 4)
            )

    with h5py.File(obs_path, "w") as obs_h5:
        station = obs_h5.create_group(station_path)
        station.attrs["position_lat"] = 39.0
        station.attrs["position_lon"] = -121.0
        time = station.create_dataset("time", data=obs_time)
        time.attrs["time_origin"] = origin.isoformat()
        time.attrs["time_units"] = "s"
        station.create_dataset(fmc_name, data=np.zeros_like(obs_time))

    if entrypoint == "api":
        with (
            h5py.File(out_path, "w") as new_h5,
            h5py.File(obs_path, "r") as obs_h5,
            Dataset(wrf_path, "r") as wrf,
        ):
            get_weather_stations(
                new_h5,
                obs_h5,
                wrf,
                origin.isoformat(),
                origin,
                origin + timedelta(seconds=120),
                fire_grid_bound=1,
                compression_lvl=1,
                **fmc_options,
            )
    else:
        config_path = tmp_path / "config.yml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "main": {
                        "obs_path": str(obs_path),
                        "wrf_path": str(wrf_path),
                        "output_path": str(out_path),
                        "authors": "Test Author",
                        "sfire_ref_time": origin.isoformat(),
                        "start_dt": origin.isoformat(),
                        "end_dt": (origin + timedelta(seconds=120)).isoformat(),
                        "fire_grid_bound": 1,
                        "compression_lvl": 1,
                    },
                    "weather": {"enabled": True, **fmc_options},
                }
            ),
            encoding="utf-8",
        )
        result = CliRunner().invoke(main, ["run", str(config_path)])
        assert result.exit_code == 0, result.output

    with h5py.File(out_path, "r") as out_h5:
        result = out_h5[f"{station_path}/{fmc_name}"]
        np.testing.assert_allclose(result[:], expected)
        assert result.attrs["units"] == "dimensionless"
        np.testing.assert_array_equal(out_h5[f"{station_path}/time"][:], obs_time)


def test_one_model_time_gives_its_float32_value():
    """WRF writes float32: one output time at the one observation time is that value, not NaN."""
    result = _interp_linear_1d_strict(
        np.array([3600.0]), np.array([290.5], dtype=np.float32), np.array([3600.0])
    )
    np.testing.assert_array_equal(result, [290.5])


@pytest.mark.parametrize("fire_grid_bound", [0, 1, 2])
@pytest.mark.parametrize(
    "variable, fmc_layer",
    [
        (svn.AIR_TEMPERATURE.value, 1),
        (svn.RELATIVE_HUMIDITY.value, 1),
        (svn.WIND_DIRECTION.value, 1),
        (svn.WIND_SPEED.value, 1),
        (svn.FUEL_MOISTURE_CONTENT_10H.value, None),
        (svn.FUEL_MOISTURE_CONTENT_10H.value, 0),
        (svn.FUEL_MOISTURE_CONTENT_10H.value, 1),
    ],
)
def test_weather_station_reads_full_grid_cell_after_trimming(
    tmp_path, fire_grid_bound, variable, fmc_layer
):
    """Each variable independently reads the station's cell, including fire-grid roughness."""
    wrf_path = tmp_path / "wrfout.nc"
    obs_path = tmp_path / "obs.h5"
    out_path = tmp_path / "model.h5"
    station_path = f"{TIME_SERIES}/station_1"
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    obs_time = np.array([0.0, 60.0, 120.0])
    cells = 10.0 * np.arange(6)[:, None] + np.arange(6)[None, :]
    fire_cells = 10.0 * np.arange(10)[:, None] + np.arange(10)[None, :]
    angle = np.deg2rad(cells)

    with Dataset(wrf_path, "w") as wrf:
        for name, length in (
            ("Time", 2),
            ("south_north", 6),
            ("west_east", 6),
            ("south_north_subgrid", 10),
            ("west_east_subgrid", 10),
        ):
            wrf.createDimension(name, length)
        dims = ("Time", "south_north", "west_east")
        fields = {
            "XLAT": np.broadcast_to(38.0 + 0.5 * np.arange(6)[:, None], (6, 6)),
            "XLONG": np.broadcast_to(-122.0 + 0.5 * np.arange(6)[None, :], (6, 6)),
            "T2": 280.0 + 0.1 * cells + np.array([0.0, 2.0])[:, None, None],
            "Q2": 0.002 + 0.00001 * cells,
            "PSFC": 90000.0 + 10.0 * cells,
            "U10": 1.0 + 0.1 * cells,
            "V10": 2.0 + 0.03 * cells,
            "COSALPHA": np.cos(angle),
            "SINALPHA": np.sin(angle),
        }
        for name, values in fields.items():
            wrf.createVariable(name, "f8", dims)[:] = np.broadcast_to(values, (2, 6, 6))
        fire_dims = ("Time", "south_north_subgrid", "west_east_subgrid")
        fire_fields = {
            "FXLAT": np.broadcast_to(37.0 + 0.5 * np.arange(10)[:, None], (10, 10)),
            "FXLONG": np.broadcast_to(-123.0 + 0.5 * np.arange(10)[None, :], (10, 10)),
            "WZ0": 0.01 + 0.001 * fire_cells,
        }
        for name, values in fire_fields.items():
            wrf.createVariable(name, "f8", fire_dims)[:] = np.broadcast_to(values, (2, 10, 10))
        wrf.createVariable("XTIME", "f8", ("Time",))[:] = [0.0, 2.0]
        fmc = 0.1 + 0.001 * cells + np.array([0.0, 0.02])[:, None, None]
        if fmc_layer is None:
            wrf.createVariable("FMC_GC", "f8", dims)[:] = fmc
        else:
            wrf.createDimension("fuel_layer", 2)
            wrf.createVariable("FMC_GC", "f8", ("Time", "fuel_layer", *dims[1:]))[:] = np.stack(
                [fmc, fmc + 0.1], axis=1
            )

    with h5py.File(obs_path, "w") as obs:
        station = obs.create_group(station_path)
        station.attrs["position_lat"], station.attrs["position_lon"] = 39.0, -121.0
        time = station.create_dataset("time", data=obs_time)
        time.attrs["time_origin"], time.attrs["time_units"] = origin.isoformat(), "s"
        dataset = station.create_dataset(variable, data=np.zeros(3))
        dataset.attrs["sensor_height"] = 6.1 if variable == svn.WIND_SPEED.value else 2.0
        dataset.attrs["sensor_height_units"] = "m"

    with (
        Dataset(wrf_path) as wrf,
        h5py.File(obs_path) as obs,
        h5py.File(out_path, "w") as output,
    ):
        get_weather_stations(
            output,
            obs,
            wrf,
            origin.isoformat(),
            origin,
            origin + timedelta(seconds=120),
            fire_grid_bound=fire_grid_bound,
            i_fmc_10h=fmc_layer,
        )
        # The station is at atmospheric cell (2, 2) and fire cell (4, 4), before trimming.
        temperature = np.array([282.2, 284.2])
        vapor_pressure = 0.00222 * 90220.0 / (0.622 + 0.00222)
        saturation_pressure = 611.2 * np.exp(17.67 * (temperature - 273.15) / (temperature - 29.65))
        direction = (np.degrees(np.arctan2(-3.2, -2.66)) - 22.0) % 360.0
        expected = {
            svn.AIR_TEMPERATURE.value: [282.2, 283.2, 284.2],
            svn.RELATIVE_HUMIDITY.value: np.interp(
                obs_time, [0.0, 120.0], 100.0 * vapor_pressure / saturation_pressure
            ),
            svn.WIND_DIRECTION.value: direction,
            svn.WIND_SPEED.value: np.hypot(3.2, 2.66) * np.log(6.1 / 0.054) / np.log(10.0 / 0.054),
            svn.FUEL_MOISTURE_CONTENT_10H.value: np.array([0.122, 0.132, 0.142])
            + (0.1 if fmc_layer == 1 else 0.0),
        }
        np.testing.assert_allclose(output[f"{station_path}/{variable}"][:], expected[variable])
        np.testing.assert_array_equal(output[f"{station_path}/time"][:], obs_time)


@pytest.mark.parametrize(
    "directions, speeds, expected",
    [
        pytest.param([350.0, 10.0], [5.0, 5.0], [350.0, 0.0, 10.0], id="north-crossing"),
        pytest.param([10.0, 350.0], [5.0, 5.0], [10.0, 0.0, 350.0], id="reverse-crossing"),
        pytest.param([350.0, 10.0], [2.0, 6.0], [350.0, 5.0383687733, 10.0], id="unequal-speeds"),
        pytest.param([90.0, 180.0], [5.0, 5.0], [90.0, 135.0, 180.0], id="ordinary-turn"),
    ],
)
@pytest.mark.parametrize("rotation", [None, [30.0, -20.0]], ids=["cartesian", "rotated"])
@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize("single_time", [False, True], ids=["interpolated", "exact-time"])
def test_weather_station_wind_direction_interpolation(
    tmp_path, directions, speeds, expected, rotation, dtype, single_time
):
    """Rotate and interpolate wind vectors at the station's full-grid cell."""
    wrf_path = tmp_path / "wrfout.nc"
    obs_path = tmp_path / "obs.h5"
    out_path = tmp_path / "model.h5"
    station_path = f"{TIME_SERIES}/station_1"
    wd_name = svn.WIND_DIRECTION.value
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    obs_time = np.array([0.0] if single_time else [0.0, 60.0, 120.0])

    wd_model = np.radians(directions)
    u_true = -np.array(speeds) * np.sin(wd_model)
    v_true = -np.array(speeds) * np.cos(wd_model)
    angle = np.radians([0.0, 0.0] if rotation is None else rotation)
    # Invert the grid-to-earth rotation to construct the WRF inputs.
    u10 = u_true * np.cos(angle) + v_true * np.sin(angle)
    v10 = -u_true * np.sin(angle) + v_true * np.cos(angle)

    with Dataset(wrf_path, "w") as wrf:
        wrf.createDimension("Time", 2)
        wrf.createDimension("south_north", 4)
        wrf.createDimension("west_east", 4)
        dims = ("Time", "south_north", "west_east")
        lat = np.broadcast_to(np.arange(38.0, 42.0)[:, None], (4, 4))
        lon = np.broadcast_to(np.arange(-122.0, -118.0)[None, :], (4, 4))
        for name, values in (("XLAT", lat), ("XLONG", lon), ("FXLAT", lat), ("FXLONG", lon)):
            wrf.createVariable(name, "f8", dims)[:] = np.broadcast_to(values, (2, 4, 4))
        wrf.createVariable("XTIME", "f8", ("Time",))[:] = [0.0, 2.0]
        # Other cells deliberately differ: trimmed indices must be offset before reading.
        for name, values in (("U10", u10), ("V10", v10)):
            field = wrf.createVariable(name, dtype, dims)
            field[:] = 1.0
            field[:, 1, 1] = values
        if rotation is not None:
            for name, values in (("COSALPHA", np.cos(angle)), ("SINALPHA", np.sin(angle))):
                field = wrf.createVariable(name, dtype, dims)
                field[:] = 0.0
                field[:, 1, 1] = values

    with h5py.File(obs_path, "w") as obs_h5:
        station = obs_h5.create_group(station_path)
        station.attrs["position_lat"] = 39.0
        station.attrs["position_lon"] = -121.0
        time = station.create_dataset("time", data=obs_time)
        time.attrs["time_origin"] = origin.isoformat()
        time.attrs["time_units"] = "s"
        wd = station.create_dataset(wd_name, data=np.zeros_like(obs_time))
        wd.attrs["sensor_height"] = 10.0
        wd.attrs["sensor_height_units"] = "m"

    with (
        h5py.File(out_path, "w") as new_h5,
        h5py.File(obs_path, "r") as obs_h5,
        Dataset(wrf_path, "r") as wrf,
    ):
        get_weather_stations(
            new_h5,
            obs_h5,
            wrf,
            origin.isoformat(),
            origin,
            origin + timedelta(seconds=120),
            fire_grid_bound=1,
            compression_lvl=1,
        )

    with h5py.File(out_path, "r") as out_h5:
        result = out_h5[f"{station_path}/{wd_name}"]
        assert result.attrs["units"] == "degree"
        assert np.all((result[:] >= 0.0) & (result[:] < 360.0))
        # Compare on the circle: 0 and 360 are the same direction.
        expected = np.array(expected[:1] if single_time else expected)
        angular_error = (result[:] - expected + 180.0) % 360.0 - 180.0
        np.testing.assert_allclose(angular_error, 0.0, atol=1e-5)
        time = out_h5[f"{station_path}/time"]
        np.testing.assert_array_equal(time[:], obs_time)
        assert time.attrs["time_units"] == "s"
        assert datetime.fromisoformat(time.attrs["time_origin"]) == origin
