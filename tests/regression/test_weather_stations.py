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
