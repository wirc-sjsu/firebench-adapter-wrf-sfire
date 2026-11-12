"""The adapter run from a forecast window's kit: the model is not laid out on observations."""

import zipfile
from datetime import datetime, timezone

import h5py
import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from firebench import svn
from firebench.standardize import TIME_SERIES
from netCDF4 import Dataset

fire_window = pytest.importorskip("firebench.benchmarks.fire_window", reason="FireBench has no windows")
from firebench.benchmarks.fire_case_definition import definition_sha256  # noqa: E402

from firebench_adapter_wrf_sfire import get_arrival_time_from_t_ign_g  # noqa: E402
from firebench_adapter_wrf_sfire.cli import main  # noqa: E402

ORIGIN = datetime(2026, 1, 1, tzinfo=timezone.utc)
START = datetime(2026, 1, 1, 1, 10, tzinfo=timezone.utc)
END = datetime(2026, 1, 1, 2, 20, tzinfo=timezone.utc)
FRAME = [-121.1, 37.9, -120.8, 38.2]
FIRE_FRAME = [-120.99, 38.01, -120.95, 38.05]
INSIDE = (38.035, -120.965)
OUTSIDE = (38.15, -120.85)  # in the frame, beyond the model's grid
VARIABLES = (
    svn.AIR_TEMPERATURE.value,
    svn.RELATIVE_HUMIDITY.value,
    svn.WIND_SPEED.value,
    svn.WIND_DIRECTION.value,
    svn.FUEL_MOISTURE_CONTENT_10H.value,
)
GROUP = "/spatial_2d/fire_window"


def _write_wrf(path, xtime):
    """One domain of 8 x 8 cells of 0.01 degree, output every 30 minutes."""
    ny = nx = 8
    lat = 38.0 + 0.01 * np.arange(ny)[:, None] * np.ones((1, nx))
    lon = -121.0 + 0.01 * np.arange(nx)[None, :] * np.ones((ny, 1))
    # Burnt cells arrive every 10 minutes along a row; one cell arrives after the last output, the
    # others were never ignited (WRF-SFIRE's large time).
    tign = np.full((ny, nx), 1e9)
    tign[2, 2:6] = [600.0, 1200.0, 1800.0, 2400.0]
    tign[3, 3] = 20000.0
    with Dataset(path, "w") as wrf:
        wrf.SIMULATION_START_DATE = "2026-01-01_00:00:00"
        wrf.createDimension("Time", len(xtime))
        wrf.createDimension("south_north", ny)
        wrf.createDimension("west_east", nx)
        dims = ("Time", "south_north", "west_east")
        wrf.createVariable("XTIME", "f8", ("Time",))[:] = xtime
        fields = {
            "XLAT": lat,
            "XLONG": lon,
            "FXLAT": lat,
            "FXLONG": lon,
            "TIGN_G": tign,
            "Q2": 0.005,
            "PSFC": 90000.0,
            "U10": 3.0,
            "V10": 4.0,
            "WZ0": 0.1,
            "FMC_GC": 0.1,
        }
        for name, value in fields.items():
            wrf.createVariable(name, "f8", dims)[:] = np.broadcast_to(value, (len(xtime), ny, nx))
        # T2 follows the hour of the output, so every output time is told apart.
        wrf.createVariable("T2", "f8", dims)[:] = np.broadcast_to(
            (300.0 + np.asarray(xtime) / 60)[:, None, None], (len(xtime), ny, nx)
        )


def _observations(path, positions):
    """What a window is proposed from: the stations known when it opens."""
    with h5py.File(path, "w") as h5:
        for index, (lat, lon) in enumerate(positions):
            station = h5.create_group(f"{TIME_SERIES}/station_{index}")
            station.attrs["position_lat"], station.attrs["position_lon"] = lat, lon
            for variable in VARIABLES:
                dataset = station.create_dataset(variable, data=np.zeros(3))
                dataset.attrs["sensor_height"] = 6.1 if variable == svn.WIND_SPEED.value else 2.0
                dataset.attrs["sensor_height_units"] = "m"


def _kit(tmp_path, xtime=(0, 30, 60, 90, 120, 150, 180), positions=(INSIDE,), **sections):
    """A window kit folder with a run beside it: its template, its definition, and its config."""
    (tmp_path / "adapters").mkdir()
    _write_wrf(tmp_path / "adapters" / "wrfout.nc", list(xtime))
    _observations(tmp_path / "observations.h5", positions)
    definition = fire_window.propose_window(
        "FW-TEST", START, END, FRAME, fire_frame=FIRE_FRAME, std_files=[tmp_path / "observations.h5"]
    )["definition"]
    fire_window.write_window_template(definition, tmp_path / "template.h5")
    config = {
        "main": {
            "template_path": "../template.h5",
            "wrf_path": "wrfout.nc",
            "output_path": "../output/FW-TEST_WRF-SFIRE.h5",
            "authors": "Test Author",
            "fire_grid_bound": 0,
            "compression_lvl": 1,
        },
        "window": {"name": "FW-TEST", "definition_sha256": definition_sha256(definition)},
        **({"arrival_time": {"enabled": True}, "weather": {"enabled": True}} | sections),
    }
    config_path = tmp_path / "adapters" / "wrf-sfire.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, definition


def _run(config_path):
    return CliRunner().invoke(main, ["run", str(config_path)])


def test_window_run_is_eligible_on_both_families(tmp_path):
    config_path, definition = _kit(tmp_path)

    result = _run(config_path)

    assert result.exit_code == 0, result.output
    output = tmp_path / "output" / "FW-TEST_WRF-SFIRE.h5"
    report = fire_window.check_model_for_window(definition, output)
    assert report["accepted"], report["reason"]
    assert {item["family"]: item["status"] for item in report["families"]} == {
        "fire_arrival_time": "eligible",
        "weather_stations": "eligible",
    }
    with zipfile.ZipFile(output.with_suffix(".zip")) as bundle:
        assert bundle.namelist() == [output.name]

    with h5py.File(output, "r") as out_h5:
        assert "firebench_template" not in out_h5.attrs
        arrival = out_h5[f"{GROUP}/fire_arrival_time"]
        assert arrival.attrs["units"] == "s"
        assert arrival.attrs["time_origin"] == "2026-01-01T00:00:00+00:00"
        field = arrival[:]
        np.testing.assert_array_equal(field[2, 2:6], [600.0, 1200.0, 1800.0, 2400.0])
        # Never ignited (1e9) and not yet burnt at the last output (20000 s > 10800 s): NaN.
        assert np.isnan(field[3, 3]) and np.isnan(field[0, 0])
        assert np.isfinite(field).sum() == 4
        assert out_h5[f"{GROUP}/position_lat"].shape == field.shape
        assert out_h5[f"{GROUP}/time"].attrs["time_units"] == "s"
        np.testing.assert_array_equal(out_h5[f"{GROUP}/time"][:], np.arange(0, 181, 30) * 60.0)

        station = out_h5[f"{TIME_SERIES}/station_0"]
        # The model's own outputs from the last at or before the start to the first at or after the
        # end: 01:00, 01:30, 02:00, 02:30, in seconds after the start of the period, not interpolated.
        np.testing.assert_array_equal(station["time"][:], [-600.0, 1200.0, 3000.0, 4800.0])
        assert station["time"].attrs["time_origin"] == "2026-01-01T01:10:00+00:00"
        assert station["time"].attrs["time_units"] == "s"
        np.testing.assert_allclose(station[svn.AIR_TEMPERATURE.value][:], [301.0, 301.5, 302.0, 302.5])
        assert station[svn.AIR_TEMPERATURE.value].attrs["units"] == "degK"
        assert station[svn.AIR_TEMPERATURE.value].attrs["sensor_height"] == 2.0
        assert station[svn.WIND_SPEED.value].attrs["sensor_height"] == 6.1
        for variable in VARIABLES:
            assert station[variable].shape == (4,)


def test_window_run_reports_the_station_outside_the_model(tmp_path):
    config_path, definition = _kit(tmp_path, positions=(INSIDE, OUTSIDE))

    result = _run(config_path)

    assert result.exit_code == 0, result.output
    assert "station_1 skipped" in result.output and "outside every model domain" in result.output
    output = tmp_path / "output" / "FW-TEST_WRF-SFIRE.h5"
    with h5py.File(output, "r") as out_h5:
        assert f"{TIME_SERIES}/station_0" in out_h5 and f"{TIME_SERIES}/station_1" not in out_h5
    report = fire_window.check_model_for_window(definition, output)
    stations = next(item for item in report["families"] if item["family"] == "weather_stations")
    assert stations["status"] == "unavailable"


def test_window_run_warns_when_the_run_stops_before_the_period_ends(tmp_path):
    config_path, definition = _kit(tmp_path, xtime=(0, 30, 60, 90, 120))

    result = _run(config_path)

    assert result.exit_code == 0, result.output
    assert result.output.count("outside the run's output") == 2
    report = fire_window.check_model_for_window(definition, tmp_path / "output" / "FW-TEST_WRF-SFIRE.h5")
    assert {item["family"]: item["status"] for item in report["families"]} == {
        "fire_arrival_time": "unavailable",
        "weather_stations": "unavailable",
    }


def test_window_run_with_one_section(tmp_path):
    config_path, definition = _kit(tmp_path, weather={"enabled": False})

    assert _run(config_path).exit_code == 0

    report = fire_window.check_model_for_window(definition, tmp_path / "output" / "FW-TEST_WRF-SFIRE.h5")
    assert {item["family"]: item["status"] for item in report["families"]} == {
        "fire_arrival_time": "eligible",
        "weather_stations": "unavailable",
    }


def _edit(config_path, **changes):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for section, values in changes.items():
        config[section] = values if values is None else {**config.get(section, {}), **values}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")


@pytest.mark.parametrize(
    "changes, message",
    [
        pytest.param(
            {"window": {"definition_sha256": "0" * 64}}, "another version of the window", id="other-hash"
        ),
        pytest.param({"window": {"name": "FW-OTHER"}}, "window.name is FW-OTHER", id="other-name"),
        pytest.param(
            {"perimeters": {"enabled": True, "paths": ["/x"]}}, "cannot be enabled", id="perimeters"
        ),
        pytest.param(
            {"arrival_time": {"enabled": False}, "weather": {"enabled": False}},
            "At least one of arrival_time and weather",
            id="nothing-enabled",
        ),
        pytest.param({"main": {"obs_path": "x.h5"}}, "not main.obs_path", id="obs-path"),
    ],
)
def test_window_config_refusals(tmp_path, changes, message):
    config_path, _ = _kit(tmp_path)
    _edit(config_path, **changes)

    result = _run(config_path)

    assert result.exit_code != 0
    assert message in result.output
    assert not (tmp_path / "output").exists()


def test_window_refuses_a_template_that_is_not_a_window(tmp_path):
    config_path, _ = _kit(tmp_path)
    with h5py.File(tmp_path / "template.h5", "w") as h5:
        h5.attrs["firebench_template"] = "model-output"

    result = _run(config_path)

    assert result.exit_code != 0
    assert "is not a forecast window's template" in result.output


def test_window_refuses_a_firebench_without_windows(tmp_path, monkeypatch):
    config_path, _ = _kit(tmp_path)
    monkeypatch.setitem(__import__("sys").modules, "firebench.benchmarks.fire_window", None)

    result = _run(config_path)

    assert result.exit_code != 0
    assert "needs a FireBench with forecast windows" in result.output


def test_arrival_time_is_refused_in_case_mode(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "main": {"obs_path": "x.h5", "wrf_path": "w", "output_path": "o.h5", "authors": "a"},
                "arrival_time": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )

    result = _run(config)

    assert result.exit_code != 0
    assert "arrival_time is a section of a forecast window's config" in result.output


def test_arrival_time_api_trims_the_fire_grid(tmp_path):
    _write_wrf(tmp_path / "wrfout.nc", [0.0, 60.0])
    with h5py.File(tmp_path / "out.h5", "w") as new_h5, Dataset(tmp_path / "wrfout.nc") as wrf:
        get_arrival_time_from_t_ign_g(new_h5, wrf, GROUP, None, fire_grid_bound=1, compression_lvl=1)
        field = new_h5[f"{GROUP}/fire_arrival_time"][:]
        assert field.shape == (6, 6)
        # The last output is 3600 s: the cells that arrive later than that are not burnt.
        np.testing.assert_array_equal(field[1, 1:5], [600.0, 1200.0, 1800.0, 2400.0])
        assert new_h5[f"{GROUP}/time"].attrs["time_origin"] == "2026-01-01T00:00:00+00:00"
