from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from firebench import svn
from firebench.benchmarks.fire_case_definition import definition_sha256, write_case_definition
from firebench.standardize import TIME_SERIES
from netCDF4 import Dataset

import firebench_adapter_wrf_sfire.perimeters as perimeters_module
from firebench_adapter_wrf_sfire.cli import main

CBI_GROUP = "/spatial_2d/ravg_cbi"
CBI_PATH = f"{CBI_GROUP}/ravg_composite_burn_index_severity"
ORIGIN = datetime(2026, 1, 1, tzinfo=timezone.utc)
DEFINITION = {"schema": "fb-case-v1", "name": "synthetic"}


def _write_wrf(path):
    with Dataset(path, "w") as wrf:
        wrf.createDimension("Time", 2)
        wrf.createDimension("south_north", 4)
        wrf.createDimension("west_east", 4)
        dims = ("Time", "south_north", "west_east")
        lat = np.broadcast_to(np.arange(38.0, 42.0)[:, None], (4, 4))
        lon = np.broadcast_to(np.arange(-122.0, -118.0)[None, :], (4, 4))
        for name, values in (("XLAT", lat), ("XLONG", lon), ("FXLAT", lat), ("FXLONG", lon)):
            wrf.createVariable(name, "f8", dims)[:] = np.broadcast_to(values, (2, 4, 4))
        wrf.createVariable("XTIME", "f8", ("Time",))[:] = [0.0, 2.0]
        wrf.createVariable("T2", "f8", dims)[:] = np.broadcast_to(
            np.array([290.0, 292.0])[:, None, None], (2, 4, 4)
        )
        fuel_frac = np.array(
            [[0.1, 0.5, 0.7, 0.9], [0.1, 0.5, 0.7, 0.9], [0.1, 0.5, 0.7, 0.9], [0.1, 0.5, 0.7, 0.9]]
        )
        wrf.createVariable("FUEL_FRAC", "f8", dims)[:] = np.broadcast_to(fuel_frac, (2, 4, 4))


def _write_template(path, definition=DEFINITION):
    """A model-output template: blank values on the observation layout, and the case definition."""
    with h5py.File(path, "w") as h5:
        h5.attrs["firebench_template"] = "model-output"
        group = h5.create_group(CBI_GROUP)
        group.attrs["crs"] = "EPSG:4326"
        group.create_dataset("position_lat", data=np.array([[38.0, 38.0, 45.0], [39.0, 39.0, 45.0]]))
        group.create_dataset(
            "position_lon", data=np.array([[-122.0, -121.0, -100.0], [-122.0, -121.0, -100.0]])
        )
        cbi = group.create_dataset(CBI_PATH, data=np.zeros((2, 3), dtype=np.uint8))
        cbi.attrs["units"] = "dimensionless"
        cbi.attrs["_FillValue"] = 0

        obs_time = np.array([0.0, 60.0, 120.0])
        for name, lat, variables in (
            ("station_in", 39.0, (svn.AIR_TEMPERATURE.value,)),
            ("station_out", 50.0, (svn.AIR_TEMPERATURE.value,)),
        ):
            station = h5.create_group(f"{TIME_SERIES}/{name}")
            station.attrs["position_lat"] = lat
            station.attrs["position_lon"] = -121.0
            time = station.create_dataset("time", data=obs_time)
            time.attrs["time_origin"] = ORIGIN.isoformat()
            time.attrs["time_units"] = "s"
            for variable in variables:
                dataset = station.create_dataset(variable, data=np.full_like(obs_time, np.nan))
                dataset.attrs["sensor_height"] = 2.0
                dataset.attrs["sensor_height_units"] = "m"
    if definition is not None:
        write_case_definition(path, definition)


def _config(tmp_path, main_extra=None, **sections):
    main_config = {
        "wrf_path": "wrfout.nc",
        "output_path": "output/model.h5",
        "authors": "Test Author",
        "sfire_ref_time": ORIGIN.isoformat(),
        "start_dt": ORIGIN.isoformat().replace("+00:00", "Z"),
        "end_dt": (ORIGIN + timedelta(seconds=120)).isoformat().replace("+00:00", "Z"),
        "fire_grid_bound": 0,
        "compression_lvl": 1,
        **({"template_path": "template.h5"} if main_extra is None else main_extra),
    }
    config_path = tmp_path / "adapters" / "config.yml"
    config_path.write_text(yaml.safe_dump({"main": main_config, **sections}), encoding="utf-8")
    return config_path


@pytest.fixture
def kit(tmp_path):
    (tmp_path / "adapters").mkdir()
    _write_wrf(tmp_path / "adapters" / "wrfout.nc")
    _write_template(tmp_path / "adapters" / "template.h5")
    return tmp_path


def _run(config_path):
    return CliRunner().invoke(main, ["run", str(config_path)])


def test_cli_runs_cbi_and_weather_on_a_template(kit):
    config_path = _config(
        kit,
        case={"id": "FBF-0000", "definition_sha256": definition_sha256(DEFINITION)},
        burn_severity_cbi={"enabled": True, "group_path": CBI_GROUP},
        weather={"enabled": True},
    )

    result = _run(config_path)

    assert result.exit_code == 0, result.output
    with h5py.File(kit / "adapters" / "output" / "model.h5", "r") as out_h5:
        cbi = out_h5[CBI_PATH]
        # FUEL_FRAC 0.1 is high (4), 0.5 moderate (3); the third column is outside the fire grid.
        np.testing.assert_array_equal(cbi[:], np.array([[4, 3, 0], [4, 3, 0]], dtype=np.uint8))
        assert cbi.dtype == np.uint8
        assert cbi.attrs["_FillValue"] == 0
        assert "benchmark_case" not in out_h5

        temperature = out_h5[f"{TIME_SERIES}/station_in/{svn.AIR_TEMPERATURE.value}"]
        np.testing.assert_allclose(temperature[:], [290.0, 291.0, 292.0])
        assert temperature.attrs["sensor_height"] == 2.0
        assert temperature.attrs["sensor_height_units"] == "m"
        assert f"{TIME_SERIES}/station_in/{svn.RELATIVE_HUMIDITY.value}" not in out_h5
        assert f"{TIME_SERIES}/station_out" not in out_h5


def test_cli_refuses_obs_path_and_template_path(kit):
    config_path = _config(
        kit,
        main_extra={"template_path": "template.h5", "obs_path": "template.h5"},
        burn_severity_cbi={"enabled": True, "group_path": CBI_GROUP},
    )

    result = _run(config_path)

    assert result.exit_code != 0
    assert "not both" in result.output


def test_cli_requires_a_reference_file(kit):
    config_path = _config(kit, main_extra={}, burn_severity_cbi={"enabled": True, "group_path": CBI_GROUP})

    result = _run(config_path)

    assert result.exit_code != 0
    assert "main.template_path" in result.output


def test_cli_refuses_a_template_path_that_is_no_template(kit):
    with h5py.File(kit / "adapters" / "obs.h5", "w"):
        pass
    config_path = _config(
        kit,
        main_extra={"template_path": "obs.h5"},
        burn_severity_cbi={"enabled": True, "group_path": CBI_GROUP},
    )

    result = _run(config_path)

    assert result.exit_code != 0
    assert "not a model-output template" in result.output


def test_cli_refuses_another_version_of_the_case(kit):
    config_path = _config(
        kit,
        case={"id": "FBF-0000", "definition_sha256": "0" * 64},
        burn_severity_cbi={"enabled": True, "group_path": CBI_GROUP},
    )

    result = _run(config_path)

    assert result.exit_code != 0
    assert "made for another version of the case FBF-0000" in result.output
    assert not (kit / "adapters" / "output" / "model.h5").exists()


def _add_station(path, name, times, variables, lat=39.0):
    with h5py.File(path, "a") as h5:
        station = h5.create_group(f"{TIME_SERIES}/{name}")
        station.attrs["position_lat"] = lat
        station.attrs["position_lon"] = -121.0
        time = station.create_dataset("time", data=np.array(times, dtype=float))
        time.attrs["time_origin"] = ORIGIN.isoformat()
        time.attrs["time_units"] = "s"
        for variable in variables:
            dataset = station.create_dataset(variable, data=np.full(len(times), np.nan))
            dataset.attrs["sensor_height"] = 2.0
            dataset.attrs["sensor_height_units"] = "m"


def test_weather_keeps_what_the_simulation_gives(kit):
    template = kit / "adapters" / "template.h5"
    _add_station(
        template,
        "station_late",
        [0.0, 60.0, 120.0, 180.0, 240.0],
        (svn.AIR_TEMPERATURE.value, svn.WIND_SPEED.value),
    )
    _add_station(template, "station_after", [600.0, 660.0], (svn.AIR_TEMPERATURE.value,))
    end_dt = (ORIGIN + timedelta(seconds=700)).isoformat()
    config_path = _config(
        kit, main_extra={"template_path": "template.h5", "end_dt": end_dt}, weather={"enabled": True}
    )

    result = _run(config_path)

    assert result.exit_code == 0, result.output
    assert "WARNING: time_series/station_late/wind_speed skipped" in result.output
    assert (
        "WARNING: time_series/station_after skipped: no observation time inside the simulation"
        in result.output
    )
    assert "WARNING: time_series/station_out skipped: outside the model domain" in result.output
    assert "3 item(s) skipped" in result.output
    with h5py.File(kit / "adapters" / "output" / "model.h5", "r") as out_h5:
        late = out_h5[f"{TIME_SERIES}/station_late"]
        # The simulation ends at 120 s: the later observation times are left out.
        np.testing.assert_array_equal(late["time"][:], [0.0, 60.0, 120.0])
        np.testing.assert_allclose(late[svn.AIR_TEMPERATURE.value][:], [290.0, 291.0, 292.0])
        assert svn.WIND_SPEED.value not in late
        assert f"{TIME_SERIES}/station_after" not in out_h5


def test_perimeters_outside_the_simulation_are_skipped(kit, monkeypatch):
    with Dataset(kit / "adapters" / "wrfout.nc", "a") as wrf:
        tign = wrf.createVariable("TIGN_G", "f8", ("Time", "south_north", "west_east"))
        tign[:] = np.broadcast_to(np.linspace(0.0, 300.0, 16).reshape(4, 4), (2, 4, 4))
    paths = {"early": -60, "mid": 60, "late": 600}
    with h5py.File(kit / "adapters" / "template.h5", "a") as h5:
        for name, seconds in paths.items():
            group = h5.create_group(f"/polygons/perimeter_{name}")
            group.attrs["time"] = (ORIGIN + timedelta(seconds=seconds)).isoformat()

    class FakePerimeters:
        empty = False

        @staticmethod
        def to_file(filename, **_kwargs):
            Path(filename).write_text("synthetic KML", encoding="utf-8")

    monkeypatch.setattr(
        perimeters_module, "_extract_perimeter_geopolygons", lambda *_a, **_k: FakePerimeters()
    )
    config_path = _config(
        kit,
        perimeters={
            "enabled": True,
            "paths": [f"/polygons/perimeter_{name}" for name in paths],
            "kml_dir": "output/kml_model",
        },
    )

    result = _run(config_path)

    assert result.exit_code == 0, result.output
    assert "WARNING: /polygons/perimeter_early skipped" in result.output
    assert (
        "WARNING: /polygons/perimeter_late skipped: its time 600.0 s is after the end of the simulation"
        in result.output
    )
    with h5py.File(kit / "adapters" / "output" / "model.h5", "r") as out_h5:
        assert list(out_h5["polygons"]) == ["perimeter_mid"]
        assert out_h5["polygons/perimeter_mid"].attrs["rel_path"].startswith("kml_model/")


def test_a_section_the_data_cannot_give_is_skipped(kit):
    config_path = _config(
        kit,
        burn_severity_cbi={"enabled": True, "group_path": "/spatial_2d/not_in_the_template"},
        weather={"enabled": True},
    )

    result = _run(config_path)

    assert result.exit_code == 0, result.output
    assert "WARNING: Section burn_severity_cbi skipped" in result.output
    with h5py.File(kit / "adapters" / "output" / "model.h5", "r") as out_h5:
        assert "spatial_2d" not in out_h5
        assert f"{TIME_SERIES}/station_in/{svn.AIR_TEMPERATURE.value}" in out_h5


def test_an_unknown_perimeter_method_stops_before_writing(kit):
    config_path = _config(
        kit, perimeters={"enabled": True, "paths": ["/polygons/a"], "extraction_method": "guess"}
    )

    result = _run(config_path)

    assert result.exit_code != 0
    assert "perimeters.extraction_method must be one of" in result.output
    assert not (kit / "adapters" / "output" / "model.h5").exists()
