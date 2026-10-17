from datetime import datetime, timedelta, timezone

import h5py
import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from firebench import svn
from firebench.standardize import TIME_SERIES
from netCDF4 import Dataset

from firebench_adapter_wrf_sfire._wrfout import WrfDomain, find_wrfout_files, open_run
from firebench_adapter_wrf_sfire.cli import main

ORIGIN = datetime(2026, 1, 1, tzinfo=timezone.utc)
CBI_GROUP = "/spatial_2d/ravg_cbi"
CBI_PATH = f"{CBI_GROUP}/ravg_composite_burn_index_severity"
BD_GROUP = "/points/building_damage"
INNER = (39.0, -121.0)  # inside d02, so inside d01 too
OUTER = (42.0, -119.0)  # inside d01 only
FAR = (50.0, -100.0)  # inside neither
WIND_HEIGHT = 6.1


def _grid(lat0, lon0, n, step):
    lat = lat0 + step * np.arange(n)[:, None] * np.ones((1, n))
    lon = lon0 + step * np.arange(n)[None, :] * np.ones((n, 1))
    return lat, lon


def _write_wrfout(path, xtime, lat, lon, fields, fire=True):
    with Dataset(path, "w") as wrf:
        wrf.SIMULATION_START_DATE = "2026-01-01_00:00:00"
        ny, nx = lat.shape
        wrf.createDimension("Time", len(xtime))
        wrf.createDimension("south_north", ny)
        wrf.createDimension("west_east", nx)
        dims = ("Time", "south_north", "west_east")
        wrf.createVariable("XTIME", "f8", ("Time",))[:] = xtime
        fire_lat, fire_lon = (lat, lon) if fire else (np.zeros_like(lat), np.zeros_like(lon))
        static = {"XLAT": lat, "XLONG": lon, "FXLAT": fire_lat, "FXLONG": fire_lon}
        for name, value in {**static, **fields}.items():
            wrf.createVariable(name, "f8", dims)[:] = np.stack(
                [np.broadcast_to(value(t) if callable(value) else value, (ny, nx)) for t in xtime]
            )


def _write_run(folder, d01_fire=False, drop=()):
    """d01: 9x9 cells of 1 degree, one file, hourly. d02: 5x5 cells of 0.25 degree, three files
    every 30 minutes, the second starting with the last time of the first, as after a restart."""
    folder.mkdir(parents=True, exist_ok=True)
    common = {"U10": 3.0, "V10": 4.0, "ZNT": 0.1}
    d01 = {**common, "T2": lambda t: 280.0 + t, "FMC_GC": 0.1, "FUEL_FRAC": 0.9, "FIRE_AREA": 0.0}
    d02 = {**common, "T2": lambda t: 300.0 + t, "WZ0": 0.5, "FUEL_FRAC": 0.1, "FIRE_AREA": 1.0}
    # A reduced output (WRF's iofields_filename) leaves variables out.
    d01 = {name: value for name, value in d01.items() if name not in drop}
    d02 = {name: value for name, value in d02.items() if name not in drop}
    lat1, lon1 = _grid(36.0, -125.0, 9, 1.0)
    lat2, lon2 = _grid(38.5, -121.5, 5, 0.25)
    _write_wrfout(folder / "wrfout_d01_2026-01-01_00:00:00", [0.0, 60.0, 120.0], lat1, lon1, d01, d01_fire)
    for name, xtime in (
        ("wrfout_d02_2026-01-01_01:00:00", [60.0, 90.0]),
        ("wrfout_d02_2026-01-01_00:00:00", [0.0, 30.0, 60.0]),
        ("wrfout_d02_2026-01-01_02:00:00", [120.0]),
    ):
        _write_wrfout(folder / name, xtime, lat2, lon2, d02)
    (folder / "rsl.out.0000").write_text("not a wrfout", encoding="utf-8")
    (folder / "namelist.input").write_text("not a wrfout", encoding="utf-8")


def _add_station(h5, name, position, variables):
    station = h5.create_group(f"{TIME_SERIES}/{name}")
    station.attrs["position_lat"], station.attrs["position_lon"] = position
    time = station.create_dataset("time", data=np.array([0.0, 1800.0, 3600.0, 5400.0, 7200.0]))
    time.attrs["time_origin"] = ORIGIN.isoformat()
    time.attrs["time_units"] = "s"
    for variable in variables:
        dataset = station.create_dataset(variable, data=np.full(5, np.nan))
        dataset.attrs["sensor_height"] = WIND_HEIGHT if variable == svn.WIND_SPEED.value else 2.0
        dataset.attrs["sensor_height_units"] = "m"


def _write_template(path):
    points = (INNER, OUTER, FAR)
    with h5py.File(path, "w") as h5:
        h5.attrs["firebench_template"] = "model-output"
        variables = (svn.AIR_TEMPERATURE.value, svn.WIND_SPEED.value, svn.FUEL_MOISTURE_CONTENT_10H.value)
        for name, position in zip(("station_inner", "station_outer", "station_far"), points):
            _add_station(h5, name, position, variables)
        group = h5.create_group(CBI_GROUP)
        group.attrs["crs"] = "EPSG:4326"
        group.create_dataset("position_lat", data=np.array([[p[0] for p in points]]))
        group.create_dataset("position_lon", data=np.array([[p[1] for p in points]]))
        cbi = group.create_dataset(CBI_PATH, data=np.zeros((1, 3), dtype=np.uint8))
        cbi.attrs["units"] = "dimensionless"
        cbi.attrs["_FillValue"] = 0
        group = h5.create_group(BD_GROUP)
        group.create_dataset("position_lat", data=np.array([INNER[0], OUTER[0]]))
        group.create_dataset("position_lon", data=np.array([INNER[1], OUTER[1]]))
        group.create_dataset("building_damage", data=np.array([b"", b""], dtype="S16"))


def _config(tmp_path, wrf_path, **sections):
    main_config = {
        "template_path": "template.h5",
        "wrf_path": str(wrf_path),
        "output_path": "output/model.h5",
        "authors": "Test Author",
        "start_dt": "2026-01-01T00:00:00Z",
        "end_dt": (ORIGIN + timedelta(hours=2)).isoformat(),
        "fire_grid_bound": 0,
        "compression_lvl": 1,
    }
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump({"main": main_config, **sections}), encoding="utf-8")
    return config_path


@pytest.fixture
def case(tmp_path):
    _write_template(tmp_path / "template.h5")
    return tmp_path


def _run(config_path):
    return CliRunner().invoke(main, ["run", str(config_path)])


def _wind(z0):
    return 5.0 * np.log(WIND_HEIGHT / z0) / np.log(10.0 / z0)


def test_a_folder_gives_its_wrfout_files_by_domain_in_time_order(tmp_path):
    _write_run(tmp_path / "run")

    files = find_wrfout_files(tmp_path / "run")

    assert sorted(files) == [1, 2]
    assert [item.name[11:24] for item in files[2]] == ["2026-01-01_00", "2026-01-01_01", "2026-01-01_02"]
    with open_run(tmp_path / "run") as run:
        assert [domain.name for domain in run.domains] == ["d02", "d01"]
        assert run.start_time() == "2026-01-01T00:00:00Z"
        assert [domain.name for domain in run.fire_domains()] == ["d02"]


def test_one_file_is_a_run_of_one_domain(tmp_path):
    _write_run(tmp_path / "run")

    with open_run(tmp_path / "run" / "wrfout_d01_2026-01-01_00:00:00") as run:
        assert [domain.name for domain in run.domains] == ["d01"]
        # Alone, a domain is used for the fire grid sections as before, fire grid or not.
        assert run.fire_domains() == run.domains


def test_a_folder_without_wrfout_is_refused(tmp_path):
    (tmp_path / "rsl.out.0000").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no wrfout_dNN_"):
        find_wrfout_files(tmp_path)


def test_a_domain_is_one_time_series_across_its_files(tmp_path):
    _write_run(tmp_path / "run")

    with open_run(tmp_path / "run") as run:
        d02 = run.domains[0]
        # 60 min is in two files: the second gives it again and is left out.
        np.testing.assert_array_equal(d02.xtime(), [0.0, 30.0, 60.0, 90.0, 120.0])
        np.testing.assert_array_equal(d02.series("T2", slice(1, 5), 2, 2), [330.0, 360.0, 390.0, 420.0])


def test_a_point_beyond_the_corners_of_a_rotated_grid_is_outside(tmp_path):
    # A grid turned by 45 degrees: a diamond whose bounding box holds (0.9, 0.9).
    lat = np.array([[-1.0, 0.0], [0.0, 1.0]])
    lon = np.array([[0.0, 1.0], [-1.0, 0.0]])
    _write_wrfout(tmp_path / "wrfout_d01_x", [0.0], lat, lon, {})

    with Dataset(tmp_path / "wrfout_d01_x") as dataset:
        domain = WrfDomain(1, [dataset])
        inside = domain.covers(np.array([0.0, 0.5, 0.9]), np.array([0.0, 0.5, 0.9]), slice(None))

    np.testing.assert_array_equal(inside, [True, True, False])


def test_weather_takes_each_station_from_the_finest_domain_that_gives_it(case):
    _write_run(case / "run")
    config_path = _config(case, case / "run", weather={"enabled": True})

    result = _run(config_path)

    assert result.exit_code == 0, result.output
    assert "WRF-SFIRE domain d02: 3 file(s), XTIME 0 to 120 min, fire grid" in result.output
    assert "WRF-SFIRE domain d01: 1 file(s), XTIME 0 to 120 min, no fire grid" in result.output
    assert "sfire_ref_time 2026-01-01T00:00:00Z" in result.output
    assert "WARNING: time_series/station_far skipped: outside every model domain" in result.output
    assert "1 item(s) skipped" in result.output
    with h5py.File(case / "output" / "model.h5", "r") as out_h5:
        inner = out_h5[f"{TIME_SERIES}/station_inner"]
        np.testing.assert_allclose(inner[svn.AIR_TEMPERATURE.value][:], [300, 330, 360, 390, 420])
        np.testing.assert_allclose(inner[svn.WIND_SPEED.value][:], np.full(5, _wind(0.5)))
        # d02 has no FMC_GC: d01 gives it.
        np.testing.assert_allclose(inner[svn.FUEL_MOISTURE_CONTENT_10H.value][:], np.full(5, 0.1))

        outer = out_h5[f"{TIME_SERIES}/station_outer"]
        np.testing.assert_allclose(outer[svn.AIR_TEMPERATURE.value][:], [280, 310, 340, 370, 400])
        # d01 has no fire grid: the wind speed profile takes ZNT.
        np.testing.assert_allclose(outer[svn.WIND_SPEED.value][:], np.full(5, _wind(0.1)))
        assert f"{TIME_SERIES}/station_far" not in out_h5


def test_fire_grid_sections_take_the_finest_fire_grid_that_holds_a_point(case):
    _write_run(case / "run", d01_fire=True)
    config_path = _config(
        case,
        case / "run",
        burn_severity_cbi={"enabled": True, "group_path": CBI_GROUP},
        building_damage={"enabled": True, "group_path": BD_GROUP},
    )

    result = _run(config_path)

    assert result.exit_code == 0, result.output
    with h5py.File(case / "output" / "model.h5", "r") as out_h5:
        # FUEL_FRAC 0.1 in d02 is high (4), 0.9 in d01 unchanged (1); the third is outside both.
        np.testing.assert_array_equal(out_h5[CBI_PATH][:], [[4, 1, 0]])
        damage = [value.decode() for value in out_h5[f"{BD_GROUP}/building_damage"][:]]
        assert damage == ["Destroyed (>50%)", "No Damage"]


def test_sfire_ref_time_is_required_when_the_output_does_not_give_it(case):
    _write_run(case / "run")
    for item in (case / "run").glob("wrfout_*"):
        with Dataset(item, "a") as wrf:
            wrf.delncattr("SIMULATION_START_DATE")
    config_path = _config(case, case / "run", weather={"enabled": True})

    result = _run(config_path)

    assert result.exit_code != 0
    assert "main.sfire_ref_time is required" in result.output
    assert not (case / "output" / "model.h5").exists()


def test_a_station_reads_its_own_cell_when_the_grid_is_trimmed(case):
    lat, lon = _grid(38.0, -122.0, 6, 0.5)
    cells = 10.0 * np.arange(6)[:, None] + np.arange(6)[None, :]
    _write_wrfout(case / "wrfout_d01_one", [0.0, 120.0], lat, lon, {"T2": cells})
    config_path = _config(case, case / "wrfout_d01_one", weather={"enabled": True})
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["main"]["fire_grid_bound"] = 1
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = _run(config_path)

    assert result.exit_code == 0, result.output
    with h5py.File(case / "output" / "model.h5", "r") as out_h5:
        # INNER (39, -121) is the cell j=2, i=2 of the full grid.
        temperature = out_h5[f"{TIME_SERIES}/station_inner/{svn.AIR_TEMPERATURE.value}"]
        np.testing.assert_allclose(temperature[:], np.full(5, 22.0))


def test_a_reduced_output_skips_what_it_lacks_and_names_it(case):
    _write_run(case / "run", drop=("FUEL_FRAC", "ZNT"))
    config_path = _config(
        case,
        case / "run",
        burn_severity_cbi={"enabled": True, "group_path": CBI_GROUP},
        weather={"enabled": True},
    )

    result = _run(config_path)

    assert result.exit_code == 0, result.output
    assert (
        "WARNING: Section burn_severity_cbi skipped: 'FUEL_FRAC is not in the output of d02'"
        in result.output
    )
    # d02 has WZ0; d01 has neither WZ0 nor ZNT.
    assert (
        "WARNING: time_series/station_outer/wind_speed skipped: "
        "'no roughness length (WZ0 or ZNT) in the output of d01'" in result.output
    )
    with h5py.File(case / "output" / "model.h5", "r") as out_h5:
        assert "spatial_2d" not in out_h5
        assert svn.WIND_SPEED.value in out_h5[f"{TIME_SERIES}/station_inner"]
        assert svn.AIR_TEMPERATURE.value in out_h5[f"{TIME_SERIES}/station_outer"]


def test_a_field_laid_out_otherwise_is_skipped_with_its_dimensions(case):
    lat, lon = _grid(38.0, -122.0, 6, 0.5)
    _write_wrfout(case / "wrfout_d01_one", [0.0, 120.0], lat, lon, {})
    with Dataset(case / "wrfout_d01_one", "a") as wrf:
        wrf.createVariable("T2", "f8", ("south_north", "west_east"))[:] = 290.0
    config_path = _config(case, case / "wrfout_d01_one", weather={"enabled": True})

    result = _run(config_path)

    assert result.exit_code == 0, result.output
    assert (
        "time_series/station_inner/air_temperature skipped: T2 of d01 has the dimensions "
        "('south_north', 'west_east'), not the 3 expected" in result.output
    )


def test_minus_o_replaces_an_existing_output(case):
    _write_run(case / "run")
    config_path = _config(case, case / "run", weather={"enabled": True})
    assert _run(config_path).exit_code == 0

    refused = _run(config_path)
    replaced = CliRunner().invoke(main, ["run", "-o", str(config_path)])

    assert refused.exit_code != 0
    assert "Output file already exists" in refused.output
    assert "run with -o to replace it" in refused.output
    assert replaced.exit_code == 0, replaced.output
