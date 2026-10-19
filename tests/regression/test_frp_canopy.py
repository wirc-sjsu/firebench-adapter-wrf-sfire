"""The FRP band of a run, from its sensible heat flux, and RAVG canopy cover loss from its canopy burnt."""

import h5py
import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from netCDF4 import Dataset

from firebench_adapter_wrf_sfire.cli import main
from firebench_adapter_wrf_sfire.frp import EARTH_RADIUS

FRP_GROUP = "/time_series/fire_total"
CC_GROUP = "/spatial_2d/ravg_cc"
CC_PATH = f"{CC_GROUP}/ravg_canopy_cover_loss"
XTIME = [0.0, 30.0, 60.0]
STEP = 0.01
# A 4 x 4 grid of 0.01 degree from (39.0, -121.0); the frame holds its 2 x 2 cells of the south-west.
FRAME = [-121.005, 38.995, -120.995 + STEP, 39.005 + STEP]


def _grid(lat0, lon0, n, step):
    lat = lat0 + step * np.arange(n)[:, None] * np.ones((1, n))
    lon = lon0 + step * np.arange(n)[None, :] * np.ones((n, 1))
    return lat, lon


def _write_wrfout(path, lat, lon, fields):
    with Dataset(path, "w") as wrf:
        wrf.SIMULATION_START_DATE = "2026-01-01_00:00:00"
        ny, nx = lat.shape
        wrf.createDimension("Time", len(XTIME))
        wrf.createDimension("south_north", ny)
        wrf.createDimension("west_east", nx)
        dims = ("Time", "south_north", "west_east")
        wrf.createVariable("XTIME", "f8", ("Time",))[:] = XTIME
        static = {"XLAT": lat, "XLONG": lon, "FXLAT": lat, "FXLONG": lon}
        for name, value in {**static, **fields}.items():
            wrf.createVariable(name, "f8", dims)[:] = np.stack(
                [np.broadcast_to(value(t) if callable(value) else value, (ny, nx)) for t in XTIME]
            )


def _cell_area(lat_deg):
    """The area of a cell of STEP x STEP degrees at a latitude, on the sphere."""
    return EARTH_RADIUS**2 * np.cos(np.radians(lat_deg)) * np.radians(STEP) ** 2


def _write_template(path, frame=True):
    with h5py.File(path, "w") as h5:
        h5.attrs["firebench_template"] = "model-output"
        group = h5.create_group(FRP_GROUP)
        if frame:
            for key, value in zip(("frame_west", "frame_south", "frame_east", "frame_north"), FRAME):
                group.attrs[key] = value
        group = h5.create_group(CC_GROUP)
        group.attrs["crs"] = "EPSG:4326"
        # Two cells of the grid, then one far outside it.
        group.create_dataset("position_lat", data=np.array([[39.0, 39.03, 45.0]]))
        group.create_dataset("position_lon", data=np.array([[-121.0, -120.97, -100.0]]))
        values = group.create_dataset("ravg_canopy_cover_loss", data=np.zeros((1, 3), dtype=np.uint8))
        values.attrs["units"] = "dimensionless"
        values.attrs["_FillValue"] = 0


def _config(tmp_path, wrf_path, **sections):
    main_config = {
        "template_path": "template.h5",
        "wrf_path": str(wrf_path),
        "output_path": "output/model.h5",
        "authors": "Test Author",
        "fire_grid_bound": 0,
        "compression_lvl": 1,
    }
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump({"main": main_config, **sections}), encoding="utf-8")
    return config_path


def _run(config_path):
    return CliRunner().invoke(main, ["run", str(config_path)])


def _burnt():
    """The south-west cell unburnt, the others burnt."""
    fire_area = np.ones((4, 4))
    fire_area[0, 0] = 0.0
    return fire_area


@pytest.fixture
def case(tmp_path):
    _write_template(tmp_path / "template.h5")
    return tmp_path


def _one_domain(folder, **fields):
    folder.mkdir(parents=True, exist_ok=True)
    lat, lon = _grid(39.0, -121.0, 4, STEP)
    _write_wrfout(folder / "wrfout_d01_2026-01-01_00:00:00", lat, lon, fields)
    return folder


def test_the_frp_band_is_the_sensible_heat_over_the_frame_divided_by_the_factors(case):
    run = _one_domain(case / "run", FGRNHFX=lambda t: 1000.0 * t / 30.0, FCANHFX=500.0)

    result = _run(_config(case, run, frp={"enabled": True, "conversion_factor": [5, 10]}))

    assert result.exit_code == 0, result.output
    with h5py.File(case / "output" / "model.h5", "r") as out_h5:
        group = out_h5[FRP_GROUP]
        # Ground and crown fire, on the 2 x 2 cells inside the frame: two at 39.00, two at 39.01.
        area = 2 * _cell_area(39.0) + 2 * _cell_area(39.01)
        heat = (np.array([0.0, 1000.0, 2000.0]) + 500.0) * area / 1e6
        np.testing.assert_allclose(group["heat_release_rate_sensible"][:], heat, rtol=1e-6)
        np.testing.assert_allclose(group["fire_radiative_power_lower"][:], heat / 10, rtol=1e-6)
        np.testing.assert_allclose(group["fire_radiative_power_upper"][:], heat / 5, rtol=1e-6)
        assert group["fire_radiative_power_upper"].attrs["units"] == "MW"
        np.testing.assert_array_equal(group["time"][:], [0.0, 1800.0, 3600.0])
        assert group["time"].attrs["time_origin"] == "2026-01-01T00:00:00Z"
        assert group.attrs["heat_flux_variables"] == "FCANHFX,FGRNHFX"
        assert (group.attrs["conversion_factor_min"], group.attrs["conversion_factor_max"]) == (5.0, 10.0)
        assert [group.attrs[key] for key in ("frame_west", "frame_east")] == [FRAME[0], FRAME[2]]


def test_a_finer_fire_domain_takes_its_cells_from_the_coarser_one(case):
    folder = case / "run"
    folder.mkdir()
    # d01: 6 x 6 cells of 0.01 degree, 100 W m-2; d02, inside it: 3 x 3 cells of 0.005 degree, 1000.
    lat, lon = _grid(38.985, -121.015, 6, STEP)
    _write_wrfout(folder / "wrfout_d01_2026-01-01_00:00:00", lat, lon, {"FGRNHFX": 100.0})
    lat2, lon2 = _grid(39.0, -121.0, 3, STEP / 2)
    _write_wrfout(folder / "wrfout_d02_2026-01-01_00:00:00", lat2, lon2, {"FGRNHFX": 1000.0})
    frame = [-121.1, 38.9, -120.9, 39.1]

    result = _run(_config(case, folder, frp={"enabled": True, "frame": frame}))

    assert result.exit_code == 0, result.output
    with h5py.File(case / "output" / "model.h5", "r") as out_h5:
        heat = out_h5[f"{FRP_GROUP}/heat_release_rate_sensible"][:]
    # d02 spans 39.0 to 39.01 by -121.0 to -120.99: it holds the centre of one d01 cell, (39.005, -120.995).
    fine = sum(_cell_area(value) / 4 * 1000.0 for value in lat2.ravel())
    coarse = sum(_cell_area(value) * 100.0 for value in lat.ravel()) - _cell_area(39.005) * 100.0
    np.testing.assert_allclose(heat, np.full(3, (fine + coarse) / 1e6), rtol=1e-6)


def test_without_a_fire_grid_flux_the_atmosphere_s_is_used(case):
    run = _one_domain(case / "run", GRNHFX=1000.0)

    result = _run(_config(case, run, frp={"enabled": True}))

    assert result.exit_code == 0, result.output
    with h5py.File(case / "output" / "model.h5", "r") as out_h5:
        group = out_h5[FRP_GROUP]
        assert group.attrs["heat_flux_variables"] == "GRNHFX"
        area = 2 * _cell_area(39.0) + 2 * _cell_area(39.01)
        np.testing.assert_allclose(
            group["heat_release_rate_sensible"][:], np.full(3, area * 1e-3), rtol=1e-6
        )


def test_a_run_without_heat_flux_skips_the_frp(case):
    run = _one_domain(case / "run", FIRE_AREA=_burnt(), CAN_FRAC_BURNT=0.3)

    result = _run(
        _config(case, run, frp={"enabled": True}, ravg_cc={"enabled": True, "group_path": CC_GROUP})
    )

    assert result.exit_code == 0, result.output
    assert "Section frp skipped: 'neither FGRNHFX nor GRNHFX is in the output of d01'" in result.output
    with h5py.File(case / "output" / "model.h5", "r") as out_h5:
        assert FRP_GROUP not in out_h5
        assert CC_PATH in out_h5


def test_a_template_without_a_frame_needs_one_in_the_config(tmp_path):
    _write_template(tmp_path / "template.h5", frame=False)
    run = _one_domain(tmp_path / "run", FGRNHFX=1000.0)

    result = _run(_config(tmp_path, run, frp={"enabled": True}))

    assert "Section frp skipped" in result.output and "give frp.frame" in result.output


def test_a_conversion_factor_upside_down_stops_before_writing(case):
    run = _one_domain(case / "run", FGRNHFX=1000.0)

    result = _run(_config(case, run, frp={"enabled": True, "conversion_factor": [10, 5]}))

    assert result.exit_code != 0
    assert "frp.conversion_factor must be [smallest, largest]" in result.output
    assert not (case / "output" / "model.h5").exists()


def test_canopy_cover_loss_takes_the_canopy_burnt_over_the_run(case):
    # CFB is 0.8 at 00:30 only, after the front, 0; CAN_FRAC_BURNT, its running maximum, says 0.3.
    run = _one_domain(
        case / "run", FIRE_AREA=_burnt(), CFB=lambda t: 0.8 if t == 30.0 else 0.0, CAN_FRAC_BURNT=0.3
    )

    result = _run(_config(case, run, ravg_cc={"enabled": True, "group_path": CC_GROUP}))

    assert result.exit_code == 0, result.output
    with h5py.File(case / "output" / "model.h5", "r") as out_h5:
        # Unburnt (9), 25-<50 % (3), and outside the fire grid: no data (0).
        np.testing.assert_array_equal(out_h5[CC_PATH][:], [[9, 3, 0]])
        assert out_h5[CC_PATH].attrs["_FillValue"] == 0


def test_without_can_frac_burnt_the_largest_cfb_is_used(case):
    run = _one_domain(case / "run", FIRE_AREA=_burnt(), CFB=lambda t: 0.8 if t == 30.0 else 0.0)

    result = _run(_config(case, run, ravg_cc={"enabled": True, "group_path": CC_GROUP}))

    assert result.exit_code == 0, result.output
    with h5py.File(case / "output" / "model.h5", "r") as out_h5:
        np.testing.assert_array_equal(out_h5[CC_PATH][:], [[9, 5, 0]])


def test_without_canopy_burnt_the_canopy_cover_loss_is_skipped(case):
    run = _one_domain(case / "run", FIRE_AREA=_burnt())

    result = _run(_config(case, run, ravg_cc={"enabled": True, "group_path": CC_GROUP}))

    assert "Section ravg_cc skipped" in result.output and "neither CAN_FRAC_BURNT nor CFB" in result.output
    with h5py.File(case / "output" / "model.h5", "r") as out_h5:
        assert CC_PATH not in out_h5
