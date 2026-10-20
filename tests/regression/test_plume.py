"""The plume of a run at the radar's times and places, from a FireBench template."""

import h5py
import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from firebench.benchmarks.fire_case_definition import write_case_definition
from firebench.benchmarks.fire_case_radar_model import (
    FrameTarget,
    PlumeTargets,
    SectionTarget,
    write_target_groups,
)
from firebench.standardize import new_std_file
from netCDF4 import Dataset

from firebench_adapter_wrf_sfire.cli import main
from firebench_adapter_wrf_sfire.plume import bracket

GROUP = "/spatial_3d/plume"
SECTION = f"{GROUP}/sections/KTST/axis"
FRAME = f"{GROUP}/frame/KTST"
ORIGIN = "2026-01-01T00:00:00+00:00"
XTIME = [0.0, 30.0, 60.0]
GRAVITY = 9.81
# Seven staggered levels 500 m apart: mass levels at 250, 750, ..., 2750 m.
Z_STAG = 500.0 * np.arange(7)
ALPHA = np.radians(30.0)
# The section runs out of the fine domain (latitude 39.02 and up) into the coarse one, and so does the
# frame. The fine domain's tracer is ten times the coarse one's, so each value tells its domain.
SECTION_LAT = np.array([[39.0, 39.01, 39.03, 39.045]] * 3)
FINE = SECTION_LAT >= 39.02
FINE_SCALE = 10.0
SECTION_LON = np.full((3, 4), -120.97)
SECTION_HEIGHT = np.array([[400.0] * 4, [1200.0] * 4, [2000.0] * 4])
FRAME_BOX = (-120.995, 38.995, -120.955, 39.035)


def _when(minutes):
    return np.datetime64("2026-01-01T00:00", "ms") + np.array(minutes, dtype="timedelta64[m]")


def _write_wrfout(path, lat0, lon0, n, step, tracers=("tr17_1", "tr17_2", "tr17_3"), scale=1.0):
    """A domain whose tracer tr17_k is scale k (t + z / 1000), t the output index, with a wind of 10 m/s
    along the grid's x axis turned 30 degrees, and w the number of the level."""
    lat = lat0 + step * np.arange(n)[:, None] * np.ones((1, n))
    lon = lon0 + step * np.arange(n)[None, :] * np.ones((n, 1))
    n_z = len(Z_STAG) - 1
    with Dataset(path, "w") as wrf:
        wrf.SIMULATION_START_DATE = "2026-01-01_00:00:00"
        for name, size in (
            ("Time", len(XTIME)),
            ("bottom_top", n_z),
            ("bottom_top_stag", n_z + 1),
            ("south_north", n),
            ("south_north_stag", n + 1),
            ("west_east", n),
            ("west_east_stag", n + 1),
        ):
            wrf.createDimension(name, size)
        wrf.createVariable("XTIME", "f8", ("Time",))[:] = XTIME
        flat = ("Time", "south_north", "west_east")
        for name, value in (
            ("XLAT", lat),
            ("XLONG", lon),
            ("COSALPHA", np.cos(ALPHA)),
            ("SINALPHA", np.sin(ALPHA)),
        ):
            wrf.createVariable(name, "f8", flat)[:] = np.broadcast_to(value, (len(XTIME), n, n))
        stag = ("Time", "bottom_top_stag", "south_north", "west_east")
        wrf.createVariable("PHB", "f8", stag)[:] = np.broadcast_to(
            (Z_STAG * GRAVITY)[None, :, None, None], (len(XTIME), n_z + 1, n, n)
        )
        wrf.createVariable("PH", "f8", stag)[:] = 0.0
        wrf.createVariable("W", "f8", stag)[:] = np.broadcast_to(
            np.arange(n_z + 1, dtype=float)[None, :, None, None], (len(XTIME), n_z + 1, n, n)
        )
        wrf.createVariable("U", "f8", ("Time", "bottom_top", "south_north", "west_east_stag"))[:] = 10.0
        wrf.createVariable("V", "f8", ("Time", "bottom_top", "south_north_stag", "west_east"))[:] = 0.0
        heights = 0.5 * (Z_STAG[:-1] + Z_STAG[1:])
        for number, name in enumerate(tracers, start=1):
            values = scale * number * (np.arange(len(XTIME))[:, None] + heights[None, :] / 1000.0)
            variable = wrf.createVariable(name, "f4", ("Time", "bottom_top", "south_north", "west_east"))
            variable[:] = np.broadcast_to(values[:, :, None, None], (len(XTIME), n_z, n, n))
            variable.units = "ug kg-1"


@pytest.fixture
def case(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_wrfout(run / "wrfout_d01_2026-01-01_00:00:00", 38.99, -121.01, 8, 0.01)
    _write_wrfout(run / "wrfout_d02_2026-01-01_00:00:00", 39.02, -120.98, 8, 0.005, scale=FINE_SCALE)
    template = tmp_path / "template.h5"
    targets = PlumeTargets(
        model_group=GROUP,
        sections=[
            SectionTarget(
                section="KTST/axis",
                path=SECTION,
                time=_when([15, 30, 90]),
                time_origin=ORIGIN,
                latitude=SECTION_LAT,
                longitude=SECTION_LON,
                height=SECTION_HEIGHT,
            )
        ],
        frames=[
            FrameTarget(
                site="KTST",
                path=FRAME,
                time=_when([15]),
                time_origin=ORIGIN,
                frame=FRAME_BOX,
                height_bins=np.arange(0.0, 3100.0, 100.0),
                threshold=0.01,
            )
        ],
    )
    with new_std_file(str(template), "test") as out:
        out.attrs["firebench_template"] = "model-output"
        write_target_groups(out, targets, "test")
    write_case_definition(
        template,
        {
            "schema": "fb-case-v1",
            "families": [
                {
                    "family": "radar_plume",
                    "periods": ["P01"],
                    "sections": ["KTST/axis"],
                    "model_group": GROUP,
                    "echo_top": ["KTST"],
                }
            ],
        },
    )
    return tmp_path


def _config(tmp_path, **plume):
    config = {
        "main": {
            "template_path": "template.h5",
            "wrf_path": "run",
            "output_path": "output/model.h5",
            "authors": "Test Author",
            "compression_lvl": 1,
        },
        "plume": {"enabled": True, **plume},
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run(config_path):
    return CliRunner().invoke(main, ["run", str(config_path)])


def _domains(values):
    """The value of each node from the domain that gives it: the fine one where it covers the node."""
    return np.where(FINE, FINE_SCALE * values, values)


def test_the_tracer_and_the_wind_are_sampled_on_the_section_between_two_outputs(case):
    result = _run(_config(case))
    assert result.exit_code == 0, result.output
    assert "KTST/axis: 2 time(s)" in result.output
    with h5py.File(case / "output" / "model.h5", "r") as h5:
        group = h5[SECTION]
        assert group.attrs["source_variable"] == "tr17_1"
        assert group.attrs["domain"] == "d02, d01"
        assert group["passive_tracer"].attrs["units"] == "ug kg-1"
        tracer = group["passive_tracer"][()]
        # 15 min: halfway between outputs 0 and 1; 30 min: output 1; 90 min: after the run. Each node
        # comes from the finest domain that covers it.
        assert np.allclose(tracer[0], _domains(0.5 + SECTION_HEIGHT / 1000.0), atol=1e-4)
        assert np.allclose(tracer[1], _domains(1.0 + SECTION_HEIGHT / 1000.0), atol=1e-4)
        assert np.isnan(tracer[2]).all()
        assert np.allclose(group["wind_u"][0], 10.0 * np.cos(ALPHA), atol=1e-5)
        assert np.allclose(group["wind_v"][0], 10.0 * np.sin(ALPHA), atol=1e-5)
        # W on the staggered levels is their number: on the mass levels, z / 500.
        assert np.allclose(group["wind_w"][0], SECTION_HEIGHT / 500.0, atol=1e-5)
        assert np.allclose(group["position_alt"][()], SECTION_HEIGHT)


def test_the_tracer_is_the_one_chosen(case):
    assert _run(_config(case, tracer="tr17_2")).exit_code == 0
    with h5py.File(case / "output" / "model.h5", "r") as h5:
        assert h5[SECTION].attrs["source_variable"] == "tr17_2"
        assert np.allclose(
            h5[SECTION]["passive_tracer"][1], _domains(2.0 * (1.0 + SECTION_HEIGHT / 1000.0)), atol=1e-4
        )


def test_a_tracer_the_run_does_not_hold_is_skipped_with_the_ones_it_does(case):
    result = _run(_config(case, tracer="tr17_5"))
    assert result.exit_code == 0, result.output
    assert "tr17_5 is not in the output; the run has tr17_1, tr17_2, tr17_3" in result.output
    with h5py.File(case / "output" / "model.h5", "r") as h5:
        assert GROUP not in h5


def test_a_tracer_that_is_not_one_of_the_eight_is_refused(case):
    result = _run(_config(case, tracer="tr17_9"))
    assert result.exit_code == 2
    assert "plume.tracer must be one of tr17_1 to tr17_8" in result.output


def test_two_outputs_too_far_apart_leave_the_time_between_them_out(case):
    assert _run(_config(case, max_time_gap=600)).exit_code == 0
    with h5py.File(case / "output" / "model.h5", "r") as h5:
        tracer = h5[SECTION]["passive_tracer"][()]
    assert np.isnan(tracer[0]).all()
    assert np.isfinite(tracer[1]).all()


def test_the_frame_counts_the_plume_cells_of_every_domain_weighted_by_their_area(case):
    assert _run(_config(case)).exit_code == 0
    with h5py.File(case / "output" / "model.h5", "r") as h5:
        frame = h5[FRAME]
        assert frame.attrs["domain"] == "d02, d01"
        assert "cell area" in frame.attrs["cell_weights"]
        counts = frame["plume_cells"][()]
        assert frame["plume_cells"].attrs["tracer_relative_threshold"] == 0.01
    # The frame holds 4 x 4 columns of the coarse grid, 6 of them inside the fine grid: 10 coarse
    # columns count, and the 3 x 6 fine columns inside the frame, a quarter as large, count a quarter
    # each. Each column has six levels whose tracer is above a hundredth of the largest.
    columns = 10 + 18 / 4
    assert counts.shape == (1, 30)
    assert counts.sum() == pytest.approx(columns * 6, rel=0.02)
    assert counts[0, 2] == pytest.approx(columns, rel=0.02) and counts[0, 7] == pytest.approx(
        columns, rel=0.02
    )


def test_one_domain_counts_whole_cells(case):
    assert _run(_config(case, domain="d01")).exit_code == 0
    with h5py.File(case / "output" / "model.h5", "r") as h5:
        frame = h5[FRAME]
        assert frame.attrs["domain"] == "d01" and "cell_weights" not in frame.attrs
        assert frame["plume_cells"][()].sum() == 16 * 6


def test_the_points_of_the_plume_are_written_for_figures_unless_left_out(case):
    assert _run(_config(case, points={"every_s": 60, "max_per_time": 10})).exit_code == 0
    with h5py.File(case / "output" / "model.h5", "r") as h5:
        points = h5[f"{GROUP}/points"]
        # Every 60 s of the target times the run gives (15 and 30 min), over the frame.
        assert points["count"][()].tolist() == [10, 10]
        assert len(points["position_z"]) == 20
    result = CliRunner().invoke(main, ["run", "-o", str(_config(case, points=False))])
    assert result.exit_code == 0, result.output
    with h5py.File(case / "output" / "model.h5", "r") as h5:
        assert f"{GROUP}/points" not in h5


def test_a_named_domain_takes_every_target(case):
    assert _run(_config(case, domain="d01")).exit_code == 0
    with h5py.File(case / "output" / "model.h5", "r") as h5:
        assert h5[SECTION].attrs["domain"] == "d01"
        assert np.allclose(h5[SECTION]["passive_tracer"][1], 1.0 + SECTION_HEIGHT / 1000.0, atol=1e-5)


def test_firebench_reads_what_the_adapter_wrote(case):
    from firebench.benchmarks.fire_case_radar_model import ModelSection, model_frame

    assert _run(_config(case)).exit_code == 0
    with h5py.File(case / "output" / "model.h5", "r") as h5:
        section = ModelSection(h5, GROUP, "KTST/axis", SECTION_LAT.shape)
        assert section.nearest(_when([30])[0], 300) == 1
        times, edges, counts, threshold = model_frame(h5, GROUP, "KTST")
        assert times.tolist() == _when([15]).tolist() and threshold == 0.01 and len(edges) == 31


def test_a_time_takes_the_outputs_around_it():
    xtime = np.array(XTIME)
    assert bracket(xtime, 15.0, 1800.0) == (0, 1, 0.5)
    assert bracket(xtime, 30.0, 1800.0) == (1, 1, 0.0)
    assert bracket(xtime, -1.0, 1800.0) is None
    assert bracket(xtime, 61.0, 1800.0) is None
    assert bracket(xtime, 15.0, 600.0) is None
