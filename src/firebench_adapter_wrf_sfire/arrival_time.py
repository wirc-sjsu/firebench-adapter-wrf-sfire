from datetime import datetime, timezone

import hdf5plugin
import h5py
import numpy as np
from netCDF4 import Dataset
import firebench.standardize as fs

from ._grid import fire_grid_slice
from ._skip import skip
from ._wrfout import WrfRun, as_run


def get_arrival_time_from_t_ign_g(
    new_h5: h5py.File,
    wrf_dataset: WrfRun | Dataset,
    group_path: str,
    t_ign_g_ref_time: str | None,
    fire_grid_bound: int = 25,
    compression_lvl: int = 3,
    period: tuple[datetime, datetime] | None = None,
):
    """Write the fire's arrival time on the run's own fire grid, for FireBench's forecast windows.

    The field is ``TIGN_G`` at the last output time of the run, which holds the whole history of the
    fire, on the finest fire grid (as the perimeters), trimmed by ``fire_grid_bound``. It counts
    seconds after ``t_ign_g_ref_time`` (the run's ``SIMULATION_START_DATE`` by default). A cell the
    fire had not reached at that time, which WRF-SFIRE marks with a time beyond the run, is NaN.
    ``time`` holds the output times of the run (XTIME, in seconds after the same reference).

    ``group_path`` gets ``position_lat``, ``position_lon``, ``fire_arrival_time``, and ``time``. When
    ``period`` (start, end) is given and the output times do not cover it, the field is written all
    the same and a ``SkippedWarning`` says so.
    """
    run = as_run(wrf_dataset)
    domains = run.fire_domains()
    if not domains:
        raise ValueError("no domain of the run has a fire grid")
    domain = domains[0]
    if t_ign_g_ref_time is None:
        t_ign_g_ref_time = run.start_time()
    origin = datetime.fromisoformat(t_ign_g_ref_time.replace("Z", "+00:00"))
    if origin.tzinfo is None:
        raise ValueError("t_ign_g_ref_time must include a timezone offset")
    origin = origin.astimezone(timezone.utc)
    grid_slice = fire_grid_slice(fire_grid_bound)

    output_times_s = np.asarray(domain.xtime(), dtype=np.float64) * 60
    if not output_times_s.size:
        raise ValueError(f"{domain.name} has no output time")
    horizon_s = float(output_times_s[-1])
    t_ign_g = np.ma.masked_invalid(domain.last("TIGN_G", grid_slice)).astype(np.float64).filled(np.nan)
    arrival = np.where(t_ign_g <= horizon_s, t_ign_g, np.nan)
    if not np.isfinite(arrival).any():
        raise ValueError(
            f"TIGN_G holds no cell burnt by the last output time ({horizon_s} s) in {domain.name}"
        )

    group = new_h5.create_group(group_path)
    group.attrs["crs"] = "EPSG:4326"
    for name, values in (
        ("position_lat", domain.static("FXLAT", grid_slice)),
        ("position_lon", domain.static("FXLONG", grid_slice)),
    ):
        group.create_dataset(name, data=values, **hdf5plugin.Zstd(clevel=compression_lvl)).attrs[
            "units"
        ] = "degree"
    field = group.create_dataset(
        "fire_arrival_time", data=arrival, **hdf5plugin.Zstd(clevel=compression_lvl)
    )
    field.attrs["units"] = "s"
    field.attrs["time_origin"] = fs.datetime_to_iso8601(origin)
    time = group.create_dataset("time", data=output_times_s)
    time.attrs["time_origin"] = fs.datetime_to_iso8601(origin)
    time.attrs["time_units"] = "s"

    if period is not None:
        start_s, end_s = ((moment.astimezone(timezone.utc) - origin).total_seconds() for moment in period)
        if output_times_s[0] > start_s or horizon_s < end_s:
            skip(
                f"The part of {group_path} outside the run's output",
                f"{domain.name} covers {output_times_s[0]:g} s to {horizon_s:g} s after "
                f"{t_ign_g_ref_time}, the period is {start_s:g} s to {end_s:g} s",
            )
