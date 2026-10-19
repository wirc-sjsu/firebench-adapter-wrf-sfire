"""The fire radiative power of a WRF-SFIRE run, as a band derived from its sensible heat flux."""

from __future__ import annotations

from functools import reduce

import hdf5plugin
import h5py
import numpy as np
from netCDF4 import Dataset

from ._grid import fire_grid_slice
from ._wrfout import WrfRun, as_run

DEFAULT_GROUP_PATH = "/time_series/fire_total"
# The sensible heat flux is taken to be 5 to 10 times the FRP: FRP = heat / k, k in [5, 10].
DEFAULT_CONVERSION_FACTOR = (5.0, 10.0)
# Heat fluxes of the ground and the crown fire, in W m-2: on the fire grid, else on the atmosphere grid.
FIRE_GRID_FLUXES = ("FGRNHFX", "FCANHFX")
ATMOSPHERE_FLUXES = ("GRNHFX", "CANHFX")
FRAME_KEYS = ("frame_west", "frame_south", "frame_east", "frame_north")
EARTH_RADIUS = 6_371_000.0
LOWER = "fire_radiative_power_lower"
UPPER = "fire_radiative_power_upper"
HEAT = "heat_release_rate_sensible"


def cell_area(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """The area of each cell of a grid of centres in degrees, in m2, from the spacing of its neighbours."""
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    x, y = EARTH_RADIUS * np.cos(lat_rad) * lon_rad, EARTH_RADIUS * lat_rad
    if min(lat.shape) < 2:
        raise ValueError("a grid needs two cells in each direction to give the area of its cells")
    dx_i, dx_j = np.gradient(x, axis=1), np.gradient(x, axis=0)
    dy_i, dy_j = np.gradient(y, axis=1), np.gradient(y, axis=0)
    return np.abs(dx_i * dy_j - dx_j * dy_i)


def _frame(ref_h5: h5py.File, group_path: str, frame) -> tuple[float, float, float, float]:
    if frame is None:
        group = ref_h5.get(group_path)
        if group is None or not all(key in group.attrs for key in FRAME_KEYS):
            raise ValueError(
                f"the reference file has no frame at {group_path} ({', '.join(FRAME_KEYS)}); "
                "give frp.frame as [west, south, east, north]"
            )
        frame = [float(group.attrs[key]) for key in FRAME_KEYS]
    west, south, east, north = (float(value) for value in frame)
    if not (west < east and south < north):
        raise ValueError(f"the frame {frame} is not [west, south, east, north]")
    return west, south, east, north


def get_fire_total(  # pylint: disable=too-many-arguments,too-many-locals
    new_h5: h5py.File,
    ref_h5: h5py.File,
    group_path: str,
    wrf_dataset: WrfRun | Dataset,
    ref_time: str,
    conversion_factor=DEFAULT_CONVERSION_FACTOR,
    frame=None,
    fire_grid_bound: int = 25,
    compression_lvl: int = 3,
):
    """Write the FRP band of the run over a frame at every output time, in ``group_path``.

    The sensible heat released over the frame is the heat flux of the ground and of the crown fire
    (FGRNHFX + FCANHFX, else GRNHFX + CANHFX on the atmosphere grid) times the area of each cell inside
    the frame, a cell taken from the finest fire domain that holds it. The FRP band is that heat
    divided by the largest and by the smallest ``conversion_factor``. The frame is the reference's
    (``frame_west`` ... attributes of ``group_path`` in a benchmark kit's template) unless given.
    ``ref_time`` is the time XTIME counts from.
    """
    k_min, k_max = (float(value) for value in conversion_factor)
    if not 0 < k_min <= k_max:
        raise ValueError(f"conversion_factor must be [smallest, largest], above 0; got {conversion_factor}")
    west, south, east, north = _frame(ref_h5, group_path, frame)
    domains = as_run(wrf_dataset).fire_domains()
    if not domains:
        raise ValueError("no domain of the run has a fire grid")
    grid_slice = fire_grid_slice(fire_grid_bound)

    plan = []
    for index, domain in enumerate(domains):
        names = [name for name in FIRE_GRID_FLUXES if domain.has(name)]
        fire = bool(names) and "FGRNHFX" in names
        if not fire:
            names = [name for name in ATMOSPHERE_FLUXES if domain.has(name)]
        if not names:
            raise KeyError(f"neither FGRNHFX nor GRNHFX is in the output of {domain.name}")
        # The atmosphere grid has no padding to trim.
        domain_slice = grid_slice if fire else slice(None)
        lat_name, lon_name = ("FXLAT", "FXLONG") if fire else ("XLAT", "XLONG")
        lat, lon = domain.static(lat_name, domain_slice), domain.static(lon_name, domain_slice)
        inside = (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
        for finer in domains[:index]:
            inside &= ~finer.covers(lat, lon, grid_slice, fire=True)
        plan.append((domain, names, domain_slice, inside, cell_area(lat, lon)))
    if not any(inside.any() for _, _, _, inside, _ in plan):
        raise ValueError(
            f"no cell of the run's fire domains is inside the frame {[west, south, east, north]}"
        )

    times = reduce(np.intersect1d, [domain.xtime() for domain in domains])
    heat = np.zeros(times.size)
    for domain, names, domain_slice, inside, area in plan:
        weights = area[inside]
        for position, index in enumerate(np.searchsorted(domain.xtime(), times)):
            for name in names:
                heat[position] += np.sum(domain.at(name, int(index), domain_slice)[inside] * weights)
    heat_mw = heat / 1e6
    variables = ",".join(sorted({name for _, names, _, _, _ in plan for name in names}))

    group = new_h5.require_group(group_path)
    for key, value in zip(FRAME_KEYS, (west, south, east, north)):
        group.attrs[key] = value
        group.attrs[f"{key}_units"] = "degree"
    group.attrs["frp_conversion"] = f"{HEAT} / k, k from conversion_factor_min to conversion_factor_max"
    group.attrs["conversion_factor_min"] = k_min
    group.attrs["conversion_factor_max"] = k_max
    group.attrs["heat_flux_variables"] = variables
    zstd = hdf5plugin.Zstd(clevel=compression_lvl)
    time = group.create_dataset("time", data=times * 60.0, **zstd)
    time.attrs["time_origin"] = ref_time
    time.attrs["time_units"] = "s"
    for name, values, description in (
        (HEAT, heat_mw, f"Sensible heat released over the frame ({variables} times the cell areas)"),
        (LOWER, heat_mw / k_max, f"{HEAT} / {k_max:g}"),
        (UPPER, heat_mw / k_min, f"{HEAT} / {k_min:g}"),
    ):
        dataset = group.create_dataset(name, data=values, **zstd)
        dataset.attrs["units"] = "MW"
        dataset.attrs["description"] = description
