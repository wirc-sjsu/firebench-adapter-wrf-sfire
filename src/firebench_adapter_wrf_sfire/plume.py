"""The plume of a WRF-SFIRE run at the radar's times and places: one of its fire tracers and its wind.

A FireBench case with the ``radar_plume`` family scores the model on the radar's cross sections, at
the start of each radar volume, and, for the sites of ``echo_top``, by its plume cells over the site's
frame. The case gives these targets (``plume_targets``, from the template of a benchmark kit or an
observation file); the run is read where it is and only its values on the targets are written, never a
3D field.

Every domain that holds the tracer gives the targets, so the plume is followed past the nests: each
position (a node of a section, a column of a frame) takes the finest domain that holds the tracer and
covers it. Each domain is read cut to the cells around the positions it gives. The wind is destaggered
to the mass points and turned to east and north with COSALPHA and SINALPHA; heights are those of the
mass levels, from PH + PHB. The tracer, a mass per kg of dry air (WRF-SFIRE's ``tr17_*`` are in ug/kg),
is turned into a mass concentration in ug/m^3, the units of FireBench's plume threshold, by the dry-air
density of each cell (``dry_air_density``: WRF's inverse of ALT, from P + PB, T + 300 K, and QVAPOR)
before it is interpolated. A target time between two output times of a domain takes both, linearly in
time: on a section, the samples of both outputs on its nodes (FireBench's ``ColumnSampler``, the nearest
column then linear in height); over a frame, the tracer and heights of both outputs, whose plume cells
FireBench counts over every domain at once at the case's threshold (``plume_cell_counts``), each cell
weighted by its area when several domains give the frame, and puts the highest of each column on the
radar's map grid (``column_tops``, ``tops_on_map``). Over a frame, only the cells the radar can see
count: between the lowest and the highest beam over their map cell at that time (``beam_band_at``,
``visible_cells``). A target time outside a domain's output, or between two of its outputs more than
``max_time_gap`` s apart, is NaN where that domain gives the position; a frame needs every domain that
gives it. ``inspect_plume`` says what a run gives, reading no 3D field.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import h5py
import numpy as np
from firebench.benchmarks.fire_case_parameters import CONCENTRATION_UNITS, load_parameters
from firebench.benchmarks.fire_case_radar_model import (
    FIELDS,
    TRACER,
    ColumnSampler,
    FrameTarget,
    SectionTarget,
    beam_band_at,
    column_tops,
    plume_cell_counts,
    plume_cell_mask,
    tops_on_map,
    visible_cells,
    write_plume_frame,
    write_plume_points,
    write_plume_section,
)
from firebench.benchmarks.fire_case_template import plume_targets
from firebench.tools.units import ureg
from netCDF4 import Dataset
from scipy.spatial import cKDTree

from ._wrfout import WrfDomain, WrfRun, as_run
from .frp import cell_area

TRACER_NAME = re.compile(r"^tr17_[1-8]$")
DEFAULT_TRACER = "tr17_1"
# The units of WRF-SFIRE's fire tracers, which its netCDF output does not state.
DEFAULT_TRACER_UNITS = "ug kg-1"
DEFAULT_MAX_TIME_GAP = 1800.0
DEFAULT_POINTS_EVERY = 900.0
DEFAULT_MAX_POINTS = 20_000
GRAVITY = 9.81
# WRF's constants (share/module_model_constants.F): gas constants of dry air and water vapour, heat
# capacity of dry air at constant pressure, reference pressure, and base potential temperature.
R_D = 287.0
R_V = 461.6
CP = 7.0 * R_D / 2.0
P0 = 100000.0
T0 = 300.0
# What a domain must hold besides the tracer.
ATMOSPHERE = (
    "XLAT",
    "XLONG",
    "PH",
    "PHB",
    "P",
    "PB",
    "T",
    "QVAPOR",
    "U",
    "V",
    "W",
    "COSALPHA",
    "SINALPHA",
)
# Units that give no scale: a tracer in them has no concentration.
NO_UNITS = ("", "1", "-", "dimensionless")
TRACER_CONVERSION = (
    "the tracer (a mass per kg of dry air) times the dry-air density of its cell, "
    "p / (R_d T (1 + (R_v / R_d) qv)), with p = P + PB, T = (T + 300 K) (p / 1000 hPa)^(R_d / c_p), and "
    "qv = QVAPOR (WRF's inverse of ALT), at each output time before the interpolation in time"
)
# Cells kept around the targets, so a node near the edge of the cut still finds its column.
MARGIN = 2
INTERPOLATION = (
    "each position from the finest domain that holds the tracer and covers it; linear in time between "
    "the two output times of that domain around each target time, at most max_time_gap s apart; on a "
    "section, FireBench's ColumnSampler (nearest column, then linear in height) at each"
)


@dataclass
class _Plan:  # pylint: disable=too-many-instance-attributes
    """What one domain gives, and the cells of the domain around it."""

    domain: WrfDomain
    # The nodes of each section this domain gives, as indices into the section's flattened nodes.
    nodes: dict[str, np.ndarray] = field(default_factory=dict)
    # The columns of each frame this domain gives: on its full grid, then on the cut.
    columns: dict[str, np.ndarray] = field(default_factory=dict)
    # The columns of the cut no finer domain covers, for the plume cells of figures.
    own: np.ndarray | None = None
    rows: slice = slice(None)
    cols: slice = slice(None)
    lat: np.ndarray | None = None
    lon: np.ndarray | None = None
    cos: np.ndarray | None = None
    sin: np.ndarray | None = None
    area: np.ndarray | None = None


def _utc(text: str) -> np.datetime64:
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError(f"the time {text} has no UTC offset")
    return np.datetime64(moment.astimezone(timezone.utc).replace(tzinfo=None), "ms")


def _tracers(run: WrfRun) -> list[str]:
    return sorted(
        {name for domain in run.domains for name in domain.datasets[0].variables if TRACER_NAME.match(name)}
    )


def _candidates(run: WrfRun, tracer: str, domain: str | None) -> list[WrfDomain]:
    """The domains that give the targets, finest first: the one named, or every one holding the tracer."""
    if domain is not None:
        named = [item for item in run.domains if item.name == domain]
        if not named:
            raise ValueError(
                f"the run has no domain {domain}; it has {', '.join(item.name for item in run.domains)}"
            )
        candidates = named
    else:
        candidates = [item for item in run.domains if item.has(tracer)]
    if not any(item.has(tracer) for item in candidates):
        present = _tracers(run)
        raise KeyError(
            f"{tracer} is not in the output{f' of {domain}' if domain else ''}; "
            f"the run has {', '.join(present) if present else 'no tr17 tracer'}"
        )
    for item in candidates:
        missing = [name for name in ATMOSPHERE if not item.has(name)]
        if missing:
            raise KeyError(f"{', '.join(missing)} not in the output of {item.name}")
    return candidates


def _owners(candidates: list[WrfDomain], lat, lon) -> np.ndarray:
    """For each point, the index of the finest candidate that covers it; -1 when none does."""
    owners = np.full(np.shape(lat), -1, dtype=int)
    for index, item in enumerate(candidates):
        mine = (owners < 0) & item.covers(lat, lon, slice(None))
        owners[mine] = index
    return owners


def _assign(
    candidates: list[WrfDomain], sections: list[SectionTarget], frames: list[FrameTarget]
) -> list[_Plan]:
    """What each candidate gives: the nodes and frame columns no finer candidate covers."""
    plans = [_Plan(item) for item in candidates]
    for target in sections:
        owners = _owners(candidates, np.ravel(target.latitude), np.ravel(target.longitude))
        if (owners < 0).all():
            raise ValueError(f"no domain holding the tracer covers the cross section {target.section}")
        for index, plan in enumerate(plans):
            if (owners == index).any():
                plan.nodes[target.section] = np.flatnonzero(owners == index)
    full = slice(None)
    for target in frames:
        west, south, east, north = target.frame
        found = False
        for index, plan in enumerate(plans):
            xlat, xlon = plan.domain.box("XLAT", full, full), plan.domain.box("XLONG", full, full)
            mine = (xlon >= west) & (xlon <= east) & (xlat >= south) & (xlat <= north)
            for finer in candidates[:index]:
                mine &= ~finer.covers(xlat, xlon, full)
            if mine.any():
                plan.columns[target.site] = mine
                found = True
        if not found:
            raise ValueError(f"no domain holding the tracer covers the frame of {target.site}")
    return [plan for plan in plans if plan.nodes or plan.columns]


def _cut(plan: _Plan, sections: list[SectionTarget], candidates: list[WrfDomain]) -> None:
    """The cells of the domain around what it gives: inside their box, or nearest one of its nodes."""
    full = slice(None)
    xlat, xlon = plan.domain.box("XLAT", full, full), plan.domain.box("XLONG", full, full)
    lats, lons = [], []
    for target in sections:
        if target.section in plan.nodes:
            lats.append(np.ravel(target.latitude)[plan.nodes[target.section]])
            lons.append(np.ravel(target.longitude)[plan.nodes[target.section]])
    for mine in plan.columns.values():
        lats.append(xlat[mine])
        lons.append(xlon[mine])
    lat, lon = np.concatenate(lats), np.concatenate(lons)
    keep = (xlat >= lat.min()) & (xlat <= lat.max()) & (xlon >= lon.min()) & (xlon <= lon.max())
    scale = np.cos(np.radians(float(np.mean(lat))))
    tree = cKDTree(np.column_stack([xlon.ravel() * scale, xlat.ravel()]))
    _, nearest = tree.query(np.column_stack([lon * scale, lat]))
    keep.ravel()[nearest] = True
    rows, cols = np.nonzero(keep)
    ny, nx = xlat.shape
    plan.rows = slice(max(int(rows.min()) - MARGIN, 0), min(int(rows.max()) + MARGIN + 1, ny))
    plan.cols = slice(max(int(cols.min()) - MARGIN, 0), min(int(cols.max()) + MARGIN + 1, nx))
    plan.lat, plan.lon = xlat[plan.rows, plan.cols], xlon[plan.rows, plan.cols]
    plan.cos = plan.domain.box("COSALPHA", plan.rows, plan.cols)
    plan.sin = plan.domain.box("SINALPHA", plan.rows, plan.cols)
    plan.columns = {site: mine[plan.rows, plan.cols] for site, mine in plan.columns.items()}
    own = np.ones(plan.lat.shape, dtype=bool)
    for finer in candidates[: candidates.index(plan.domain)]:
        own &= ~finer.covers(plan.lat, plan.lon, full)
    plan.own = own
    plan.area = cell_area(plan.lat, plan.lon) if min(plan.lat.shape) >= 2 else np.ones(plan.lat.shape)


def mixing_ratio_factor(units: str) -> float:
    """
    What a tracer in ``units``, a mass per mass of dry air (``ug kg-1``, ``ug/kg``, ``g/kg``), is
    multiplied by to be in ug/kg.

    Raises
    ------
    ValueError
        If ``units`` are not a mass per mass: a tracer without units, or on any scale, has no
        concentration to compare with FireBench's threshold.
    """
    text = units.strip()
    # netCDF's exponents (kg-1) as Pint's (kg^-1).
    pint_text = re.sub(r"(?<=[A-Za-z])(-?\d+)\b", r"^\1", text)
    try:
        quantity = ureg.Quantity(1.0, pint_text)
    # Pint's errors are AttributeError (an unknown unit), TypeError, or ValueError (a syntax).
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"the tracer's units {units!r} are not units") from error
    if text in NO_UNITS or not quantity.dimensionless:
        raise ValueError(
            f"the tracer's units {units!r} are not a mass per kg of dry air (such as ug kg-1); set "
            "plume.tracer_units when the output's are wrong"
        )
    return float(quantity.to("ug/kg").magnitude)


def dry_air_density(pressure, theta_perturbation, vapour) -> np.ndarray:
    """The density of dry air, kg/m^3, from WRF's full pressure (Pa), perturbation potential temperature
    (K, from 300 K), and water vapour mixing ratio (kg/kg): the inverse of WRF's ALT."""
    temperature = (theta_perturbation + T0) * (pressure / P0) ** (R_D / CP)
    return pressure / (R_D * temperature * (1.0 + (R_V / R_D) * vapour))


def read_fields(plan: _Plan, index: int, tracer: str, factor: float = 1.0) -> dict[str, np.ndarray]:
    """Heights (m above mean sea level), the tracer in ug/m^3 (the output's values times ``factor``,
    in ug/kg, times the dry-air density), and the earth-relative wind on the mass points of the cut at
    output time ``index``, each (bottom_top, rows, columns)."""
    domain, rows, cols = plan.domain, plan.rows, plan.cols
    density = dry_air_density(
        domain.volume("P", index, rows, cols) + domain.volume("PB", index, rows, cols),
        domain.volume("T", index, rows, cols),
        domain.volume("QVAPOR", index, rows, cols),
    )
    geopotential = domain.volume("PH", index, rows, cols) + domain.volume("PHB", index, rows, cols)
    u_stag = domain.volume("U", index, rows, slice(cols.start, cols.stop + 1))
    v_stag = domain.volume("V", index, slice(rows.start, rows.stop + 1), cols)
    w_stag = domain.volume("W", index, rows, cols)
    u = 0.5 * (u_stag[..., :-1] + u_stag[..., 1:])
    v = 0.5 * (v_stag[:, :-1, :] + v_stag[:, 1:, :])
    return {
        "z": 0.5 * (geopotential[:-1] + geopotential[1:]) / GRAVITY,
        TRACER: domain.volume(tracer, index, rows, cols).astype(np.float64) * factor * density,
        "wind_u": u * plan.cos - v * plan.sin,
        "wind_v": u * plan.sin + v * plan.cos,
        "wind_w": 0.5 * (w_stag[:-1] + w_stag[1:]),
    }


def bracket(xtime: np.ndarray, minutes: float, max_time_gap: float) -> tuple[int, int, float] | None:
    """The output times around a time (minutes of XTIME) and the weight of the later; None outside the
    run or between outputs more than ``max_time_gap`` s apart."""
    position = int(np.searchsorted(xtime, minutes, side="left"))
    if position < xtime.size and np.isclose(xtime[position], minutes, rtol=0.0, atol=1e-6):
        return position, position, 0.0
    if position in (0, xtime.size):
        return None
    low, high = position - 1, position
    if (xtime[high] - xtime[low]) * 60.0 > max_time_gap:
        return None
    return low, high, float((minutes - xtime[low]) / (xtime[high] - xtime[low]))


def _point_times(times: np.ndarray, every: float) -> set:
    kept, last = set(), None
    for time in times:
        if last is None or (time - last) / np.timedelta64(1, "s") >= every:
            kept.add(time)
            last = time
    return kept


def _thin(cells: np.ndarray, most: int) -> np.ndarray:
    if cells.size <= most:
        return cells
    return cells[np.linspace(0, cells.size - 1, most).astype(np.int64)]


class _Run:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """One domain's plan read output time after output time, each read once."""

    def __init__(
        self,
        plan: _Plan,
        sections: list[SectionTarget],
        tracer: str,
        origin: np.datetime64,
        max_time_gap: float,
        factor: float = 1.0,
    ):
        self.plan = plan
        self.tracer = tracer
        self.factor = factor
        self.origin = origin
        self.max_time_gap = max_time_gap
        self.xtime = plan.domain.xtime()
        self.fields: dict[int, dict[str, np.ndarray]] = {}
        self.samples: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        self.samplers: dict[str, ColumnSampler] = {}
        self.heights: dict[str, np.ndarray] = {}
        for target in sections:
            nodes = plan.nodes.get(target.section)
            if nodes is None:
                continue
            self.samplers[target.section] = ColumnSampler(
                plan.lat, plan.lon, np.ravel(target.latitude)[nodes], np.ravel(target.longitude)[nodes]
            )
            self.heights[target.section] = np.ravel(target.height)[nodes]

    def at(self, time: np.datetime64):
        """The bracket of ``time``, with the fields of its output times read; None when it has none."""
        minutes = (time - self.origin) / np.timedelta64(60_000, "ms")
        found = bracket(self.xtime, float(minutes), self.max_time_gap)
        if found is None:
            return None
        low, high, _ = found
        for index in [index for index in self.fields if index < low]:
            del self.fields[index]
            for key in [key for key in self.samples if key[1] == index]:
                del self.samples[key]
        for index in (low, high):
            if index not in self.fields:
                self.fields[index] = read_fields(self.plan, index, self.tracer, self.factor)
        return found

    def section(self, name: str, found) -> dict[str, np.ndarray]:
        """The fields on the nodes of a section this domain gives, at the time of ``found``."""
        low, high, weight = found
        parts = []
        for index in (low, high):
            key = (name, index)
            if key not in self.samples:
                values = self.fields[index]
                self.samples[key] = self.samplers[name].sample(
                    values["z"], {item: values[item] for item in FIELDS}, self.heights[name]
                )
            parts.append(self.samples[key])
        return {item: (1 - weight) * parts[0][item] + weight * parts[1][item] for item in FIELDS}

    def cube(self, found) -> tuple[np.ndarray, np.ndarray]:
        """Heights and tracer of the cut at the time of ``found``."""
        low, high, weight = found
        first, second = self.fields[low], self.fields[high]
        return (
            (1 - weight) * first["z"] + weight * second["z"],
            (1 - weight) * first[TRACER] + weight * second[TRACER],
        )


def _cells(parts: list[tuple[_Plan, tuple[np.ndarray, np.ndarray], np.ndarray]]):
    """Heights, tracer, whether each cell counts, its area, longitude, and latitude, over several
    domains' cuts, flattened into one list of cells."""
    columns = {name: [] for name in ("heights", "tracer", "inside", "area", "lon", "lat")}
    for plan, (z, values), counted in parts:
        shape = values.shape
        columns["heights"].append(z.ravel())
        columns["tracer"].append(values.ravel())
        for name, grid in (("inside", counted), ("area", plan.area), ("lon", plan.lon), ("lat", plan.lat)):
            columns[name].append(np.broadcast_to(grid, shape).ravel())
    return tuple(np.concatenate(items) for items in columns.values())


class _Writer:  # pylint: disable=too-few-public-methods
    """What the targets get, time after time, from every domain that gives them."""

    def __init__(self, targets, readers: list[_Run]):
        self.targets = targets
        self.readers = readers
        self.sections = {
            target.section: {
                name: np.full((len(target.time), target.latitude.size), np.nan) for name in FIELDS
            }
            for target in targets.sections
        }
        self.counts = {
            target.site: np.full((len(target.time), len(target.height_bins) - 1), np.nan)
            for target in targets.frames
        }
        self.tops = {
            target.site: np.full((len(target.time), len(target.latitude), len(target.longitude)), np.nan)
            for target in targets.frames
        }
        self.clouds: list[dict] = []
        self.cloud_times: list[np.datetime64] = []

    def at(self, time: np.datetime64, points: dict | None) -> None:
        """Fill in every target at ``time``, and keep the plume cells when ``points`` asks for them."""
        found = [reader.at(time) for reader in self.readers]
        cubes: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def cube(index: int):
            if index not in cubes:
                cubes[index] = self.readers[index].cube(found[index])
            return cubes[index]

        for target in self.targets.sections:
            for position in np.flatnonzero(target.time == time):
                for reader, bracketed in zip(self.readers, found):
                    nodes = reader.plan.nodes.get(target.section)
                    if nodes is None or bracketed is None:
                        continue
                    values = reader.section(target.section, bracketed)
                    for name in FIELDS:
                        self.sections[target.section][name][position, nodes] = values[name]
        for target in self.targets.frames:
            giving = [
                index for index, reader in enumerate(self.readers) if target.site in reader.plan.columns
            ]
            positions = np.flatnonzero(target.time == time)
            if positions.size == 0 or any(found[index] is None for index in giving):
                continue
            for position in positions:
                cells, columns = self._frame_cells(target, int(position), giving, cube)
                # One domain counts whole cells; several weight each cell by its area, in coarsest cells.
                weights = (
                    cells["area"] / float(np.median(self.readers[giving[-1]].plan.area))
                    if len(giving) > 1
                    else None
                )
                self.counts[target.site][position] = plume_cell_counts(
                    cells["heights"],
                    cells["tracer"],
                    cells["inside"],
                    target.height_bins,
                    target.threshold,
                    weights=weights,
                )
                plume = plume_cell_mask(cells["tracer"], cells["inside"], target.threshold)
                tops = column_tops(cells["heights"], plume, cells["column"], len(columns["lat"]))
                owned = columns["owned"]
                self.tops[target.site][position] = tops_on_map(
                    columns["lat"][owned],
                    columns["lon"][owned],
                    tops[owned],
                    columns["size"][owned],
                    target.latitude,
                    target.longitude,
                )
                if points is not None and target is self.targets.frames[0]:
                    self._frame_cloud(time, cells, columns, plume, points)
        if points is not None and not self.targets.frames:
            self._cloud(time, found, cube, points)

    def _frame_cells(self, target: FrameTarget, position: int, giving: list[int], cube):
        """
        The cells of every domain that gives the frame, flattened into one list: their height, tracer,
        whether they count (a column the domain gives, inside the radar's beam band at that time), area,
        and column; and each column's latitude, longitude, size (m), and whether the domain gives it.
        """
        cells = {name: [] for name in ("heights", "tracer", "inside", "area", "column")}
        columns = {name: [] for name in ("lat", "lon", "size", "owned")}
        offset = 0
        for index in giving:
            plan = self.readers[index].plan
            heights, tracer = cube(index)
            owned = plan.columns[target.site]
            lowest, highest = beam_band_at(target, position, plan.lat, plan.lon)
            inside = owned[None] & visible_cells(heights, lowest[None], highest[None])
            ids = offset + np.arange(plan.lat.size).reshape(plan.lat.shape)
            cells["heights"].append(heights.ravel())
            cells["tracer"].append(tracer.ravel())
            cells["inside"].append(inside.ravel())
            cells["area"].append(np.broadcast_to(plan.area, heights.shape).ravel())
            cells["column"].append(np.broadcast_to(ids, heights.shape).ravel())
            columns["lat"].append(plan.lat.ravel())
            columns["lon"].append(plan.lon.ravel())
            columns["size"].append(np.sqrt(plan.area).ravel())
            columns["owned"].append(owned.ravel())
            offset += plan.lat.size
        return (
            {name: np.concatenate(items) for name, items in cells.items()},
            {name: np.concatenate(items) for name, items in columns.items()},
        )

    def _frame_cloud(self, time, cells: dict, columns: dict, plume: np.ndarray, points: dict) -> None:
        """The plume cells of figures over the first frame: those the frame's KPIs count."""
        kept = _thin(np.flatnonzero(plume), int(points.get("max_per_time", DEFAULT_MAX_POINTS)))
        column = cells["column"][kept]
        self.clouds.append(
            {
                "position_lon": columns["lon"][column],
                "position_lat": columns["lat"][column],
                "position_z": cells["heights"][kept],
                TRACER: cells["tracer"][kept],
            }
        )
        self.cloud_times.append(time)

    def _cloud(self, time, found, cube, points: dict) -> None:
        """The plume cells of figures when the case has no frame: around the sections."""
        giving = [index for index, _ in enumerate(self.readers) if found[index] is not None]
        if not giving:
            return
        heights, tracer, inside, _, lon, lat = _cells(
            [(self.readers[index].plan, cube(index), self.readers[index].plan.own) for index in giving]
        )
        threshold = load_parameters().radar_plume.tracer_threshold
        cells = _thin(
            np.flatnonzero(plume_cell_mask(tracer, inside, threshold)),
            int(points.get("max_per_time", DEFAULT_MAX_POINTS)),
        )
        self.clouds.append(
            {
                "position_lon": lon[cells],
                "position_lat": lat[cells],
                "position_z": heights[cells],
                TRACER: tracer[cells],
            }
        )
        self.cloud_times.append(time)


def get_plume(  # pylint: disable=too-many-arguments,too-many-locals
    new_h5: h5py.File,
    ref_path,
    wrf_dataset: WrfRun | Dataset,
    ref_time: str,
    tracer: str = DEFAULT_TRACER,
    domain: str | None = None,
    max_time_gap: float = DEFAULT_MAX_TIME_GAP,
    points: dict | None = None,
    compression_lvl: int = 3,
    tracer_units: str | None = None,
) -> dict[str, int]:
    """Write the run's plume at the targets of the case of ``ref_path`` (a benchmark kit's template or
    an observation file), and return the number of times written for each target.

    ``tracer`` is the WRF-SFIRE fire tracer compared with the radar (``tr17_1`` to ``tr17_8``). Every
    domain that holds it gives the targets, each position from the finest one that covers it;
    ``domain`` (``d01``, ...) takes every position from that one alone, NaN where it does not reach.
    ``points`` (``max_per_time``, ``every_s``) also writes the model's plume cells for figures, over the
    first frame (or around the sections), every ``every_s`` s; None leaves them out. ``ref_time`` is
    the time XTIME counts from. The tracer is written in ug/m^3 (``TRACER_CONVERSION``) from its units
    in the output, or ``tracer_units`` when given, which must be a mass per kg of dry air; an output
    that states none is in ``DEFAULT_TRACER_UNITS``.
    """
    if not TRACER_NAME.fullmatch(tracer):
        raise ValueError(f"plume.tracer must be one of tr17_1 to tr17_8, not {tracer!r}")
    if max_time_gap <= 0:
        raise ValueError("plume.max_time_gap must be above 0 s")
    targets = plume_targets(ref_path)
    if not targets.sections and not targets.frames:
        raise ValueError(
            "the reference holds no radar plume target (cross section or frame): a kit's template made by "
            "a FireBench before them; download the kit again"
        )
    run = as_run(wrf_dataset)
    candidates = _candidates(run, tracer, domain)
    units, _ = _tracer_units(candidates, tracer, tracer_units)
    factor = mixing_ratio_factor(units)
    plans = _assign(candidates, targets.sections, targets.frames)
    for plan in plans:
        _cut(plan, targets.sections, candidates)
    origin = _utc(ref_time)
    writer = _Writer(
        targets,
        [_Run(plan, targets.sections, tracer, origin, float(max_time_gap), factor) for plan in plans],
    )
    times = np.unique(np.concatenate([target.time for target in (*targets.sections, *targets.frames)]))
    wanted = (
        _point_times(times, float(points.get("every_s", DEFAULT_POINTS_EVERY)))
        if points is not None
        else set()
    )
    for time in times:
        writer.at(time, points if time in wanted else None)

    attrs = {
        "source_variable": tracer,
        "source_units": units,
        "tracer_conversion": TRACER_CONVERSION,
        "time_interpolation": INTERPOLATION,
        "max_time_gap": float(max_time_gap),
        "max_time_gap_units": "s",
    }
    new_h5.require_group(targets.model_group).attrs[
        "description"
    ] = f"WRF-SFIRE's fire tracer {tracer} and wind at the radar's times and places"
    written: dict[str, int] = {}
    for target in targets.sections:
        values = {
            name: array.reshape(len(target.time), *target.latitude.shape)
            for name, array in writer.sections[target.section].items()
        }
        written[target.section] = int(np.isfinite(values[TRACER]).any(axis=(1, 2)).sum())
        used = ", ".join(plan.domain.name for plan in plans if target.section in plan.nodes)
        write_plume_section(
            new_h5,
            target,
            values,
            tracer_units=CONCENTRATION_UNITS,
            attrs={**attrs, "domain": used},
            compression_lvl=compression_lvl,
        )
    for target in targets.frames:
        written[target.site] = int(np.isfinite(writer.counts[target.site]).all(axis=1).sum())
        used = [plan.domain.name for plan in plans if target.site in plan.columns]
        frame_attrs = {**attrs, "domain": ", ".join(used)}
        if len(used) > 1:
            frame_attrs["cell_weights"] = f"cell area over the median cell area of {used[-1]}"
        write_plume_frame(
            new_h5,
            target,
            writer.counts[target.site],
            writer.tops[target.site],
            attrs={
                **frame_attrs,
                "visibility": "radar beam band (lowest_beam to highest_beam of the template)",
            },
            compression_lvl=compression_lvl,
        )
    if not any(written.values()):
        raise ValueError(
            "no target time of the case falls in the run's output times (or within max_time_gap of two)"
        )
    if points is not None and writer.clouds:
        origin_text = targets.frames[0].time_origin if targets.frames else targets.sections[0].time_origin
        write_plume_points(
            new_h5,
            targets.model_group,
            np.array(writer.cloud_times, dtype="datetime64[ms]"),
            origin_text,
            writer.clouds,
            tracer_units=CONCENTRATION_UNITS,
            compression_lvl=compression_lvl,
        )
    return written


def _tracer_units(candidates: list[WrfDomain], tracer: str, given: str | None) -> tuple[str, str]:
    """The units of the tracer and where they come from: ``given``, else those of the first domain that
    holds it, else ``DEFAULT_TRACER_UNITS`` when the output states none."""
    if given is not None:
        return given, "plume.tracer_units"
    holder = next(item for item in candidates if item.has(tracer))
    stated = str(getattr(holder.variable(tracer), "units", "")).strip()
    if stated:
        return stated, "the output"
    return DEFAULT_TRACER_UNITS, "WRF-SFIRE's, the output states none"


def _describe_domain(item: WrfDomain, tracer: str) -> list[str]:
    tracers = [
        f"{name} {tuple(item.datasets[0][name].dimensions)}"
        + (f" [{item.datasets[0][name].units}]" if getattr(item.datasets[0][name], "units", "") else "")
        for name in item.datasets[0].variables
        if "tr17" in name.lower()
    ]
    missing = [name for name in ("XTIME", *ATMOSPHERE) if not item.has(name)]
    xtime = item.xtime() if item.has("XTIME") else np.empty(0)
    steps = np.diff(xtime)
    lines = [
        f"{item.name}: {len(item.datasets)} file(s); "
        + (f"tracers {', '.join(tracers)}" if tracers else "no tr17 tracer"),
        f"  {tracer}: {'yes' if item.has(tracer) else 'NOT IN THE OUTPUT'}; "
        + (f"missing: {', '.join(missing)}" if missing else "every other field the plume needs is there"),
    ]
    if xtime.size:
        every = f"every {steps.min():g} to {steps.max():g} min" if steps.size else "one output time"
        lines.append(f"  output {every}, XTIME {xtime[0]:g} to {xtime[-1]:g} min")
    return lines


def _in_run(readers: list[_Run], times: np.ndarray) -> int:
    """How many of ``times`` every one of ``readers`` brackets, without reading a field."""
    count = 0
    for time in times:
        minutes = float((time - readers[0].origin) / np.timedelta64(60_000, "ms"))
        if all(bracket(reader.xtime, minutes, reader.max_time_gap) is not None for reader in readers):
            count += 1
    return count


def inspect_plume(  # pylint: disable=too-many-arguments,too-many-locals
    ref_path,
    wrf_dataset: WrfRun | Dataset,
    ref_time: str,
    tracer: str = DEFAULT_TRACER,
    domain: str | None = None,
    max_time_gap: float = DEFAULT_MAX_TIME_GAP,
    tracer_units: str | None = None,
) -> list[str]:
    """What ``get_plume`` would read and write, as lines of text, without reading any 3D field: the
    tracers and fields of each domain, the tracer's units and their conversion to ug/m^3, which domain
    gives each target, how many radar times the run covers, and about how large the output is."""
    run = as_run(wrf_dataset)
    lines = [line for item in run.domains for line in _describe_domain(item, tracer)]
    try:
        targets = plume_targets(ref_path)
    except ValueError as error:
        return [*lines, f"Targets: {error}"]
    if not targets.sections and not targets.frames:
        return [
            *lines,
            "Targets: none. The template was made by a FireBench before the radar plume targets: "
            "download the kit again.",
        ]
    try:
        candidates = _candidates(run, tracer, domain)
        plans = _assign(candidates, targets.sections, targets.frames)
    except (KeyError, ValueError) as error:
        return [*lines, f"Targets: {str(error).strip(chr(39))}"]
    units, origin = _tracer_units(candidates, tracer, tracer_units)
    try:
        factor = mixing_ratio_factor(units)
        lines.append(
            f"Tracer {tracer} in {units} ({origin}): times {factor:g} to ug/kg, then times the dry-air "
            "density, to ug/m^3"
        )
    except ValueError as error:
        lines.append(f"Tracer {tracer}: {error}")
    threshold = (
        targets.frames[0].threshold if targets.frames else load_parameters().radar_plume.tracer_threshold
    )
    lines.append(f"Plume threshold of the case: {threshold:g} {CONCENTRATION_UNITS}")
    origin = _utc(ref_time)
    size = 0
    for target in targets.sections:
        giving = [plan for plan in plans if target.section in plan.nodes]
        shares = ", ".join(
            f"{plan.domain.name} {100 * plan.nodes[target.section].size / target.latitude.size:.0f} %"
            for plan in giving
        )
        readers = [_Run(plan, [], tracer, origin, float(max_time_gap)) for plan in giving]
        # A node takes the time when its own domain gives it: count the times the finest giving domain has.
        times = _in_run(readers[:1], target.time)
        lines.append(
            f"Section {target.section}: {target.latitude.size} nodes ({shares}); "
            f"{times} of {len(target.time)} radar times inside the run"
        )
        size += len(target.time) * target.latitude.size * 4 * 4
    for target in targets.frames:
        giving = [plan for plan in plans if target.site in plan.columns]
        shares = ", ".join(
            f"{plan.domain.name} {int(plan.columns[target.site].sum())} columns" for plan in giving
        )
        readers = [_Run(plan, [], tracer, origin, float(max_time_gap)) for plan in giving]
        lines.append(
            f"Frame of {target.site}: {shares}; map {len(target.latitude)} x {len(target.longitude)}; "
            f"{_in_run(readers, target.time)} of {len(target.time)} radar times inside the run"
        )
        size += len(target.time) * (
            (len(target.height_bins) - 1) * 8 + len(target.latitude) * len(target.longitude) * 4
        )
    lines.append(f"Output: about {size / 1e6:.1f} MB before compression, without the points for figures")
    return lines
