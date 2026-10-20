"""A WRF-SFIRE run: its wrfout files grouped by domain, each domain read as one time series."""

from __future__ import annotations

import re
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import shapely
from firebench.adapter_common.interpolation import nearest_2d
from netCDF4 import Dataset

WRFOUT_NAME = re.compile(r"^wrfout_d(\d{2})_")
# Global attribute WRF writes in every output file: the time XTIME and TIGN_G count from.
START_ATTRIBUTE = "SIMULATION_START_DATE"


class WrfDomain:
    """The output files of one domain, in time order, read as one time series.

    The time index of ``series`` is the index into ``xtime()``: the output times of every file, a
    time already given by the file before left out (a restart writes its first time again).
    """

    def __init__(self, number: int, datasets: list[Dataset]):
        self.number = number
        self.datasets = datasets
        self._pieces: list[tuple[Dataset, np.ndarray, int]] | None = None
        self._xtime: np.ndarray | None = None
        # Keyed by the bounds of the slice: a slice is hashable only from Python 3.12.
        self._rings: dict[tuple[str, str, int | None, int | None], shapely.Polygon] = {}

    @property
    def name(self) -> str:
        """The domain as WRF names its files: d01, d02, ..."""
        return f"d{self.number:02d}"

    def has(self, variable: str) -> bool:
        """Whether the files of the domain hold ``variable``."""
        return variable in self.datasets[0].variables

    def variable(self, name: str, dataset: Dataset | None = None, ndim: int | None = None):
        """The netCDF variable ``name``, refused clearly when the output lacks it or lays it out otherwise.

        A reduced output (WRF's ``iofields_filename``) leaves variables out; the error names the one
        missing, and the section or the variable that needs it is skipped.
        """
        dataset = self.datasets[0] if dataset is None else dataset
        if name not in dataset.variables:
            raise KeyError(f"{name} is not in the output of {self.name}")
        variable = dataset[name]
        if ndim is not None and variable.ndim != ndim:
            raise ValueError(
                f"{name} of {self.name} has the dimensions {variable.dimensions}, not the {ndim} expected"
            )
        return variable

    def ndim(self, name: str) -> int:
        """The number of dimensions of ``name``, time included."""
        return self.variable(name).ndim

    def has_fire_grid(self) -> bool:
        """Whether the domain runs the fire model: a domain without one writes FXLAT as zeros."""
        if not self.has("FXLAT"):
            return False
        fxlat = np.asarray(self.variable("FXLAT", ndim=3)[0])
        return fxlat.size > 1 and float(np.ptp(fxlat)) > 0

    def xtime(self) -> np.ndarray:
        """Output times in minutes since the start of the run (XTIME), over every file."""
        if self._xtime is None:
            pieces, times, first = [], [], 0
            for dataset in self.datasets:
                xtime = np.asarray(self.variable("XTIME", dataset, 1)[:], dtype=np.float64)
                local = np.flatnonzero(xtime > times[-1][-1]) if times else np.arange(xtime.size)
                if local.size and np.any(np.diff(xtime[local]) <= 0):
                    raise ValueError(f"XTIME of {dataset.filepath()} is not increasing")
                if local.size:
                    pieces.append((dataset, local, first))
                    times.append(xtime[local])
                    first += local.size
            self._pieces = pieces
            self._xtime = np.concatenate(times) if times else np.empty(0)
        return self._xtime

    def static(self, variable: str, grid_slice: slice) -> np.ndarray:
        """A field that does not change in time (grid coordinates), from the first output time."""
        return np.array(self.variable(variable, ndim=3)[0, grid_slice, grid_slice])

    def last(self, variable: str, grid_slice: slice):
        """A field at the last output time of the run, as netCDF4 reads it."""
        return self.variable(variable, self.datasets[-1], 3)[-1, grid_slice, grid_slice]

    def at(self, variable: str, index: int, grid_slice: slice) -> np.ndarray:
        """A field at output time ``index`` of ``xtime()``."""
        self.xtime()
        for dataset, local, first in self._pieces:
            if first <= index < first + local.size:
                values = self.variable(variable, dataset, 3)[local[index - first], grid_slice, grid_slice]
                return np.asarray(np.ma.getdata(values))
        raise IndexError(f"{self.name} has no output time {index}")

    def box(self, variable: str, rows: slice, cols: slice) -> np.ndarray:
        """A 2D field (south_north, west_east) cut to ``rows`` and ``cols``, from the first output time."""
        return np.array(np.ma.getdata(self.variable(variable, ndim=3)[0, rows, cols]))

    def volume(self, variable: str, index: int, rows: slice, cols: slice) -> np.ndarray:
        """A 3D field (bottom_top, south_north, west_east) at output time ``index`` of ``xtime()``,
        cut to ``rows`` and ``cols``; the staggered dimension, if any, as the file gives it."""
        self.xtime()
        for dataset, local, first in self._pieces:
            if first <= index < first + local.size:
                values = self.variable(variable, dataset, 4)[local[index - first], :, rows, cols]
                return np.asarray(np.ma.getdata(values))
        raise IndexError(f"{self.name} has no output time {index}")

    def series(self, variable: str, time_slice: slice, *point: int) -> np.ndarray:
        """The values of ``variable`` at ``point`` (layer, j, i or j, i) over ``time_slice`` of xtime()."""
        self.xtime()
        start, stop = time_slice.start, time_slice.stop
        parts = []
        for dataset, local, first in self._pieces:
            low, high = max(start, first), min(stop, first + local.size)
            if low >= high:
                continue
            indices = local[low - first : high - first]
            values = self.variable(variable, dataset, 1 + len(point))[
                (slice(indices[0], indices[-1] + 1),) + point
            ]
            parts.append(np.ma.getdata(values)[indices - indices[0]])
        return np.concatenate(parts)

    def covers(self, lat, lon, grid_slice: slice, fire: bool = False) -> np.ndarray:
        """Whether each point is inside the trimmed atmospheric grid (or fire grid), its edge included."""
        lat_name, lon_name = ("FXLAT", "FXLONG") if fire else ("XLAT", "XLONG")
        key = (lat_name, lon_name, grid_slice.start, grid_slice.stop)
        if key not in self._rings:
            lats = self.static(lat_name, grid_slice)
            lons = self.static(lon_name, grid_slice)
            ring_lat = np.concatenate([lats[0, :], lats[1:, -1], lats[-1, -2::-1], lats[-2:0:-1, 0]])
            ring_lon = np.concatenate([lons[0, :], lons[1:, -1], lons[-1, -2::-1], lons[-2:0:-1, 0]])
            self._rings[key] = shapely.Polygon(np.column_stack([ring_lon, ring_lat]))
        return shapely.intersects_xy(self._rings[key], np.asarray(lon), np.asarray(lat))

    def describe(self) -> str:
        """One line on the domain, for the log of a run."""
        xtime = self.xtime() if self.has("XTIME") else np.empty(0)
        span = f"XTIME {xtime[0]:g} to {xtime[-1]:g} min" if xtime.size else "no XTIME"
        fire = "fire grid" if self.has_fire_grid() else "no fire grid"
        return f"{self.name}: {len(self.datasets)} file(s), {span}, {fire}"


class WrfRun:
    """The domains of a run, finest first: the highest domain number is the innermost nest."""

    def __init__(self, domains: list[WrfDomain], stack: ExitStack | None = None):
        self.domains = sorted(domains, key=lambda domain: domain.number, reverse=True)
        self._stack = stack

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> WrfRun:
        """One open wrfout file as a run of one domain."""
        return cls([WrfDomain(_domain_number(Path(dataset.filepath())), [dataset])])

    def fire_domains(self) -> list[WrfDomain]:
        """The domains that run the fire model, finest first; the only domain when there is one."""
        if len(self.domains) == 1:
            return self.domains
        return [domain for domain in self.domains if domain.has_fire_grid()]

    def start_time(self) -> str:
        """The start of the run (SIMULATION_START_DATE, UTC) as ISO 8601."""
        dataset = self.domains[-1].datasets[0]
        if START_ATTRIBUTE not in dataset.ncattrs():
            raise ValueError(f"{dataset.filepath()} has no {START_ATTRIBUTE} attribute")
        start = datetime.strptime(dataset.getncattr(START_ATTRIBUTE), "%Y-%m-%d_%H:%M:%S")
        return start.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

    def close(self) -> None:
        """Close the files the run opened."""
        if self._stack is not None:
            self._stack.close()

    def __enter__(self) -> WrfRun:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def as_run(wrf: WrfRun | Dataset) -> WrfRun:
    """A run, or one open wrfout file as a run of one domain."""
    return wrf if isinstance(wrf, WrfRun) else WrfRun.from_dataset(wrf)


def _domain_number(path: Path) -> int:
    match = WRFOUT_NAME.match(path.name)
    return int(match.group(1)) if match else 1


def find_wrfout_files(path: Path) -> dict[int, list[Path]]:
    """The output files of a run by domain, in time order: one file, or the wrfout_dNN_* of a folder."""
    if path.is_file():
        return {_domain_number(path): [path]}
    files: dict[int, list[Path]] = {}
    for item in sorted(path.iterdir()):
        if item.is_file() and WRFOUT_NAME.match(item.name):
            files.setdefault(_domain_number(item), []).append(item)
    if not files:
        raise ValueError(f"no wrfout_dNN_* file in {path}")
    return files


def open_run(path: Path) -> WrfRun:
    """Open a run from one wrfout file or from the folder that holds its wrfout files."""
    files = find_wrfout_files(path)
    stack = ExitStack()
    try:
        domains = [
            WrfDomain(number, [stack.enter_context(Dataset(item, "r")) for item in items])
            for number, items in files.items()
        ]
    except BaseException:
        stack.close()
        raise
    return WrfRun(domains, stack)


def point_owners(domains: list[WrfDomain], lat, lon, grid_slice: slice, fire: bool = False) -> np.ndarray:
    """For each point, the index of the first domain that covers it, 0 when none does.

    With one domain every point is its own, inside the grid or not, as before the adapter read runs.
    """
    owners = np.zeros(np.shape(lat), dtype=int)
    if len(domains) < 2:
        return owners
    taken = domains[0].covers(lat, lon, grid_slice, fire)
    for index, domain in enumerate(domains[1:], start=1):
        mine = domain.covers(lat, lon, grid_slice, fire) & ~taken
        owners[mine] = index
        taken |= mine
    return owners


def nearest_by_domain(domains: list[WrfDomain], lat, lon, grid_slice: slice, field_of) -> np.ndarray:
    """Nearest value of ``field_of(domain)``, a field on the domain's fire grid, at each point.

    A point takes the field of the first fire domain that covers it (see ``point_owners``).
    """
    if not domains:
        raise ValueError("no domain of the run has a fire grid")
    lat, lon = np.asarray(lat), np.asarray(lon)
    owners = point_owners(domains, lat, lon, grid_slice, fire=True)
    result = None
    for index, domain in enumerate(domains):
        mine = owners == index
        if not mine.any():
            continue
        xlat = domain.static("FXLAT", grid_slice)
        xlon = domain.static("FXLONG", grid_slice)
        values = nearest_2d(field_of(domain), xlon, xlat, lon[mine], lat[mine])
        if result is None:
            result = np.zeros(lat.shape, dtype=values.dtype)
        result[mine] = values
    return result
