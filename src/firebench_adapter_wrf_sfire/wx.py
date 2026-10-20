import numpy as np
import hdf5plugin
import h5py
from netCDF4 import Dataset
import firebench.standardize as fs
from datetime import datetime, timezone
from firebench import Quantity, svn
from bisect import bisect_left, bisect_right
from firebench.adapter_common.interpolation import find_nearest_grid_point
from firebench.adapter_common.weather import write_model_sensor_height_metadata
from scipy.interpolate import interp1d

from ._grid import fire_grid_slice
from ._skip import skip
from ._wrfout import WrfDomain, WrfRun, as_run

GRAVITY = 9.81


def _as_utc_datetime(dt: datetime) -> datetime:
    """Return aware datetimes in UTC; leave naive datetimes unchanged."""
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc)


def _datetime_fromisoformat_utc(value: str) -> datetime:
    """Parse ISO-8601 datetimes, accepting trailing Z, and normalize aware values to UTC."""
    return _as_utc_datetime(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _validate_matching_timezone(*datetimes: datetime):
    """Avoid mixing aware UTC datetimes with naive wall-clock datetimes."""
    has_tz = [dt.tzinfo is not None for dt in datetimes]
    if any(has_tz) and not all(has_tz):
        raise ValueError(
            "Cannot mix timezone-aware and timezone-naive datetimes. "
            "Use UTC consistently for sfire_ref_time, period bounds, and station time_origin."
        )


def _read_station_time_origin(obs_h5: h5py.File, station_path: str) -> datetime:
    """Read station time origin as datetime from HDF5 attributes."""
    station = obs_h5[station_path]
    return _datetime_fromisoformat_utc(station["time"].attrs["time_origin"])


def _station_time_vector_seconds(obs_h5: h5py.File, station_path: str) -> np.ndarray:
    """Return station 'time' vector converted to seconds (float64)."""
    station = obs_h5[station_path]
    time_raw = np.array(station["time"][:])
    time_units = station["time"].attrs["time_units"]
    return Quantity(time_raw, time_units).to("s").magnitude.astype(np.float64)


def _period_offsets_seconds(
    dt_origin: datetime, start_dt: datetime, end_dt: datetime
) -> tuple[float, float]:
    """Compute (start, end) offsets in seconds relative to dt_origin."""
    _validate_matching_timezone(dt_origin, start_dt, end_dt)
    dt_origin = _as_utc_datetime(dt_origin)
    start_dt = _as_utc_datetime(start_dt)
    end_dt = _as_utc_datetime(end_dt)
    offset_start = (start_dt - dt_origin).total_seconds()
    offset_end = (end_dt - dt_origin).total_seconds()
    return offset_start, offset_end


def _mask_between_inclusive(x: np.ndarray, x0: float, x1: float) -> np.ndarray:
    """Boolean mask for x0 <= x <= x1 (inclusive)."""
    return (x >= x0) & (x <= x1)


def _time_in_period_seconds(
    obs_h5: h5py.File,
    station_path: str,
    period_start_dt: datetime,
    period_end_dt: datetime,
) -> np.ndarray:
    """
    Equivalent to __get_time_vector_s: station time in seconds filtered to a period window.
    """
    dt_origin = _read_station_time_origin(obs_h5, station_path)
    time_vec_s = _station_time_vector_seconds(obs_h5, station_path)
    offset_start, offset_end = _period_offsets_seconds(dt_origin, period_start_dt, period_end_dt)
    return (
        time_vec_s[_mask_between_inclusive(time_vec_s, offset_start, offset_end)].astype(np.float64),
        dt_origin,
    )


def _sfire_time_seconds(domain: WrfDomain, offset_start_s: float) -> np.ndarray:
    """
    Compute SFIRE time in seconds relative to the period start.
    Matches existing logic: XTIME (min) -> seconds, then subtract offset_start.
    """
    return Quantity(domain.xtime(), "min").to("s").magnitude - offset_start_s


def _bracket_indices_inclusive(model_time: np.ndarray, ref_start: float, ref_end: float):
    """
    Find indices that frame [ref_start, ref_end] using the same bisect logic:
      i_start = last index with model_time[i] <= ref_start
      i_end   = first index with model_time[i] >= ref_end
    """
    i_start = bisect_right(model_time, ref_start) - 1
    i_end = bisect_left(model_time, ref_end)
    return i_start, i_end


def _validate_bracket(
    i_start: int,
    i_end: int,
    model_time: np.ndarray,
    ref_start: float,
    ref_end: float,
):
    """Raise if bracket falls outside model time range (same as existing check)."""
    model_time_len = len(model_time)
    if i_start < 0 or i_end >= model_time_len:
        model_start = model_time[0] if model_time_len else np.nan
        model_end = model_time[-1] if model_time_len else np.nan
        print(
            i_start,
            i_end,
            model_time_len,
            f"ref=[{ref_start}, {ref_end}] s",
            f"model=[{model_start}, {model_end}] s",
        )
        raise ValueError("Reference time outside model time range")


def _create_station_group_with_attrs(
    new_h5: h5py.File,
    obs_h5: h5py.File,
    station_path: str,
    attrs_to_copy: list[str],
) -> h5py.Group:
    """Create station group and copy select attributes if present."""
    new_grp = new_h5.create_group(station_path)
    src_attrs = obs_h5[station_path].attrs
    for attr in attrs_to_copy:
        if attr in src_attrs.keys():
            new_grp.attrs[attr] = src_attrs[attr]
    return new_grp


def _write_time_dataset_seconds(
    grp: h5py.Group,
    time_s: np.ndarray,
    time_origin: datetime,
    compression_lvl: int,
) -> h5py.Dataset:
    """Create 'time' dataset with standard attrs (origin + units)."""
    ds = grp.create_dataset("time", data=time_s, **hdf5plugin.Zstd(clevel=compression_lvl))
    ds.attrs["time_origin"] = fs.datetime_to_iso8601(time_origin)
    ds.attrs["time_units"] = "s"


def _interp_linear_1d_strict(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """
    Linear interpolation with the same settings as your current usage:
    bounds_error=True, fill_value=np.nan, assume_sorted=True.

    ``y`` is read as float64: on WRF's float32 values scipy takes its own linear path, which gives NaN
    when ``x`` holds a single time (a station whose one observation time is an output time).
    """
    f = interp1d(
        x,
        np.asarray(y, dtype=np.float64),
        kind="linear",
        bounds_error=True,
        fill_value=np.nan,
        assume_sorted=True,
    )
    return f(xq)


def _sensor_height_check_ok(
    sensor_height: float,
    min_height_ok: float = None,
    max_height_ok: float = None,
    reference_height_ok: float = None,
    sym_distance_ok: float = None,
):
    """
    Validate whether a sensor height is within an acceptable interval.

    Supported interval definitions:
    - [min_height_ok, max_height_ok]
    - [reference_height_ok - sym_distance_ok, reference_height_ok + sym_distance_ok]
    """
    if min_height_ok is not None or max_height_ok is not None:
        if min_height_ok is None or max_height_ok is None:
            raise ValueError("Both min_height_ok and max_height_ok must be provided")

        if reference_height_ok is not None or sym_distance_ok is not None:
            raise ValueError("Cannot mix min/max and symmetric interval definitions")

        return min_height_ok <= sensor_height <= max_height_ok

    if reference_height_ok is not None or sym_distance_ok is not None:
        if reference_height_ok is None or sym_distance_ok is None:
            raise ValueError("Both reference_height_ok and sym_distance_ok must be provided")

        return (
            reference_height_ok - sym_distance_ok <= sensor_height <= reference_height_ok + sym_distance_ok
        )

    raise ValueError("No valid height interval definition provided")


def _rh_from_wrf_q2_t2_psfc(q2_mixratio, t2_k, psfc_pa):
    eps = 0.622
    p_hpa = 0.01 * psfc_pa

    # vapor pressure from mixing ratio (hPa)
    e_hpa = (q2_mixratio * p_hpa) / (eps + q2_mixratio)

    # saturation vapor pressure over water (Tetens, hPa)
    t_c = t2_k - 273.15
    es_hpa = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))

    rh = 100.0 * (e_hpa / es_hpa)
    return rh


def _get_z(ds: Dataset, i, j, k, n):
    return (
        0.5
        * (
            ds["PHB"][n, k, j, i]
            + ds["PH"][n, k, j, i]
            + ds["PHB"][n, k + 1, j, i]
            + ds["PH"][n, k + 1, j, i]
        )
        / GRAVITY
    )


def wind_dir_from_uv(u, v):
    """
    Compute meteorological wind direction.
    u: zonal wind (m/s), +east
    v: meridional wind (m/s), +north
    returns degrees in [0, 360)
    """
    wd = np.degrees(np.arctan2(-u, -v))
    return (wd + 360) % 360


def wind_speed(u, v):
    return np.sqrt(u**2 + v**2)


def _get_wind_rotation(domain: WrfDomain, time_slice, j: int, i: int):
    """Return mesh rotation coefficients, defaulting to a Cartesian grid."""
    if not domain.has("COSALPHA") or not domain.has("SINALPHA"):
        print("COSALPHA/SINALPHA not found. Use Cartesian grid wind rotation.")
        return 1.0, 0.0

    return (
        domain.series("COSALPHA", time_slice, j, i),
        domain.series("SINALPHA", time_slice, j, i),
    )


def wind_from10m_with_roughness(s10, z, z0, z0_min=1e-4):
    z0_eff = np.maximum(z0, z0_min)
    z_eff = np.maximum(z, z0_eff)

    denom = np.log(10.0 / z0_eff)
    denom = np.where(denom > 0, denom, np.nan)

    u = s10 * np.log(z_eff / z0_eff) / denom
    return np.maximum(u, 0.0)


# Errors that mean the simulation or the reference cannot give a variable or a station: a missing
# netCDF variable (IndexError), a missing attribute or dataset (KeyError), a value out of range.
SKIPPABLE_ERRORS = (KeyError, IndexError, ValueError)


def __process_weather_station(
    new_h5: h5py.File,
    obs_h5: h5py.File,
    run: WrfRun,
    station_path: str,
    sfire_ref_time: datetime,
    period_start_dt: datetime,
    period_end_dt: datetime,
    fire_grid_bound: int,
    compression_lvl: int,
    i_fmc_10h: int | None,
):
    STATION_ATTRS_TO_COPY = [
        "position_lat",
        "position_lon",
        "position_lat_units",
        "position_lon_units",
    ]
    time_in_period, time_origin = _time_in_period_seconds(
        obs_h5, station_path, period_start_dt, period_end_dt
    )
    if len(time_in_period) == 0:
        print("No data in time period. Skip")
        return

    grid_slice = fire_grid_slice(fire_grid_bound)
    # Indices found on the trimmed grid, read on the full one.
    grid_offset = grid_slice.start or 0
    station_lat = obs_h5[station_path].attrs["position_lat"]
    station_lon = obs_h5[station_path].attrs["position_lon"]
    # The domains that hold the station, finest first: a variable comes from the first that gives it.
    candidates = [domain for domain in run.domains if domain.covers(station_lat, station_lon, grid_slice)]
    if not candidates:
        # The nearest grid point would be on the edge of a domain, far from the station.
        skip(station_path, "outside every model domain")
        return

    # SFIRE time
    _validate_matching_timezone(sfire_ref_time, period_start_dt, period_end_dt)
    sfire_ref_time = _as_utc_datetime(sfire_ref_time)
    period_start_dt = _as_utc_datetime(period_start_dt)
    period_end_dt = _as_utc_datetime(period_end_dt)
    offset_start = (period_start_dt - sfire_ref_time).total_seconds()
    offset_station_to_period_start = (time_origin - period_start_dt).total_seconds()
    time_in_period_sfire = time_in_period + offset_station_to_period_start

    # Keep the observation times the simulation covers: the periods inside them can be scored. The
    # finest domain that covers one of them gives the station's times.
    covered = None
    for domain in candidates:
        time_sfire = _sfire_time_seconds(domain, offset_start)
        if time_sfire.size:
            mask = _mask_between_inclusive(time_in_period_sfire, time_sfire[0], time_sfire[-1])
            if mask.any():
                covered = mask
                break
    if covered is None:
        skip(station_path, "no observation time inside the simulation")
        return
    if not covered.all():
        print(f"Simulation covers {covered.sum()} of {covered.size} observation times")
    time_in_period = time_in_period[covered]
    time_in_period_sfire = time_in_period_sfire[covered]

    # Add time to new group
    new_wx = _create_station_group_with_attrs(new_h5, obs_h5, station_path, STATION_ATTRS_TO_COPY)
    _write_time_dataset_seconds(new_wx, time_in_period, time_origin, compression_lvl)

    points = {}

    def point(domain: WrfDomain):
        """The output times of ``domain`` that frame the station's times, and its nearest grid point."""
        if domain.number not in points:
            time_sfire = _sfire_time_seconds(domain, offset_start)
            i_start, i_end = _bracket_indices_inclusive(
                time_sfire, time_in_period_sfire[0], time_in_period_sfire[-1]
            )
            _validate_bracket(i_start, i_end, time_sfire, time_in_period_sfire[0], time_in_period_sfire[-1])
            xlat = domain.static("XLAT", grid_slice)
            xlon = domain.static("XLONG", grid_slice)
            Ny, Nx = xlat.shape
            i_nearest, j_nearest = find_nearest_grid_point(
                Nx,
                Ny,
                station_lat,
                station_lon,
                xlat,
                xlon,
                np.min(np.diff(xlat, axis=0)),
                np.min(np.diff(xlon, axis=1)),
            )
            points[domain.number] = (
                time_sfire[i_start : i_end + 1],
                slice(i_start, i_end + 1),
                j_nearest + grid_offset,
                i_nearest + grid_offset,
            )
        return points[domain.number]

    def obs_sensor_height(variable: str) -> float:
        return (
            fs.read_quantity_from_fb_attribute(
                f"{station_path}/{variable}", "sensor_height", obs_h5, dtype=float
            )
            .to("m")
            .magnitude
        )

    def write(
        variable: str,
        model_time: np.ndarray,
        raw: np.ndarray,
        units: str,
        prepared_height_m: float | None,
    ):
        var = new_wx.create_dataset(
            variable,
            data=_interp_linear_1d_strict(model_time, raw, time_in_period_sfire),
            **hdf5plugin.Zstd(clevel=compression_lvl),
        )
        var.attrs["units"] = units
        if prepared_height_m is not None:
            write_model_sensor_height_metadata(var, Quantity(prepared_height_m, "m"))

    def air_temperature(domain: WrfDomain):
        # Use T2 within interval 0m to 4m
        if not _sensor_height_check_ok(obs_sensor_height(svn.AIR_TEMPERATURE.value), 0, 4):
            raise ValueError("no method for a sensor height outside 0 m to 4 m")
        model_time, model_slice, j, i = point(domain)
        print("Nearest T2")
        write(svn.AIR_TEMPERATURE.value, model_time, domain.series("T2", model_slice, j, i), "degK", 2.0)

    def relative_humidity(domain: WrfDomain):
        if not _sensor_height_check_ok(obs_sensor_height(svn.RELATIVE_HUMIDITY.value), 0, 4):
            raise ValueError("no method for a sensor height outside 0 m to 4 m")
        model_time, model_slice, j, i = point(domain)
        print("Nearest RH2")
        t2_raw = domain.series("T2", model_slice, j, i)
        q2_raw = domain.series("Q2", model_slice, j, i)
        psfc_raw = domain.series("PSFC", model_slice, j, i)
        write(
            svn.RELATIVE_HUMIDITY.value,
            model_time,
            _rh_from_wrf_q2_t2_psfc(q2_raw, t2_raw, psfc_raw),
            "percent",
            2.0,
        )

    def wind_direction(domain: WrfDomain):
        wd_sensor_height = obs_sensor_height(svn.WIND_DIRECTION.value)
        if not _sensor_height_check_ok(wd_sensor_height, 0, 20):
            raise ValueError("no method for a sensor height outside 0 m to 20 m")
        model_time, model_slice, j, i = point(domain)
        print("Nearest WD10 used for WD measured between 0m and 20m")
        u10_raw = domain.series("U10", model_slice, j, i)
        v10_raw = domain.series("V10", model_slice, j, i)
        cosalpha_raw, sinalpha_raw = _get_wind_rotation(domain, model_slice, j, i)
        u_true = u10_raw * cosalpha_raw - v10_raw * sinalpha_raw
        v_true = u10_raw * sinalpha_raw + v10_raw * cosalpha_raw
        # The log profile that scales the wind speed keeps its direction: the 10 m direction
        # is the direction at the sensor height.
        write(
            svn.WIND_DIRECTION.value,
            model_time,
            wind_dir_from_uv(u_true, v_true),
            "degree",
            wd_sensor_height,
        )

    def wind_speed_at_sensor(domain: WrfDomain):
        sensor_height = obs_sensor_height(svn.WIND_SPEED.value)
        model_time, model_slice, j, i = point(domain)
        u10_raw = domain.series("U10", model_slice, j, i)
        v10_raw = domain.series("V10", model_slice, j, i)
        s10 = wind_speed(u10_raw, v10_raw)
        if domain.has_fire_grid() and domain.has("WZ0"):
            fxlat = domain.static("FXLAT", grid_slice)
            fxlon = domain.static("FXLONG", grid_slice)
            Nyf, Nxf = fxlat.shape
            if_nearest, jf_nearest = find_nearest_grid_point(
                Nxf,
                Nyf,
                station_lat,
                station_lon,
                fxlat,
                fxlon,
                np.min(np.diff(fxlat, axis=0)),
                np.min(np.diff(fxlon, axis=1)),
            )
            print("Nearest WS10 with S10 scaling (fire grid roughness WZ0)")
            z0 = domain.series("WZ0", model_slice, jf_nearest + grid_offset, if_nearest + grid_offset)
        elif domain.has("ZNT"):
            # A domain without a fire grid has no WZ0: the atmospheric roughness length instead.
            print("Nearest WS10 with S10 scaling (atmospheric roughness ZNT)")
            z0 = domain.series("ZNT", model_slice, j, i)
        else:
            raise KeyError(f"no roughness length (WZ0 or ZNT) in the output of {domain.name}")
        write(
            svn.WIND_SPEED.value,
            model_time,
            wind_from10m_with_roughness(s10, sensor_height, z0),
            "m/s",
            sensor_height,
        )
        # For debug
        write("WS10", model_time, s10, "m/s", None)

    def fmc_10h(domain: WrfDomain):
        model_time, model_slice, j, i = point(domain)
        if domain.ndim("FMC_GC") == 3:
            # A postprocessed field holding the 10-hour class alone.
            fmc10h_raw = domain.series("FMC_GC", model_slice, j, i)
        elif i_fmc_10h is None:
            raise ValueError(
                "FMC_GC has a fuel class dimension: set weather.i_fmc_10h to the 10-hour class"
            )
        else:
            fmc10h_raw = domain.series("FMC_GC", model_slice, i_fmc_10h, j, i)
        write(svn.FUEL_MOISTURE_CONTENT_10H.value, model_time, fmc10h_raw, "dimensionless", None)

    station = obs_h5[station_path]
    written = 0
    for variable, process in (
        (svn.AIR_TEMPERATURE.value, air_temperature),
        (svn.RELATIVE_HUMIDITY.value, relative_humidity),
        (svn.WIND_DIRECTION.value, wind_direction),
        (svn.WIND_SPEED.value, wind_speed_at_sensor),
        (svn.FUEL_MOISTURE_CONTENT_10H.value, fmc_10h),
    ):
        if variable not in station.keys():
            continue
        print(f"Process {variable}")
        errors = []
        for domain in candidates:
            try:
                process(domain)
            except SKIPPABLE_ERRORS as exc:
                errors.append(exc)
                continue
            print(f"{variable} from {domain.name}")
            written += 1
            break
        else:
            skip(f"{station_path}/{variable}", errors[0])
    if not written:
        del new_h5[station_path]


def get_weather_stations(
    new_h5: h5py.File,
    obs_h5: h5py.File,
    wrf_dataset: WrfRun | Dataset,
    sfire_ref_time: str | None,
    period_start_dt: datetime,
    period_end_dt: datetime,
    fire_grid_bound: int = 25,
    compression_lvl: int = 3,
    i_fmc_10h: int | None = 1,
):
    """Sample WRF-SFIRE weather variables at observation stations.

    ``wrf_dataset`` is a run (``open_run``: every wrfout file of every domain) or one open wrfout
    file. A station takes each variable from the finest domain that holds it and gives the variable;
    its times come from the finest domain whose output covers one of them. ``sfire_ref_time``, the
    time XTIME counts from, defaults to the run's ``SIMULATION_START_DATE``.

    ``i_fmc_10h`` is the zero-based fuel-layer index for 10-hour fuel moisture
    in a 4D ``FMC_GC`` field (time, fuel layer, y, x), defaulting to 1. A 3D field
    (time, y, x) holds only 10-hour fuel moisture and is read whatever ``i_fmc_10h`` is.

    A station or a variable the simulation cannot give is left out with a ``SkippedWarning``, and
    a station keeps only the observation times the simulation covers.
    """
    run = as_run(wrf_dataset)
    if sfire_ref_time is None:
        sfire_ref_time = run.start_time()
    origin = _datetime_fromisoformat_utc(sfire_ref_time)
    for ts in obs_h5[f"{fs.TIME_SERIES}"].keys():
        if ts.startswith("station_"):
            print(f"Process {ts}")
            station_path = f"{fs.TIME_SERIES}/{ts}"
            try:
                __process_weather_station(
                    new_h5,
                    obs_h5,
                    run,
                    station_path,
                    origin,
                    period_start_dt,
                    period_end_dt,
                    fire_grid_bound,
                    compression_lvl,
                    i_fmc_10h,
                )
            except SKIPPABLE_ERRORS as exc:
                if station_path in new_h5:
                    del new_h5[station_path]
                skip(station_path, exc)
