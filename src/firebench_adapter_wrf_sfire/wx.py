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


def _sfire_time_seconds(wrf_dataset: Dataset, offset_start_s: float) -> np.ndarray:
    """
    Compute SFIRE time in seconds relative to the period start.
    Matches existing logic: XTIME (min) -> seconds, then subtract offset_start.
    """
    xtime_min = np.array(wrf_dataset["XTIME"][:], dtype=np.float64)
    return Quantity(xtime_min, "min").to("s").magnitude - offset_start_s


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
    """
    f = interp1d(
        x,
        y,
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


def _get_wind_rotation(wrf_dataset: Dataset, time_slice, j: int, i: int):
    """Return mesh rotation coefficients, defaulting to a Cartesian grid."""
    if "COSALPHA" not in wrf_dataset.variables or "SINALPHA" not in wrf_dataset.variables:
        print("COSALPHA/SINALPHA not found. Use Cartesian grid wind rotation.")
        return 1.0, 0.0

    return (
        wrf_dataset["COSALPHA"][time_slice, j, i],
        wrf_dataset["SINALPHA"][time_slice, j, i],
    )


def wind_from10m_with_roughness(s10, z, z0, z0_min=1e-4):
    z0_eff = np.maximum(z0, z0_min)
    z_eff = np.maximum(z, z0_eff)

    denom = np.log(10.0 / z0_eff)
    denom = np.where(denom > 0, denom, np.nan)

    u = s10 * np.log(z_eff / z0_eff) / denom
    return np.maximum(u, 0.0)


def __process_weather_station(
    new_h5: h5py.File,
    obs_h5: h5py.File,
    wrf_dataset: Dataset,
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
    xlat = np.array(wrf_dataset["XLAT"][0, grid_slice, grid_slice])
    xlon = np.array(wrf_dataset["XLONG"][0, grid_slice, grid_slice])
    station_lat = obs_h5[station_path].attrs["position_lat"]
    station_lon = obs_h5[station_path].attrs["position_lon"]
    if not (xlat.min() <= station_lat <= xlat.max() and xlon.min() <= station_lon <= xlon.max()):
        # The nearest grid point would be on the edge of the domain, far from the station.
        print("Station outside the model domain. Skip")
        return

    # SFIRE time
    _validate_matching_timezone(sfire_ref_time, period_start_dt, period_end_dt)
    sfire_ref_time = _as_utc_datetime(sfire_ref_time)
    period_start_dt = _as_utc_datetime(period_start_dt)
    period_end_dt = _as_utc_datetime(period_end_dt)
    offset_start = (period_start_dt - sfire_ref_time).total_seconds()
    offset_station_to_period_start = (time_origin - period_start_dt).total_seconds()
    time_in_period_sfire = time_in_period + offset_station_to_period_start
    time_sfire = _sfire_time_seconds(wrf_dataset, offset_start)
    i_start, i_end = _bracket_indices_inclusive(
        time_sfire, time_in_period_sfire[0], time_in_period_sfire[-1]
    )
    _validate_bracket(i_start, i_end, time_sfire, time_in_period_sfire[0], time_in_period_sfire[-1])

    # Add time to new group
    new_wx = _create_station_group_with_attrs(new_h5, obs_h5, station_path, STATION_ATTRS_TO_COPY)
    _write_time_dataset_seconds(new_wx, time_in_period, time_origin, compression_lvl)

    # get SFIRE data
    Ny, Nx = xlat.shape
    min_dlat = np.min(np.diff(xlat, axis=0))
    min_dlon = np.min(np.diff(xlon, axis=1))

    fxlat = np.array(wrf_dataset["FXLAT"][0, grid_slice, grid_slice])
    fxlon = np.array(wrf_dataset["FXLONG"][0, grid_slice, grid_slice])
    Nyf, Nxf = fxlat.shape
    min_dlatf = np.min(np.diff(fxlat, axis=0))
    min_dlonf = np.min(np.diff(fxlon, axis=1))

    station = obs_h5[station_path]
    # Air temperature
    if svn.AIR_TEMPERATURE.value in station.keys():
        print("Process air temperature")
        # Use T2 within interval 0m to 4m
        if _sensor_height_check_ok(
            sensor_height=fs.read_quantity_from_fb_attribute(
                f"{station_path}/{svn.AIR_TEMPERATURE.value}", "sensor_height", obs_h5, dtype=float
            )
            .to("m")
            .magnitude,
            min_height_ok=0,
            max_height_ok=4,
        ):
            print("Nearest T2")
            i_nearest, j_nearest = find_nearest_grid_point(
                Nx,
                Ny,
                obs_h5[station_path].attrs["position_lat"],
                obs_h5[station_path].attrs["position_lon"],
                xlat,
                xlon,
                min_dlat,
                min_dlon,
            )
            sfire_t2_interp = _interp_linear_1d_strict(
                time_sfire[i_start : i_end + 1],
                wrf_dataset["T2"][i_start : i_end + 1, j_nearest, i_nearest],
                time_in_period_sfire,
            )
            var = new_wx.create_dataset(
                svn.AIR_TEMPERATURE.value, data=sfire_t2_interp, **hdf5plugin.Zstd(clevel=compression_lvl)
            )
            var.attrs["units"] = "degK"
            write_model_sensor_height_metadata(var, Quantity(2.0, "m"))
        else:
            print("No method for other height than 2m")

    # Relative Humidity
    if svn.RELATIVE_HUMIDITY.value in station.keys():
        print("Process RH2")
        # Verify RH2
        if _sensor_height_check_ok(
            sensor_height=fs.read_quantity_from_fb_attribute(
                f"{station_path}/{svn.RELATIVE_HUMIDITY.value}", "sensor_height", obs_h5, dtype=float
            )
            .to("m")
            .magnitude,
            min_height_ok=0,
            max_height_ok=4,
        ):
            print("Nearest RH2")
            i_nearest, j_nearest = find_nearest_grid_point(
                Nx,
                Ny,
                obs_h5[station_path].attrs["position_lat"],
                obs_h5[station_path].attrs["position_lon"],
                xlat,
                xlon,
                min_dlat,
                min_dlon,
            )
            t2_raw = wrf_dataset["T2"][i_start : i_end + 1, j_nearest, i_nearest]
            q2_raw = wrf_dataset["Q2"][i_start : i_end + 1, j_nearest, i_nearest]
            psfc_raw = wrf_dataset["PSFC"][i_start : i_end + 1, j_nearest, i_nearest]
            sfire_data_interp = _interp_linear_1d_strict(
                time_sfire[i_start : i_end + 1],
                _rh_from_wrf_q2_t2_psfc(q2_raw, t2_raw, psfc_raw),
                time_in_period_sfire,
            )
            var = new_wx.create_dataset(
                svn.RELATIVE_HUMIDITY.value,
                data=sfire_data_interp,
                **hdf5plugin.Zstd(clevel=compression_lvl),
            )
            var.attrs["units"] = "percent"
            write_model_sensor_height_metadata(var, Quantity(2.0, "m"))
        else:
            print("No method for other height than 2m")

    # Wind direction
    if svn.WIND_DIRECTION.value in station.keys():
        print("Process wind direction")
        # Verify WD10
        wd_sensor_height = (
            fs.read_quantity_from_fb_attribute(
                f"{station_path}/{svn.WIND_DIRECTION.value}", "sensor_height", obs_h5, dtype=float
            )
            .to("m")
            .magnitude
        )
        if _sensor_height_check_ok(
            sensor_height=wd_sensor_height,
            min_height_ok=0,
            max_height_ok=20,
        ):
            print("Nearest WD10 used for WD measured between 0m and 20m")
            i_nearest, j_nearest = find_nearest_grid_point(
                Nx,
                Ny,
                obs_h5[station_path].attrs["position_lat"],
                obs_h5[station_path].attrs["position_lon"],
                xlat,
                xlon,
                min_dlat,
                min_dlon,
            )
            u10_raw = wrf_dataset["U10"][i_start : i_end + 1, j_nearest, i_nearest]
            v10_raw = wrf_dataset["V10"][i_start : i_end + 1, j_nearest, i_nearest]
            cosalpha_raw, sinalpha_raw = _get_wind_rotation(
                wrf_dataset,
                slice(i_start, i_end + 1),
                j_nearest,
                i_nearest,
            )
            u_true = u10_raw * cosalpha_raw - v10_raw * sinalpha_raw
            v_true = u10_raw * sinalpha_raw + v10_raw * cosalpha_raw
            wind_dir_raw = wind_dir_from_uv(u_true, v_true)
            sfire_data_interp = _interp_linear_1d_strict(
                time_sfire[i_start : i_end + 1],
                wind_dir_raw,
                time_in_period_sfire,
            )
            var = new_wx.create_dataset(
                svn.WIND_DIRECTION.value,
                data=sfire_data_interp,
                **hdf5plugin.Zstd(clevel=compression_lvl),
            )
            var.attrs["units"] = "degree"
            # The log profile that scales the wind speed keeps its direction: the 10 m direction
            # is the direction at the sensor height.
            write_model_sensor_height_metadata(var, Quantity(wd_sensor_height, "m"))
        else:
            print("No method for other height than 10m")

    # Wind speed
    if svn.WIND_SPEED.value in station.keys():
        print("Process wind speed")
        sensor_height = (
            fs.read_quantity_from_fb_attribute(
                f"{station_path}/{svn.WIND_SPEED.value}", "sensor_height", obs_h5, dtype=float
            )
            .to("m")
            .magnitude
        )
        i_nearest, j_nearest = find_nearest_grid_point(
            Nx,
            Ny,
            obs_h5[station_path].attrs["position_lat"],
            obs_h5[station_path].attrs["position_lon"],
            xlat,
            xlon,
            min_dlat,
            min_dlon,
        )
        if_nearest, jf_nearest = find_nearest_grid_point(
            Nxf,
            Nyf,
            obs_h5[station_path].attrs["position_lat"],
            obs_h5[station_path].attrs["position_lon"],
            fxlat,
            fxlon,
            min_dlatf,
            min_dlonf,
        )
        print("Nearest WS10 with S10 scaling")
        u10_raw = wrf_dataset["U10"][i_start : i_end + 1, j_nearest, i_nearest]
        v10_raw = wrf_dataset["V10"][i_start : i_end + 1, j_nearest, i_nearest]
        s10 = wind_speed(u10_raw, v10_raw)
        z0 = wrf_dataset["WZ0"][i_start : i_end + 1, jf_nearest, if_nearest]
        ws = wind_from10m_with_roughness(s10, sensor_height, z0)

        sfire_data_interp = _interp_linear_1d_strict(
            time_sfire[i_start : i_end + 1],
            ws,
            time_in_period_sfire,
        )
        var = new_wx.create_dataset(
            svn.WIND_SPEED.value,
            data=sfire_data_interp,
            **hdf5plugin.Zstd(clevel=compression_lvl),
        )
        # For debug
        sfire_data_interp_10 = _interp_linear_1d_strict(
            time_sfire[i_start : i_end + 1],
            s10,
            time_in_period_sfire,
        )
        var.attrs["units"] = "m/s"
        write_model_sensor_height_metadata(var, Quantity(sensor_height, "m"))
        var = new_wx.create_dataset(
            "WS10",
            data=sfire_data_interp_10,
            **hdf5plugin.Zstd(clevel=compression_lvl),
        )
        var.attrs["units"] = "m/s"

    # Fuel Moisture 10h
    if svn.FUEL_MOISTURE_CONTENT_10H.value in station.keys():
        print("Process fmc 10h")
        i_nearest, j_nearest = find_nearest_grid_point(
            Nx,
            Ny,
            obs_h5[station_path].attrs["position_lat"],
            obs_h5[station_path].attrs["position_lon"],
            xlat,
            xlon,
            min_dlat,
            min_dlon,
        )
        if i_fmc_10h is None:
            fmc10h_raw = wrf_dataset["FMC_GC"][i_start : i_end + 1, j_nearest, i_nearest]
        else:
            fmc10h_raw = wrf_dataset["FMC_GC"][i_start : i_end + 1, i_fmc_10h, j_nearest, i_nearest]
        sfire_fmc10h_interp = _interp_linear_1d_strict(
            time_sfire[i_start : i_end + 1],
            fmc10h_raw,
            time_in_period_sfire,
        )
        var = new_wx.create_dataset(
            svn.FUEL_MOISTURE_CONTENT_10H.value,
            data=sfire_fmc10h_interp,
            **hdf5plugin.Zstd(clevel=compression_lvl),
        )
        var.attrs["units"] = "dimensionless"


def get_weather_stations(
    new_h5: h5py.File,
    obs_h5: h5py.File,
    wrf_dataset: Dataset,
    sfire_ref_time: str,
    period_start_dt: datetime,
    period_end_dt: datetime,
    fire_grid_bound: int = 25,
    compression_lvl: int = 3,
    i_fmc_10h: int | None = 1,
):
    """Sample WRF-SFIRE weather variables at observation stations.

    ``i_fmc_10h`` is the zero-based fuel-layer index for 10-hour fuel moisture
    in a 4D ``FMC_GC`` field (time, fuel layer, y, x), defaulting to 1. Set it
    to ``None`` for a 3D field (time, y, x) containing only 10-hour fuel moisture.
    """
    origin = _datetime_fromisoformat_utc(sfire_ref_time)
    for ts in obs_h5[f"{fs.TIME_SERIES}"].keys():
        if ts.startswith("station_"):
            print(f"Process {ts}")
            __process_weather_station(
                new_h5,
                obs_h5,
                wrf_dataset,
                f"{fs.TIME_SERIES}/{ts}",
                origin,
                period_start_dt,
                period_end_dt,
                fire_grid_bound,
                compression_lvl,
                i_fmc_10h,
            )
