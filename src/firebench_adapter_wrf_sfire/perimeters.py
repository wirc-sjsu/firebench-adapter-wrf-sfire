import contourpy
import numpy as np
import hdf5plugin
import h5py
from netCDF4 import Dataset
import firebench.standardize as fs
import firebench.tools as ft
from datetime import datetime, timezone
import geopandas as gpd
from pathlib import Path
from os.path import getsize, relpath
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon

from ._skip import skip

PERIMETER_EXTRACTION_METHOD_TIGN_G = "tign_g"
PERIMETER_EXTRACTION_METHOD_BURNED_MASK = "burned_mask"
PERIMETER_EXTRACTION_METHODS = (
    PERIMETER_EXTRACTION_METHOD_TIGN_G,
    PERIMETER_EXTRACTION_METHOD_BURNED_MASK,
)


def _as_utc_datetime(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc)


def _validate_matching_timezone(*datetimes: datetime):
    has_tz = [dt.tzinfo is not None for dt in datetimes]
    if any(has_tz) and not all(has_tz):
        raise ValueError(
            "Cannot mix timezone-aware and timezone-naive datetimes. "
            "Use UTC consistently for t_ign_g_ref_time and perimeter times."
        )


def _valid_tign_g_values(t_ign_g: np.ndarray) -> np.ndarray:
    masked_values = np.ma.masked_invalid(t_ign_g)
    return np.asarray(masked_values.compressed(), dtype=np.float64)


def _valid_tign_g_mask_and_values(t_ign_g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masked_values = np.ma.masked_invalid(t_ign_g)
    values = np.asarray(masked_values.filled(np.nan), dtype=np.float64)
    valid_mask = ~np.ma.getmaskarray(masked_values) & np.isfinite(values)
    return valid_mask, values


def _fire_grid_slice(fire_grid_bound: int) -> slice:
    if fire_grid_bound < 0:
        raise ValueError("fire_grid_bound must be >= 0")
    return slice(fire_grid_bound, -fire_grid_bound) if fire_grid_bound else slice(None)


def _validate_kml_has_polygons(kml_path: Path):
    perimeters = gpd.read_file(kml_path, driver="KML")
    if perimeters.empty:
        raise ValueError(
            f"KML perimeter file contains no polygons: {kml_path}. "
            "Delete it or rerun with overwrite=True to recompute it."
        )


def _perimeter_extraction_summary(
    t_ign_g: np.ndarray,
    xlon: np.ndarray,
    xlat: np.ndarray,
    time_s: float,
) -> str:
    finite, values = _valid_tign_g_mask_and_values(t_ign_g)
    burned = finite & (values <= time_s)
    finite_count = int(np.sum(finite))
    burned_count = int(np.sum(burned))
    boundary_count = 0
    if burned_count:
        boundary = np.zeros_like(burned, dtype=bool)
        boundary[0, :] = True
        boundary[-1, :] = True
        boundary[:, 0] = True
        boundary[:, -1] = True
        boundary_count = int(np.sum(burned & boundary))

    finite_values = values[finite]
    percentiles = np.percentile(finite_values, [1, 5, 50, 95, 99]) if finite_count else []
    percentile_text = ", ".join(f"{value:.1f}" for value in percentiles)
    return (
        f"finite_cells={finite_count}, cells_le_iso={burned_count}, "
        f"boundary_cells_le_iso={boundary_count}, "
        f"TIGN_G_percentiles_1_5_50_95_99=[{percentile_text}], "
        f"lon_range=[{float(np.nanmin(xlon))}, {float(np.nanmax(xlon))}], "
        f"lat_range=[{float(np.nanmin(xlat))}, {float(np.nanmax(xlat))}]"
    )


def _burned_mask_field(t_ign_g: np.ndarray, time_s: float) -> np.ndarray:
    finite, values = _valid_tign_g_mask_and_values(t_ign_g)
    return (finite & (values <= time_s)).astype(float)


def _line_contour_to_geopolygons(
    field: np.ndarray,
    xlon: np.ndarray,
    xlat: np.ndarray,
    iso_value: float,
    smooth_sigma: float = 0,
    original_crs: str | dict = "EPSG:4326",
) -> gpd.GeoDataFrame:
    if not xlon.shape == xlat.shape == field.shape:
        raise ValueError("Input arrays (lon, lat, field) must have the same shape.")

    field = np.ma.masked_invalid(field)
    if smooth_sigma > 0:
        field = gaussian_filter(field.filled(np.nan), sigma=smooth_sigma)
        field = np.ma.masked_invalid(field)

    contour_generator = contourpy.contour_generator(
        x=xlon,
        y=xlat,
        z=field,
        line_type=contourpy.LineType.Separate,
        corner_mask=True,
    )
    rings = [ring for ring in contour_generator.lines(iso_value) if len(ring) >= 4]

    polygons = []
    polygon_rings = []
    for ring in rings:
        polygon = Polygon(ring)
        if polygon.is_valid and polygon.area > 0:
            polygons.append(polygon)
            polygon_rings.append(ring)

    sorted_indices = sorted(range(len(polygons)), key=lambda index: polygons[index].area, reverse=True)
    used_as_hole = set()
    final_polygons = []

    for index in sorted_indices:
        if index in used_as_hole:
            continue
        outer = polygons[index]
        holes = []
        for inner_index in sorted_indices:
            if index == inner_index or inner_index in used_as_hole:
                continue
            inner = polygons[inner_index]
            if inner.within(outer):
                if any(inner.within(Polygon(hole)) for hole in holes):
                    continue
                holes.append(polygon_rings[inner_index])
                used_as_hole.add(inner_index)

        final_polygon = Polygon(polygon_rings[index], holes=holes)
        if final_polygon.is_valid and final_polygon.area > 0:
            final_polygons.append(final_polygon)

    return gpd.GeoDataFrame(geometry=final_polygons, crs=original_crs).assign(iso_value=iso_value)


def _extract_burned_mask_geopolygons(
    t_ign_g: np.ndarray,
    xlon: np.ndarray,
    xlat: np.ndarray,
    time_s: float,
    smooth_sigma: float,
) -> gpd.GeoDataFrame:
    burned_field = _burned_mask_field(t_ign_g, time_s)
    perimeters = _line_contour_to_geopolygons(burned_field, xlon, xlat, 0.5, smooth_sigma=smooth_sigma)
    if not perimeters.empty or not smooth_sigma:
        return perimeters

    print(
        "No smoothed burned-mask contour polygon found; retrying burned-mask "
        "contour extraction without smoothing."
    )
    return _line_contour_to_geopolygons(burned_field, xlon, xlat, 0.5, smooth_sigma=0)


def _extract_perimeter_geopolygons(
    t_ign_g: np.ndarray,
    xlon: np.ndarray,
    xlat: np.ndarray,
    time_s: float,
    extraction_method: str,
    smooth_sigma: float,
) -> gpd.GeoDataFrame:
    if extraction_method == PERIMETER_EXTRACTION_METHOD_TIGN_G:
        perimeters = _line_contour_to_geopolygons(t_ign_g, xlon, xlat, time_s, smooth_sigma=smooth_sigma)
        if not perimeters.empty:
            return perimeters

        print("No raw TIGN_G contour polygon found; falling back to burned-mask " "contour extraction.")
        return _extract_burned_mask_geopolygons(t_ign_g, xlon, xlat, time_s, smooth_sigma)

    if extraction_method == PERIMETER_EXTRACTION_METHOD_BURNED_MASK:
        return _extract_burned_mask_geopolygons(t_ign_g, xlon, xlat, time_s, smooth_sigma)

    raise ValueError(
        f"Unknown perimeter extraction method {extraction_method!r}. "
        f"Available methods: {', '.join(PERIMETER_EXTRACTION_METHODS)}"
    )


def get_perimeters_from_t_ign_g(
    new_h5: h5py.File,
    obs_h5: h5py.File,
    wrf_dataset: Dataset,
    list_perimeter_paths: list[str],
    t_ign_g_ref_time: str,
    fire_grid_bound: int = 25,
    kml_dir: str = "kml",
    overwrite: bool = False,
    extraction_method: str = PERIMETER_EXTRACTION_METHOD_TIGN_G,
    smooth_sigma: float = 0,
):
    if extraction_method not in PERIMETER_EXTRACTION_METHODS:
        raise ValueError(
            f"Unknown perimeter extraction method {extraction_method!r}. "
            f"Available methods: {', '.join(PERIMETER_EXTRACTION_METHODS)}"
        )
    Path(kml_dir).mkdir(parents=True, exist_ok=True)
    origin_raw = datetime.fromisoformat(t_ign_g_ref_time.replace("Z", "+00:00"))
    origin = _as_utc_datetime(origin_raw)
    grid_slice = _fire_grid_slice(fire_grid_bound)

    xlat = np.array(wrf_dataset["FXLAT"][0, grid_slice, grid_slice])
    xlon = np.array(wrf_dataset["FXLONG"][0, grid_slice, grid_slice])
    t_ign_g = wrf_dataset["TIGN_G"][-1, grid_slice, grid_slice]
    valid_t_ign_g = _valid_tign_g_values(t_ign_g)
    if valid_t_ign_g.size == 0:
        raise ValueError("TIGN_G contains no finite values in the selected fire grid.")
    min_t_ign_g = float(np.min(valid_t_ign_g))
    max_t_ign_g = float(np.max(valid_t_ign_g))
    # TIGN_G is read at the last output time, the horizon of the simulation (XTIME, in minutes).
    horizon_s = float(wrf_dataset["XTIME"][-1]) * 60 if "XTIME" in wrf_dataset.variables else None

    for perim_path in list_perimeter_paths:
        try:
            _write_perimeter(
                new_h5,
                obs_h5,
                perim_path,
                t_ign_g,
                xlon,
                xlat,
                origin,
                origin_raw,
                (min_t_ign_g, max_t_ign_g),
                horizon_s,
                kml_dir,
                overwrite,
                extraction_method,
                smooth_sigma,
            )
        except (KeyError, ValueError) as exc:
            skip(perim_path, exc)


def _write_perimeter(  # pylint: disable=too-many-arguments,too-many-locals
    new_h5: h5py.File,
    obs_h5: h5py.File,
    perim_path: str,
    t_ign_g: np.ndarray,
    xlon: np.ndarray,
    xlat: np.ndarray,
    origin: datetime,
    origin_raw: datetime,
    t_ign_g_range: tuple[float, float],
    horizon_s: float | None,
    kml_dir: str,
    overwrite: bool,
    extraction_method: str,
    smooth_sigma: float,
):
    min_t_ign_g, max_t_ign_g = t_ign_g_range
    obs_datetime_raw = datetime.fromisoformat(obs_h5[perim_path].attrs["time"].replace("Z", "+00:00"))
    obs_datetime = _as_utc_datetime(obs_datetime_raw)
    _validate_matching_timezone(origin_raw, obs_datetime_raw)
    time_s = (obs_datetime - origin).total_seconds()
    if horizon_s is not None and time_s > horizon_s:
        raise ValueError(
            f"its time {time_s} s is after the end of the simulation ({horizon_s} s after "
            "t_ign_g_ref_time)."
        )
    print(
        f"Perimeter {perim_path} after {time_s} s "
        f"(TIGN_G finite range: {min_t_ign_g} to {max_t_ign_g} s, "
        f"method={extraction_method}, smooth_sigma={smooth_sigma})"
    )
    sfire_kml_path = Path(
        f"{kml_dir}",
        f"SFIRE_{fs.sanitize_iso8601(fs.datetime_to_iso8601(obs_datetime_raw))}.kml",
    )
    sfire_kml_filename = sfire_kml_path.resolve()
    if not sfire_kml_filename.exists() or overwrite:
        if time_s < min_t_ign_g or time_s > max_t_ign_g:
            raise ValueError(
                f"Perimeter time {time_s} s for {perim_path} is outside the finite TIGN_G range "
                f"[{min_t_ign_g}, {max_t_ign_g}] s. Check t_ign_g_ref_time and timezone synchronization."
            )
        perim_sfire = _extract_perimeter_geopolygons(
            t_ign_g, xlon, xlat, time_s, extraction_method, smooth_sigma
        )
        if perim_sfire.empty:
            extraction_summary = _perimeter_extraction_summary(t_ign_g, xlon, xlat, time_s)
            raise ValueError(
                f"No polygon extracted for {perim_path} at {time_s} s. "
                f"method={extraction_method}, smooth_sigma={smooth_sigma}. "
                f"{extraction_summary}. "
                "Check t_ign_g_ref_time, timezone synchronization, fire_grid_bound, and TIGN_G units."
            )
        perim_sfire.to_file(sfire_kml_filename, driver="KML", layer="fire_perimeter")
    else:
        print(
            "KML file already found in directory and reused. Use overwrite = True to recompute perimeter."
        )
        _validate_kml_has_polygons(sfire_kml_filename)

    grp = new_h5.create_group(perim_path)
    grp.attrs["time"] = fs.datetime_to_iso8601(obs_datetime_raw)
    h5_parent = Path(new_h5.filename).resolve().parent
    grp.attrs["rel_path"] = Path(relpath(sfire_kml_filename, start=h5_parent)).as_posix()
    grp.attrs["sha256"] = ft.calculate_sha256(sfire_kml_filename)
    grp.attrs["file_size_bytes"] = getsize(sfire_kml_filename)
