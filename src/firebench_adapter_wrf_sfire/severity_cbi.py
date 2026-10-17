from numba import njit
import numpy as np
import hdf5plugin
import h5py
from netCDF4 import Dataset

from ._grid import fire_grid_slice
from ._wrfout import WrfRun, as_run, nearest_by_domain

CBI_DATASET = "ravg_composite_burn_index_severity"

# RAVG four-class composite burn index (CBI4) from the fraction of fuel left (FUEL_FRAC)
# <= CBI_TH[0]: high (4)
# <= CBI_TH[1]: moderate (3)
# <= CBI_TH[2]: low (2)
# > CBI_TH[2]: unchanged (1)
# The bounds are those of the MTBS section (severity_mtbs.py), on RAVG's classes.
CBI_TH_DEFAULT = np.array([0.2, 0.6, 0.8], dtype=np.float64)


@njit(fastmath=True)
def __compute_cbi4_from_fuel_frac(Nx: int, Ny: int, fuel_frac: np.ndarray, cbi_thresholds: np.ndarray):
    cbi4 = np.zeros((Ny, Nx), dtype=np.uint8)
    for j in range(Ny):
        for i in range(Nx):
            if fuel_frac[j, i] <= cbi_thresholds[0]:
                cbi4[j, i] = 4
            elif fuel_frac[j, i] <= cbi_thresholds[1]:
                cbi4[j, i] = 3
            elif fuel_frac[j, i] <= cbi_thresholds[2]:
                cbi4[j, i] = 2
            else:
                cbi4[j, i] = 1

    return cbi4


def get_ravg_cbi_severity(
    new_h5: h5py.File,
    ref_h5: h5py.File,
    cbi_group_path: str,
    wrf_dataset: WrfRun | Dataset,
    method: str = "fuel_frac",
    cbi_thresholds: np.ndarray = CBI_TH_DEFAULT,
    fire_grid_bound: int = 25,
    compression_lvl: int = 3,
):
    """Write RAVG CBI4 classes on the grid of ``cbi_group_path`` in ``ref_h5``.

    ``ref_h5`` is an observation file or a benchmark kit's template. A cell of its grid outside the
    fire grids takes the reference's ``_FillValue``, which FireBench leaves out. A cell takes the finest
    fire grid that holds it.
    """
    ref_cbi = ref_h5[f"{cbi_group_path}/{CBI_DATASET}"]
    fill_value = ref_cbi.attrs["_FillValue"]

    def __write_group_new_file(data: np.ndarray):
        raster_2d = new_h5.create_group(cbi_group_path)
        raster_2d.attrs["crs"] = ref_crs
        var = raster_2d.create_dataset(
            "position_lat", data=ref_lat, **hdf5plugin.Zstd(clevel=compression_lvl)
        )
        var.attrs["units"] = "degree"
        var = raster_2d.create_dataset(
            "position_lon", data=ref_lon, **hdf5plugin.Zstd(clevel=compression_lvl)
        )
        var.attrs["units"] = "degree"
        var = raster_2d.create_dataset(
            CBI_DATASET, data=data.astype(ref_cbi.dtype), **hdf5plugin.Zstd(clevel=compression_lvl)
        )
        var.attrs["units"] = ref_cbi.attrs.get("units", "dimensionless")
        var.attrs["_FillValue"] = fill_value

    # get SFIRE data
    grid_slice = fire_grid_slice(fire_grid_bound)
    domains = as_run(wrf_dataset).fire_domains()

    # get grid from reference file
    ref_lat = ref_h5[f"{cbi_group_path}/position_lat"][:, :]
    ref_lon = ref_h5[f"{cbi_group_path}/position_lon"][:, :]
    ref_crs = ref_h5[f"{cbi_group_path}"].attrs["crs"]
    outside = np.ones(ref_lat.shape, dtype=bool)
    for domain in domains:
        outside &= ~domain.covers(ref_lat, ref_lon, grid_slice, fire=True)

    if method == "fuel_frac":

        def cbi4(domain):
            fuel_frac = np.array(domain.last("FUEL_FRAC", grid_slice))
            Ny, Nx = fuel_frac.shape
            return __compute_cbi4_from_fuel_frac(Nx, Ny, fuel_frac, cbi_thresholds)

        cbi4_interp = nearest_by_domain(domains, ref_lat, ref_lon, grid_slice, cbi4)
        cbi4_interp[outside] = fill_value
        __write_group_new_file(cbi4_interp)
        return

    raise NotImplementedError(f"{method} not supported")
