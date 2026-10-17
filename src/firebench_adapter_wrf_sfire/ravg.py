from numba import njit
import numpy as np
import hdf5plugin
import h5py
from netCDF4 import Dataset

from ._grid import fire_grid_slice
from ._wrfout import WrfRun, as_run, nearest_by_domain


@njit(fastmath=True)
def __compute_canopy_cover_loss_from_cfb(Nx, Ny, fire_area, cfb):
    ccl = np.zeros_like(fire_area, dtype=np.uint8)
    for j in range(Ny):
        for i in range(Nx):
            if fire_area[j, i] < 0.5:
                ccl[j, i] = 9  # Outide burn area
            elif cfb[j, i] >= 0.75:
                ccl[j, i] = 5  # 75-100%
            elif cfb[j, i] >= 0.5:
                ccl[j, i] = 4  # 50-<75%
            elif cfb[j, i] >= 0.25:
                ccl[j, i] = 3  # 25-<50%
            elif cfb[j, i] > 0.0:
                ccl[j, i] = 2  # >0-<25%
            else:
                ccl[j, i] = 1  # 0%

    return ccl


def get_ravg_cc(
    new_h5: h5py.File,
    obs_h5: h5py.File,
    ravg_group_path: str,
    wrf_dataset: WrfRun | Dataset,
    method: str = "cfb",
    fire_grid_bound: int = 25,
    compression_lvl: int = 3,
):
    def __write_group_new_file(data: np.ndarray):
        raster_2d = new_h5.create_group(ravg_group_path)
        raster_2d.attrs["crs"] = obs_crs
        var = raster_2d.create_dataset(
            "position_lat", data=obs_lat, **hdf5plugin.Zstd(clevel=compression_lvl)
        )
        var.attrs["units"] = "degree"
        var = raster_2d.create_dataset(
            "position_lon", data=obs_lon, **hdf5plugin.Zstd(clevel=compression_lvl)
        )
        var.attrs["units"] = "degree"
        var = raster_2d.create_dataset(
            "ravg_canopy_cover_loss", data=data, **hdf5plugin.Zstd(clevel=compression_lvl)
        )
        var.attrs["units"] = "dimensionless"
        var.attrs["_FillValue"] = 0

    # get SFIRE data
    grid_slice = fire_grid_slice(fire_grid_bound)
    domains = as_run(wrf_dataset).fire_domains()

    # get data from std file
    obs_lat = obs_h5[f"{ravg_group_path}/position_lat"][:, :]
    obs_lon = obs_h5[f"{ravg_group_path}/position_lon"][:, :]
    obs_crs = obs_h5[f"{ravg_group_path}"].attrs["crs"]

    if method == "cfb":

        def canopy_cover_loss(domain):
            fire_area = np.array(domain.last("FIRE_AREA", grid_slice))
            try:
                cfb = np.array(domain.last("CFB", grid_slice))
            except:
                print("WARNING: CFB not found in wrfout file. Replaced by zeros.")
                cfb = np.zeros_like(fire_area)
            Ny, Nx = fire_area.shape
            return __compute_canopy_cover_loss_from_cfb(Nx, Ny, fire_area, cfb)

        ccl_interp = nearest_by_domain(domains, obs_lat, obs_lon, grid_slice, canopy_cover_loss)
        __write_group_new_file(ccl_interp)
        return

    raise NotImplementedError(f"{method} not supported")
