from numba import njit
import numpy as np
import hdf5plugin
import h5py
from netCDF4 import Dataset
from firebench.adapter_common.interpolation import nearest_2d

# Threshold for burn severity
# <= SEVERITY_TH[0]: high (4)
# <= SEVERITY_TH[1]: moderate (3)
# <= SEVERITY_TH[2]: low (2)
# > SEVERITY_TH[2]: unburnt to low (1)
SEVERITY_TH_DEFAULT = np.array([0.2, 0.6, 0.8], dtype=np.float64)


@njit(fastmath=True)
def __compute_severity_from_fuel_frac(
    Nx: int, Ny: int, fuel_frac: np.ndarray, severity_thresholds: np.ndarray
):
    severity = np.zeros((Ny, Nx))
    for j in range(Ny):
        for i in range(Nx):
            if fuel_frac[j, i] <= severity_thresholds[0]:
                severity[j, i] = 4
            elif fuel_frac[j, i] <= severity_thresholds[1]:
                severity[j, i] = 3
            elif fuel_frac[j, i] <= severity_thresholds[2]:
                severity[j, i] = 2
            else:
                severity[j, i] = 1

    return severity


def get_mtbs_severity_surf(
    new_h5: h5py.File,
    obs_h5: h5py.File,
    mtbs_group_path: str,
    wrf_dataset: Dataset,
    method: str = "fuel_frac",
    severity_thresholds: np.ndarray = SEVERITY_TH_DEFAULT,
    fire_grid_bound: int = 25,
    compression_lvl: int = 3,
):
    def __write_group_new_file(severity_data: np.ndarray):
        raster_2d = new_h5.create_group(mtbs_group_path)
        raster_2d.attrs["crs"] = mtbs_crs
        var = raster_2d.create_dataset(
            "position_lat", data=mtbs_lat, **hdf5plugin.Zstd(clevel=compression_lvl)
        )
        var.attrs["units"] = "degree"
        var = raster_2d.create_dataset(
            "position_lon", data=mtbs_lon, **hdf5plugin.Zstd(clevel=compression_lvl)
        )
        var.attrs["units"] = "degree"
        var = raster_2d.create_dataset(
            "fire_burn_severity", data=severity_data, **hdf5plugin.Zstd(clevel=compression_lvl)
        )
        var.attrs["units"] = "dimensionless"
        var.attrs["_FillValue"] = 0

    # get SFIRE data
    xlat = np.array(
        wrf_dataset["FXLAT"][0, fire_grid_bound:-fire_grid_bound, fire_grid_bound:-fire_grid_bound]
    )
    xlon = np.array(
        wrf_dataset["FXLONG"][0, fire_grid_bound:-fire_grid_bound, fire_grid_bound:-fire_grid_bound]
    )

    Ny, Nx = xlat.shape
    # get data from std file
    mtbs_lat = obs_h5[f"{mtbs_group_path}/position_lat"][:, :]
    mtbs_lon = obs_h5[f"{mtbs_group_path}/position_lon"][:, :]
    mtbs_crs = obs_h5[f"{mtbs_group_path}"].attrs["crs"]

    if method == "fuel_frac":
        fuel_frac = wrf_dataset["FUEL_FRAC"][
            -1, fire_grid_bound:-fire_grid_bound, fire_grid_bound:-fire_grid_bound
        ]
        SFIRE_severity = __compute_severity_from_fuel_frac(Nx, Ny, fuel_frac, severity_thresholds)
        severity_interp = nearest_2d(SFIRE_severity, xlon, xlat, mtbs_lon, mtbs_lat)
        __write_group_new_file(severity_interp)
        return

    raise NotImplementedError(f"{method} not supported")
