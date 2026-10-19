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


def canopy_fraction_burnt(domain, grid_slice: slice) -> np.ndarray:
    """The fraction of canopy each fire cell burnt over the run.

    CAN_FRAC_BURNT is WRF-SFIRE's running maximum of CFB while the front crosses a cell; CFB alone is
    the fraction at the time of an output, 0 once the front has passed, so its largest value over the
    output times stands in for a run that does not write CAN_FRAC_BURNT.
    """
    if domain.has("CAN_FRAC_BURNT"):
        return np.array(domain.last("CAN_FRAC_BURNT", grid_slice))
    if domain.has("CFB"):
        return domain.maximum("CFB", grid_slice)
    raise KeyError(f"neither CAN_FRAC_BURNT nor CFB is in the output of {domain.name}")


def get_ravg_cc(
    new_h5: h5py.File,
    obs_h5: h5py.File,
    ravg_group_path: str,
    wrf_dataset: WrfRun | Dataset,
    method: str = "cfb",
    fire_grid_bound: int = 25,
    compression_lvl: int = 3,
):
    """Write RAVG's five canopy cover loss classes on the grid of ``ravg_group_path`` in ``obs_h5``.

    ``obs_h5`` is an observation file or a benchmark kit's template. A cell the fire did not burn
    (FIRE_AREA below 0.5) is 9, outside the burn area; a burnt cell takes the class of its canopy
    fraction burnt (``canopy_fraction_burnt``). A cell of the grid outside every fire grid takes
    the reference's ``_FillValue`` (0, no data). A cell takes the finest fire grid that holds it.
    """
    ref_cc = obs_h5.get(f"{ravg_group_path}/ravg_canopy_cover_loss")
    fill_value = 0 if ref_cc is None else ref_cc.attrs.get("_FillValue", 0)

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
        var.attrs["_FillValue"] = fill_value

    # get SFIRE data
    grid_slice = fire_grid_slice(fire_grid_bound)
    domains = as_run(wrf_dataset).fire_domains()

    # get data from std file
    obs_lat = obs_h5[f"{ravg_group_path}/position_lat"][:, :]
    obs_lon = obs_h5[f"{ravg_group_path}/position_lon"][:, :]
    obs_crs = obs_h5[f"{ravg_group_path}"].attrs["crs"]
    outside = np.ones(obs_lat.shape, dtype=bool)
    for domain in domains:
        outside &= ~domain.covers(obs_lat, obs_lon, grid_slice, fire=True)

    if method == "cfb":

        def canopy_cover_loss(domain):
            fire_area = np.array(domain.last("FIRE_AREA", grid_slice))
            cfb = canopy_fraction_burnt(domain, grid_slice)
            Ny, Nx = fire_area.shape
            return __compute_canopy_cover_loss_from_cfb(Nx, Ny, fire_area, cfb)

        ccl_interp = nearest_by_domain(domains, obs_lat, obs_lon, grid_slice, canopy_cover_loss)
        ccl_interp[outside] = fill_value
        __write_group_new_file(ccl_interp.astype(np.uint8))
        return

    raise NotImplementedError(f"{method} not supported")
