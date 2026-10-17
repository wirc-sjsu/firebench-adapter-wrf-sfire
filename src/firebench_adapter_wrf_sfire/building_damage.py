import numpy as np
import hdf5plugin
import h5py
from netCDF4 import Dataset
from firebench.adapter_common.interpolation import find_nearest_grid_point

from ._grid import fire_grid_slice
from ._wrfout import WrfRun, as_run, point_owners


def get_building_damage(
    new_h5: h5py.File,
    obs_h5: h5py.File,
    bd_group_path: str,
    wrf_dataset: WrfRun | Dataset,
    method: str = "fire_area",
    fire_grid_bound: int = 25,
    compression_lvl: int = 3,
):
    def __write_group_new_file(data: np.ndarray):
        bd = new_h5.create_group(bd_group_path)
        var = bd.create_dataset(
            "position_lat", data=building_lat, **hdf5plugin.Zstd(clevel=compression_lvl)
        )
        var.attrs["units"] = "degree"
        var = bd.create_dataset(
            "position_lon", data=building_lon, **hdf5plugin.Zstd(clevel=compression_lvl)
        )
        var.attrs["units"] = "degree"
        var = bd.create_dataset(
            "building_damage",
            data=np.array(data, dtype="S"),
            dtype="S16",
            **hdf5plugin.Zstd(clevel=compression_lvl),
        )
        var.attrs["units"] = "dimensionless"

    # get SFIRE data
    grid_slice = fire_grid_slice(fire_grid_bound)
    domains = as_run(wrf_dataset).fire_domains()
    if not domains:
        raise ValueError("no domain of the run has a fire grid")

    building_lat = obs_h5[f"{bd_group_path}/position_lat"][:]
    building_lon = obs_h5[f"{bd_group_path}/position_lon"][:]
    building_damaged = np.array(obs_h5[f"{bd_group_path}/building_damage"][:], dtype="U")
    N_buildings = len(building_lat)
    # A building takes the finest fire grid that holds it.
    owners = point_owners(domains, building_lat, building_lon, grid_slice, fire=True)

    if method == "fire_area":
        grids = {}

        def grid(index: int):
            if index not in grids:
                domain = domains[index]
                xlat = domain.static("FXLAT", grid_slice)
                xlon = domain.static("FXLONG", grid_slice)
                grids[index] = (
                    xlat,
                    xlon,
                    np.min(np.diff(xlat, axis=0)),
                    np.min(np.diff(xlon, axis=1)),
                    np.array(domain.last("FIRE_AREA", grid_slice)),
                )
            return grids[index]

        sfire_building_damaged = []
        for k in range(N_buildings):
            print(f"{k+1}/{N_buildings}")
            if building_damaged[k] == "Inaccessible":
                sfire_building_damaged.append("Inaccessible")
                continue

            xlat, xlon, min_dlat, min_dlon, fire_area = grid(owners[k])
            Ny, Nx = xlat.shape
            i_nearest, j_nearest = find_nearest_grid_point(
                Nx, Ny, building_lat[k], building_lon[k], xlat, xlon, min_dlat, min_dlon
            )

            if fire_area[j_nearest, i_nearest] >= 0.5:
                sfire_building_damaged.append("Destroyed (>50%)")
            else:
                sfire_building_damaged.append("No Damage")

        __write_group_new_file(sfire_building_damaged)
        return

    raise NotImplementedError(f"{method} not supported")
