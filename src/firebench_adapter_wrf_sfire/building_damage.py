import numpy as np
import hdf5plugin
import h5py
from netCDF4 import Dataset
from firebench.adapter_common.interpolation import find_nearest_grid_point


def get_building_damage(
    new_h5: h5py.File,
    obs_h5: h5py.File,
    bd_group_path: str,
    wrf_dataset: Dataset,
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
    xlat = np.array(
        wrf_dataset["FXLAT"][0, fire_grid_bound:-fire_grid_bound, fire_grid_bound:-fire_grid_bound]
    )
    xlon = np.array(
        wrf_dataset["FXLONG"][0, fire_grid_bound:-fire_grid_bound, fire_grid_bound:-fire_grid_bound]
    )
    Ny, Nx = xlat.shape
    min_dlat = np.min(np.diff(xlat, axis=0))
    min_dlon = np.min(np.diff(xlon, axis=1))

    building_lat = obs_h5[f"{bd_group_path}/position_lat"][:]
    building_lon = obs_h5[f"{bd_group_path}/position_lon"][:]
    building_damaged = np.array(obs_h5[f"{bd_group_path}/building_damage"][:], dtype="U")
    N_buildings = len(building_lat)

    if method == "fire_area":
        fire_area = wrf_dataset["FIRE_AREA"][
            -1, fire_grid_bound:-fire_grid_bound, fire_grid_bound:-fire_grid_bound
        ]
        sfire_building_damaged = []
        for k in range(N_buildings):
            print(f"{k+1}/{N_buildings}")
            if building_damaged[k] == "Inaccessible":
                sfire_building_damaged.append("Inaccessible")
                continue

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
