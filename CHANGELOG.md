# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Add `fb-sfire inspect CONFIG` and `inspect_plume`: what the run gives the `plume` section, reading no 3D field (each domain's `tr17` tracers with their dimensions and units, whether the chosen tracer and the other fields are there, its output interval; the domain that gives each cross section and frame, how many radar times the run covers, and the size of the output), and that a template without radar targets must be downloaded again.
- Over an `echo_top` radar's frame, the `plume` section counts only the cells the radar can see (between the lowest and highest beam over their map cell at that time, from the template) and writes `plume_top`, the highest plume cell of the columns under each cell of the radar's map grid (FireBench's `column_tops` and `tops_on_map`, columns of every domain at once); the points for figures are those cells.
- Add the `plume` section and `get_plume`: the run's plume at the radar's times and places, for FireBench's `radar_plume` family. One WRF-SFIRE fire tracer, `plume.tracer` (`tr17_1` to `tr17_8`, `tr17_1` by default; a run without it skips the section and names the tracers it has), and the earth-relative wind (destaggered, turned with `COSALPHA` and `SINALPHA`) on the nodes of each radar cross section at the start of each radar volume, and the count of the model's plume cells over each `echo_top` radar's frame in bins of height, as the case's template (or observation file) gives them (`firebench.benchmarks.fire_case_template.plume_targets`). Every domain that holds the tracer gives the targets, each position (a node of a section, a column of a frame) from the finest one that covers it, so the plume is followed past the nests, and a frame given by several domains counts each cell for its area; `plume.domain` takes every position from one domain. Each domain is read cut to the cells around what it gives; a radar time between two output times takes both, linearly in time, through FireBench's `ColumnSampler` and `plume_cell_counts`, and a time outside the run or between outputs more than `plume.max_time_gap` s apart (1800 by default) stays NaN. `plume.points` keeps some plume cells for figures. `WrfDomain.volume` and `WrfDomain.box` read a 3D field and a cut of a 2D one.
- Add the `frp` section and `get_fire_total`: the fire radiative power band of the run over a frame, at every output time, for FireBench's `fire_radiative_power` family (`/time_series/fire_total`). The sensible heat released over the frame is the heat flux of the surface and the canopy fire on the fire grid (`FGRNHFX` + `FCANHFX`; a run without both skips the section) times the area of each fire cell inside the frame, each cell from the finest fire domain that holds it; `fire_radiative_power_lower` and `fire_radiative_power_upper` are that heat divided by the largest and smallest `frp.conversion_factor` (`[5, 10]` by default: the heat flux is 5 to 10 times the FRP), with `heat_release_rate_sensible` and the conversion as attributes. The frame is the template's, or `frp.frame`.
- Read a FireBench Foundation benchmark kit: `main.template_path` names the kit's model-output template (root attribute `firebench_template`) in place of `main.obs_path`. Giving both is refused.
- Check the optional `case.definition_sha256` against the case definition the template holds, and stop when the config was made for another version of the case.
- Add the `burn_severity_cbi` section and `get_ravg_cbi_severity`: RAVG four-class composite burn index (`ravg_composite_burn_index_severity`) from `FUEL_FRAC`, on the reference grid, with the reference's dtype, `units`, and `_FillValue`; cells outside the fire grid take the `_FillValue`.
- Record the sensor height each weather variable was prepared at (`sensor_height`, `sensor_height_units`): 2 m for air temperature and relative humidity, the station's height for wind speed and wind direction.
- Add the tutorial "Enter a FireBench Foundation benchmark case".
- Read a whole run: `main.wrf_path` may name the folder of the run, and the adapter reads every `wrfout_dNN_*` file in it, each domain as one time series across its files (a time a file gives again after a restart is left out). One file still works as before.
- Several domains: a weather station takes each variable from the finest domain that holds it and gives the variable, the next domain otherwise; the raster and building damage sections take each point from the finest domain with a fire grid that holds it; perimeters come from the finest domain with a fire grid. `fb-sfire run` prints the domains it found.
- Wind speed on a domain without the fire grid roughness `WZ0` uses the atmospheric roughness `ZNT` in the same log profile.
- `main.sfire_ref_time` is optional: it defaults to the run's `SIMULATION_START_DATE` (UTC). The Python API takes `None` for it too.
- `fb-sfire run -o` (`--overwrite`) replaces an existing output and recomputes its KML perimeters, as `main.overwrite: true`; the error for an existing output says so.

### Changed
- `ravg_cc` reads `CFB` at the last output time, the crown fraction the fire burnt, which WRF-SFIRE stops updating once a burnt cell's canopy is past its age cutoff. A run without `CFB` skips the section, instead of writing every burnt cell as class 1 (0 % loss). A cell of the reference grid outside every fire grid is the reference's `_FillValue` (0, no data), which FireBench does not score, instead of the nearest edge cell.
- Write what the simulation can give and skip the rest with a `SkippedWarning`, so one config serves several simulations of a case: a perimeter before ignition, after the last output time (`XTIME`), or without a polygon; a weather station outside the atmospheric grid (instead of sampling the nearest edge cell) or without an observation time inside the simulation; a variable whose WRF-SFIRE fields or sensor height are missing; and a whole section that fails on the data. `fb-sfire run` prints each one as `WARNING:`, the count at the end, and exits 0. An unknown method still stops the run, before the output is created.
- Keep in a station's series only the observation times the simulation covers, instead of stopping when the simulation does not cover the whole window.
- A point is inside a domain when it is inside the boundary of its (trimmed) grid, instead of inside the latitude and longitude bounds, which hold points beyond the corners of a projected grid. This applies to weather stations and to the cells of `burn_severity_cbi`.
- The public functions take a run (`open_run`) or one open wrfout `Dataset`.
- Reduced outputs (variables left out by the IO config): a missing variable is named in the warning (`FUEL_FRAC is not in the output of d02`), and a variable with other dimensions than expected is skipped with them, instead of failing on the netCDF read.
- A 3D `FMC_GC` (the 10-hour class alone) is told from a 4D one by its dimensions; `weather.i_fmc_10h: null` is no longer needed for it.

### Fixed
- A weather station whose simulation covers a single observation time, at an output time of the run, gets the model's value there instead of NaN (and scipy's `invalid value encountered in divide`).
- Python 3.10 and 3.11: reading a run no longer stops with `TypeError: unhashable type: 'slice'`.
- Weather stations with `fire_grid_bound` above 0 read the grid cell nearest to them: the nearest point was found on the trimmed grid and its indices read on the full one, `fire_grid_bound` cells away (atmospheric variables and `WZ0`).

## [0.1-dev2] - 2026-10-13

### Added
- Add configurable 10-hour fuel-moisture layer selection through `i_fmc_10h` in the Python API and `weather.i_fmc_10h` in YAML configuration, defaulting to layer 1.
- Support postprocessed 3D `FMC_GC` fields containing only 10-hour fuel moisture with Python `None` or YAML `null`.

## [0.1-dev1] - 2026-08-07

### Added
- Add the YAML-driven `fb-sfire run` CLI with validation and configurable processing sections.
- Add regression coverage for CLI workflows and relative perimeter asset references.

### Changed
- Resolve relative input, output, and KML paths from the directory containing the YAML configuration.
- Update the Caldor tutorial to document the CLI workflow and v2026.2 benchmark paths.
- Allow adapter functions to reuse an open WRF netCDF dataset across processing sections.

### Fixed
- Use `/polygons` for Caldor v2026.2 perimeter groups instead of the obsolete `/geopolygons` path.
- Store generated KML references relative to the output HDF5 file, as required by FireBench integrity checks.

## [0.1-dev0] - 2026-07-07

### Added
- Add the `firebench-adapter-wrf-sfire` package for converting native WRF-SFIRE netCDF outputs into FireBench-standardized datasets.
- Add building damage processing from WRF-SFIRE `FIRE_AREA` outputs using nearest-grid lookup.
- Add perimeter generation from `TIGN_G`, including KML output reuse, configurable KML directories, relative output paths, file size metadata, and SHA-256 metadata.
- Add RAVG canopy-cover-loss processing from `FIRE_AREA` and `CFB`, with nearest-neighbor interpolation onto the observation grid.
- Add MTBS burn-severity processing from `FUEL_FRAC`, with configurable severity thresholds and nearest-neighbor interpolation.
- Add weather-station processing for air temperature, relative humidity, wind direction, wind speed, and 10-hour fuel moisture content.
- Add weather-station time-window filtering, SFIRE time bracketing, sensor-height checks, and wind-speed extrapolation using surface roughness.
- Add package metadata, developer dependencies, Makefile targets, pylint support, and README installation notes.

### Changed
- Depend on FireBench directly for shared adapter utilities through `firebench.adapter_common`.
- Use compressed HDF5 datasets for generated FireBench-standardized outputs.
- Export adapter entry points from `firebench_adapter_wrf_sfire.__init__`.

### Fixed
- Preserve weather-station time origins when writing filtered station output.
- Use the atmospheric grid for weather variables and include the end index when slicing model time brackets.
- Store generated perimeter paths in POSIX form and avoid absolute paths in FireBench metadata.
- Reuse existing perimeter KML files unless `overwrite=True`.

[Unreleased]: https://github.com/wirc-sjsu/firebench-adapter-wrf-sfire/compare/0.1-dev2...HEAD
[0.1-dev2]: https://github.com/wirc-sjsu/firebench-adapter-wrf-sfire/compare/0.1-dev1...0.1-dev2
[0.1-dev1]: https://github.com/wirc-sjsu/firebench-adapter-wrf-sfire/compare/0.1-dev0...0.1-dev1
[0.1-dev0]: https://github.com/wirc-sjsu/firebench-adapter-wrf-sfire/releases/tag/0.1-dev0
