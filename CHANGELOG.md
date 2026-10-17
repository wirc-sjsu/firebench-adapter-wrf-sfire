# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Read a FireBench Foundation benchmark kit: `main.template_path` names the kit's model-output template (root attribute `firebench_template`) in place of `main.obs_path`. Giving both is refused.
- Check the optional `case.definition_sha256` against the case definition the template holds, and stop when the config was made for another version of the case.
- Add the `burn_severity_cbi` section and `get_ravg_cbi_severity`: RAVG four-class composite burn index (`ravg_composite_burn_index_severity`) from `FUEL_FRAC`, on the reference grid, with the reference's dtype, `units`, and `_FillValue`; cells outside the fire grid take the `_FillValue`.
- Record the sensor height each weather variable was prepared at (`sensor_height`, `sensor_height_units`): 2 m for air temperature and relative humidity, the station's height for wind speed and wind direction.
- Add the tutorial "Enter a FireBench Foundation benchmark case".

### Changed
- Write what the simulation can give and skip the rest with a `SkippedWarning`, so one config serves several simulations of a case: a perimeter before ignition, after the last output time (`XTIME`), or without a polygon; a weather station outside the atmospheric grid (instead of sampling the nearest edge cell) or without an observation time inside the simulation; a variable whose WRF-SFIRE fields or sensor height are missing; and a whole section that fails on the data. `fb-sfire run` prints each one as `WARNING:`, the count at the end, and exits 0. An unknown method still stops the run, before the output is created.
- Keep in a station's series only the observation times the simulation covers, instead of stopping when the simulation does not cover the whole window.

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
