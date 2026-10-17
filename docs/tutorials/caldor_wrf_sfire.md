# Convert WRF-SFIRE Output To A FireBench Standard File

This tutorial shows the main workflow for converting WRF-SFIRE output into a FireBench-standardized HDF5 file. The example is based on a Caldor Fire workflow.

The CLI uses one YAML config file to describe the inputs, output path, shared run settings, and enabled processing products.

## Inputs and Output

Caldor gives its observation file, so its config names it as `main.obs_path`. A benchmark case of the
FireBench Foundation site gives a kit instead; see "Enter A FireBench Foundation Benchmark Case".

The workflow uses two inputs:

- a FireBench observation/reference file, for example `/path/to/firebench-data/Caldor.h5`
- a WRF-SFIRE netCDF output file, for example `/path/to/wrf-sfire/wrfout`

It writes a new standardized model-output file, for example `outputs/SFIRE_Caldor.h5`.

## Run the CLI

Install the package locally with development dependencies:

```bash
pip install -e ".[dev]"
```

Run the adapter with:

```bash
fb-sfire run config.yml
```

Relative paths in the config are resolved from the directory containing `config.yml`.

## Example Config

```yaml
main:
  obs_path: /path/to/firebench-data/Caldor.h5
  wrf_path: /path/to/wrf-sfire/run   # the folder of wrfout_dNN_* files, or one wrfout file
  output_path: outputs/SFIRE_Caldor.h5
  authors: "Collaborator Name, Institution;"
  overwrite: true
  compression_lvl: 20
  fire_grid_bound: 25
  sfire_ref_time: "2021-08-21T00:00-00:00"   # optional: SIMULATION_START_DATE of the run by default
  start_dt: "2021-08-21T00:00:00+00:00"
  end_dt: "2021-08-23T00:00:00+00:00"

severity:
  enabled: false
  group_path: /spatial_2d/Caldor_MTBS
  method: fuel_frac
  severity_thresholds: [0.2, 0.6, 0.8]

building_damage:
  enabled: false
  group_path: points/building_damaged
  method: fire_area

ravg_cc:
  enabled: false
  group_path: /spatial_2d/ravg_cc
  method: cfb

perimeters:
  enabled: true
  paths:
    - /polygons/Caldor_2021-08-20T20:20-07:00
    - /polygons/Caldor_2021-08-21T21:15-07:00
  kml_dir: outputs/Cawfe_kml
  extraction_method: tign_g
  smooth_sigma: 10

weather:
  enabled: false
  i_fmc_10h: 1
```

## Processing Sections

Each processing section has an `enabled` flag:

- `severity`: converts WRF-SFIRE `FUEL_FRAC` into an MTBS-style burn severity surface.
- `building_damage`: samples WRF-SFIRE `FIRE_AREA` at building locations.
- `ravg_cc`: converts `FIRE_AREA` and `CFB` into a RAVG canopy-cover-loss surface.
- `burn_severity_cbi`: converts `FUEL_FRAC` into RAVG's four-class composite burn index.
- `perimeters`: extracts fire perimeters from `TIGN_G` and stores KML metadata.
- `weather`: samples and standardizes weather-station variables.

The old script flags map directly to section settings:

- `process_severity` -> `severity.enabled`
- `process_bd` -> `building_damage.enabled`
- `process_ravg_cc` -> `ravg_cc.enabled`
- `process_perims` -> `perimeters.enabled`
- `process_wx` -> `weather.enabled`

## Defaults and Notes

`main.compression_lvl` defaults to `3`, `main.fire_grid_bound` defaults to `25`, and `main.overwrite` defaults to `false`. `perimeters.kml_dir` defaults to `kml_model`, and `perimeters.smooth_sigma` defaults to `0`.

The perimeter workflow requires the perimeter groups in the observation file to have a `time` attribute. The value of `main.sfire_ref_time` must use the same timezone convention as those perimeter times so that `TIGN_G` is converted to the correct elapsed seconds.

`main.start_dt` and `main.end_dt` are required when `weather.enabled` is `true`; include timezone offsets in both values.

`weather.i_fmc_10h` defaults to `1`, the zero-based fuel-layer index for 10-hour fuel moisture in a 4D `FMC_GC` field `(time, fuel layer, y, x)`. Set another integer to select a different layer. If postprocessing adds `FMC_GC` as a 3D field `(time, y, x)` containing only 10-hour fuel moisture, use YAML `null`:

```yaml
weather:
  enabled: true
  i_fmc_10h: null
```

When calling the Python API directly, use `get_weather_stations(..., i_fmc_10h=None)` for the 3D layout; the default remains `i_fmc_10h=1`.
