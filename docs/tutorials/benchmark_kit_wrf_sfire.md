# Enter A FireBench Foundation Benchmark Case

A benchmark case of the FireBench Foundation site (an `FBF-` id, for example `FBF-0001`, the McCully
Fire) does not give out its observation file. It gives a **benchmark kit** instead: a model-output
template that holds the layout, positions, and times the case scores, but no observed value. The
adapter reads the template where the Caldor workflow reads the observation file.

## 1. Download the kit

On the dataset's page of the site, the Model output card downloads the kit. Choose **WRF-SFIRE** as
the adapter, so the kit holds a config for this adapter, already filled in:

```
FBF-0001_mccully-fire_2026.0-dev2/
  README.md            frame, periods, what the output must hold
  template.h5          the model-output template
  requirements.json    every requirement FireBench checks
  case.json            the case definition
  adapters/
    wrf-sfire.yaml     this adapter's config
```

## 2. Run WRF-SFIRE over the frame and the window

The kit's `README.md` gives the box and the window. The weather stations outside every atmospheric
grid are skipped, so domains that hold the stations of the case score more of them.

With several domains, the adapter takes each value from the finest domain that holds the point (the
highest `dNN`), and the next domain otherwise: a station outside the fire domain `d02` but inside
`d01` is sampled on `d01`. Perimeters come from the finest domain with a fire grid.

## 3. Fill in the config

Open `adapters/wrf-sfire.yaml` and change the two lines marked `CHANGE_ME`:

```yaml
main:
  template_path: ../template.h5
  wrf_path: /path/to/the/run                          # CHANGE_ME: its folder, or one wrfout file
  output_path: ../output/FBF-0001_WRF-SFIRE.h5
  authors: "Collaborator Name, Institution;"          # CHANGE_ME
  start_dt: '2026-08-22T00:00:00Z'
  end_dt: '2026-09-15T00:00:00Z'
case:
  id: FBF-0001
  definition_sha256: a1d8ee6a…
perimeters:
  enabled: true
  paths: [...]
  kml_dir: ../output/kml_model
burn_severity_cbi:
  enabled: true
  group_path: /spatial_2d/ravg_cbi
building_damage:
  enabled: false
  group_path: null
weather:
  enabled: true
ravg_cc:
  enabled: true
  group_path: /spatial_2d/ravg_cc
frp:
  enabled: true
  group_path: /time_series/fire_total
  conversion_factor: [5, 10]
plume:
  enabled: true
  tracer: tr17_1                                      # the fire tracer compared with the radar
  max_time_gap: 1800
  points: {max_per_time: 20000, every_s: 900}
```

Relative paths are resolved from the folder of the config. `wrf_path` is the folder that holds the
run's `wrfout_dNN_*` files (the adapter finds them, every domain), or one wrfout file. The time
`XTIME` and `TIGN_G` count from is the run's `SIMULATION_START_DATE`; to give another, add
`sfire_ref_time`, with a timezone offset (`Z` is UTC). The other sections and keys of the Caldor tutorial (`overwrite`, `fire_grid_bound`,
`weather.i_fmc_10h`, …) work here too.

`main.template_path` and `main.obs_path` cannot both be given. When the template holds a case
definition, the adapter checks that its SHA-256 is `case.definition_sha256`, and stops when the config
was made for another version of the case: download the kit again.

## 4. Run the adapter

```bash
fb-sfire run adapters/wrf-sfire.yaml
```

It writes `output/FBF-0001_WRF-SFIRE.h5` and the perimeters' KML files under `output/kml_model/`.
An existing output stops the run; `fb-sfire run -o adapters/wrf-sfire.yaml` replaces it and
recomputes the KML files (as `main.overwrite: true`).

The `burn_severity_cbi` section writes RAVG's four-class composite burn index
(`ravg_composite_burn_index_severity`) from the fraction of fuel left, `FUEL_FRAC`, at the end of the
run: at most 0.2 is high (4), at most 0.6 moderate (3), at most 0.8 low (2), and above unchanged (1).
`burn_severity_cbi.cbi_thresholds` changes the three bounds. A cell of the template's grid outside the
fire grid is left at the template's `_FillValue`, which FireBench does not score.

The `ravg_cc` section writes RAVG's five classes of canopy cover loss from the crown fraction burnt,
`CFB` at the last output time, the map of the canopy the fire consumed: 0 is 1 (0 %), above 0 is 2,
from 0.25 is 3, from 0.5 is 4, and from 0.75 is 5, RAVG's own bounds. A cell the fire did not burn
(`FIRE_AREA` below 0.5) is 9, outside the burn area, and a cell outside the fire grid is 0, no data.
FireBench scores the classes only where the model and RAVG both say the cell burnt: where it burns is
the perimeters' question.

The `frp` section writes the fire radiative power the run gives over the frame of the case, at every
output time, as a band: the sensible heat released over the frame, surface and canopy fire together
(`FGRNHFX` + `FCANHFX` on the fire grid times the area of each fire cell inside the frame, in MW) divided by the largest and by the smallest
`conversion_factor`. The default `[5, 10]` takes the heat flux to be 5 to 10 times the FRP. FireBench
scores the share of satellite observations inside the band. The frame comes from the template; with
an observation file as reference, give it as `frp.frame: [west, south, east, north]`.

The `plume` section writes the run's plume where and when the radar saw it, for FireBench's
`radar_plume` family: one of WRF-SFIRE's fire tracers, `tr17_1` to `tr17_8` (`plume.tracer`, `tr17_1`
by default), and the wind, on the nodes of each radar cross section of the case at the start of each
radar volume, and, for the radars of `echo_top`, the count of the model's plume cells over the radar's
frame in bins of height. The template gives these targets. Every domain that holds the tracer gives
them, so the plume is followed past the nests: each position (a node of a section, a column of a
frame) comes from the finest domain that holds the tracer and covers it, and each domain is read cut
to the cells around the positions it gives. Over a frame given by several domains, each cell counts
for its area, so a nest does not outweigh the grid around it. `plume.domain: d01` takes every
position from one domain, NaN where it does not reach. The wind is destaggered to the mass points and
turned to east and north with `COSALPHA` and `SINALPHA`; heights come from `PH` + `PHB`. A radar time
between two output times takes both, linearly in time; a time outside the run, or between outputs
more than `max_time_gap` seconds apart, stays NaN, and FireBench leaves that volume out. The output
holds no 3D field: the run stays where it ran. `plume.points` also keeps up to `max_per_time` plume
cells every `every_s` seconds for the site's 3D figures (`points: false` leaves them out); they are
never scored. The tracer's scale does not matter: FireBench compares where it reaches a share of its
largest value.

### What the simulation cannot give

The adapter writes what the simulation can give and skips the rest, so the same config serves a short
test run and the full run. Each skipped item is printed as `WARNING:`, the count at the end, and the
run still exits 0:

- a perimeter before ignition, after the last output time of the simulation, or without a polygon;
- a weather station outside the atmospheric grid, or with no observation time inside the simulation;
- a variable whose WRF-SFIRE fields (`WZ0`, `Q2`, `FMC_GC`, …) or sensor height are missing;
- a whole section that fails on the data (for example `FUEL_FRAC` missing for `burn_severity_cbi`).

A station keeps only the observation times the simulation covers. The site then tells which periods
and KPI groups the output can be scored on.

### Reduced outputs

A run whose IO config (`iofields_filename`) leaves variables out of the wrfout files is read as it
is: a missing variable skips what needs it, and the warning names it (`FUEL_FRAC is not in the output
of d02`). To score every family, keep these variables in the output:

| Section | Variables |
|---|---|
| every section | `XTIME`, `XLAT`, `XLONG`, `FXLAT`, `FXLONG` |
| `perimeters` | `TIGN_G` |
| `burn_severity_cbi`, `severity` | `FUEL_FRAC` (the last output file is enough) |
| `building_damage` | `FIRE_AREA` |
| `ravg_cc` | `FIRE_AREA`, `CFB` (the last output file is enough) |
| `frp` | `FGRNHFX` and `FCANHFX` at every output time |
| `plume` | the tracer (`tr17_1` … `tr17_8`), `PH`, `PHB`, `U`, `V`, `W`, `COSALPHA`, `SINALPHA` at every output time around the radar's volumes |
| `weather` | `T2`, `Q2`, `PSFC`, `U10`, `V10`, `COSALPHA`, `SINALPHA`, `WZ0` (fire domain) or `ZNT`, `FMC_GC` |

`FMC_GC` may hold every fuel class (`weather.i_fmc_10h` picks the 10-hour class, 1 by default) or the
10-hour class alone; the adapter tells them apart by their dimensions. A config error, such as an unknown method, still stops
the run before anything is written.

## 5. Submit

Zip the `output/` folder, so the HDF5 file and its `kml_model/` folder are together, and submit the zip
on the dataset's submit page of the site. FireBench checks it against the observations; you then
choose the KPI groups to score.
