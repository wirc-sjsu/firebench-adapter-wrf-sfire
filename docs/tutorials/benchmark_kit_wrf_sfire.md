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

The kit's `README.md` gives the box and the window. The weather stations outside the atmospheric grid
are skipped, so a domain that holds the stations of the case scores more of them.

## 3. Fill in the config

Open `adapters/wrf-sfire.yaml` and change the three lines marked `CHANGE_ME`:

```yaml
main:
  template_path: ../template.h5
  wrf_path: /path/to/wrfout_d01_2026-08-21_00:00:00   # CHANGE_ME
  output_path: ../output/FBF-0001_WRF-SFIRE.h5
  authors: "Collaborator Name, Institution;"          # CHANGE_ME
  sfire_ref_time: "2026-08-21T00:00:00Z"              # CHANGE_ME: the time TIGN_G counts from
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
```

Relative paths are resolved from the folder of the config. `sfire_ref_time` needs a timezone offset
(`Z` is UTC). The other sections and keys of the Caldor tutorial (`overwrite`, `fire_grid_bound`,
`weather.i_fmc_10h`, …) work here too.

`main.template_path` and `main.obs_path` cannot both be given. When the template holds a case
definition, the adapter checks that its SHA-256 is `case.definition_sha256`, and stops when the config
was made for another version of the case: download the kit again.

## 4. Run the adapter

```bash
fb-sfire run adapters/wrf-sfire.yaml
```

It writes `output/FBF-0001_WRF-SFIRE.h5` and the perimeters' KML files under `output/kml_model/`.

The `burn_severity_cbi` section writes RAVG's four-class composite burn index
(`ravg_composite_burn_index_severity`) from the fraction of fuel left, `FUEL_FRAC`, at the end of the
run: at most 0.2 is high (4), at most 0.6 moderate (3), at most 0.8 low (2), and above unchanged (1).
`burn_severity_cbi.cbi_thresholds` changes the three bounds. A cell of the template's grid outside the
fire grid is left at the template's `_FillValue`, which FireBench does not score.

## 5. Submit

Zip the `output/` folder, so the HDF5 file and its `kml_model/` folder are together, and submit the zip
on the dataset's submit page of the site. FireBench checks it against the observations; you then
choose the KPI groups to score.
