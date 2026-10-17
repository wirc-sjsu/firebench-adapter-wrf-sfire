# firebench-adapter-wrf-sfire

This package provides a [FireBench](https://firebench.readthedocs.io/en/latest/) adapter for the WRF-SFIRE model.

It enables the transformation of native WRF-SFIRE outputs (netCDF) into **FireBench-standardized datasets** compatible with FireBench benchmark cases.

## Purpose

FireBench benchmarks require model outputs to follow a strict standard. See the [FireBench standard format documentation](https://firebench.readthedocs.io/en/latest/standard_format.html).

This adapter:
- reads native WRF-SFIRE outputs,
- interpolates them to FireBench benchmark grids,
- applies required post-processing.

## Scope

This adapter is responsible only for **output standardization**.

It does **not**:
- run the model,
- modify model physics,
- tune parameters for specific benchmark cases.

This separation is intentional and ensures fair benchmarking.

## Installation

For users:
```bash
pip install firebench-adapter-wrf-sfire
```

For local development:
```bash
pip install -e ".[dev]"
```

## Usage

The adapter exposes functions for converting WRF-SFIRE output into FireBench-standardized groups:

- `get_mtbs_severity_surf`
- `get_building_damage`
- `get_ravg_cc`
- `get_perimeters_from_t_ign_g`
- `get_weather_stations`

The package also provides a YAML-driven CLI:
```bash
fb-sfire run config.yml
```

The config contains a `main` section for shared paths and times, plus dedicated sections such as
`severity`, `building_damage`, `ravg_cc`, `perimeters`, and `weather`.

To enter a benchmark case of the FireBench Foundation site (`FBF-` cases), download the case's
benchmark kit with the WRF-SFIRE adapter chosen, then run `fb-sfire run adapters/wrf-sfire.yaml`; see
`docs/tutorials/benchmark_kit_wrf_sfire.md`.

See the tutorial in the documentation for a complete WRF-SFIRE to FireBench workflow:

https://firebench-adapter-wrf-sfire.readthedocs.io/en/latest/tutorials/caldor_wrf_sfire.html

## Development

Run tests and checks with:
```bash
make test
make code-formatting
make check-dist
```

This project is licensed under the Apache License 2.0.
