# firebench-adapter-wrf-sfire

This package provides a FireBench adapter for the WRF-SFIRE model.

It enables the transformation of native WRF-SFIRE outputs (netCDF) into **FireBench-standardized datasets** compatible with FireBench benchmark cases.

## Purpose

FireBench benchmarks require model outputs to follow a strict standard (See [FireBench documentation](https://firebench.readthedocs.io/en/latest/standard_format.html))

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

For users, in the package directory:
```bash
pip install .
```

For developers, in the package directory:
```bash
pip install -e ".[dev]"
```
