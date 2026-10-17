from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import h5py
import numpy as np
import yaml
from netCDF4 import Dataset

import firebench.standardize as fs
from firebench_adapter_wrf_sfire import (
    get_building_damage,
    get_mtbs_severity_surf,
    get_perimeters_from_t_ign_g,
    get_ravg_cbi_severity,
    get_ravg_cc,
    get_weather_stations,
)
from firebench_adapter_wrf_sfire._skip import SkippedWarning, skip
from firebench_adapter_wrf_sfire.perimeters import PERIMETER_EXTRACTION_METHODS

DEFAULT_COMPRESSION_LVL = 3
DEFAULT_FIRE_GRID_BOUND = 25
DEFAULT_PERIMETER_KML_DIR = "kml_model"
# Root attribute of a FireBench model-output template (firebench.benchmarks.fire_case_definition).
TEMPLATE_ATTRIBUTE = "firebench_template"
# A section that fails on the data is skipped, so one config serves simulations that give different
# parts of a case. An unknown method (NotImplementedError) is a config error and stops the run.
SECTION_ERRORS = (KeyError, IndexError, ValueError)


@click.group()
def main() -> None:
    """Convert WRF-SFIRE outputs to FireBench-standard HDF5 files."""


@main.command()
@click.argument("config_yml", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def run(config_yml: Path) -> None:
    """Run WRF-SFIRE adapter processing from CONFIG_YML."""
    config = _load_config(config_yml)
    config_dir = config_yml.resolve().parent
    main_config = _required_mapping(config, "main")

    enabled_sections = _enabled_processing_sections(config)
    if not enabled_sections:
        raise click.UsageError("At least one processing section must have enabled: true.")

    obs_path = _reference_path(main_config, config_dir)
    wrf_path = _required_path(main_config, "wrf_path", config_dir)
    output_path = _required_path(main_config, "output_path", config_dir)
    authors = _required_str(main_config, "authors")
    overwrite = bool(main_config.get("overwrite", False))
    compression_lvl = int(main_config.get("compression_lvl", DEFAULT_COMPRESSION_LVL))
    fire_grid_bound = int(main_config.get("fire_grid_bound", DEFAULT_FIRE_GRID_BOUND))

    if not wrf_path.is_file():
        raise click.UsageError(f"WRF-SFIRE file does not exist: {wrf_path}")

    _check_case(config, obs_path)

    sfire_ref_time = _optional_str(main_config, "sfire_ref_time")
    start_dt = _optional_datetime(main_config, "start_dt")
    end_dt = _optional_datetime(main_config, "end_dt")

    if _section_enabled(config, "perimeters"):
        if sfire_ref_time is None:
            raise click.UsageError("main.sfire_ref_time is required when perimeters.enabled is true.")
        extraction_method = _optional_mapping(config, "perimeters").get("extraction_method", "tign_g")
        if extraction_method not in PERIMETER_EXTRACTION_METHODS:
            raise click.UsageError(
                f"perimeters.extraction_method must be one of {', '.join(PERIMETER_EXTRACTION_METHODS)}."
            )
    if _section_enabled(config, "weather"):
        if sfire_ref_time is None:
            raise click.UsageError("main.sfire_ref_time is required when weather.enabled is true.")
        if start_dt is None or end_dt is None:
            raise click.UsageError(
                "main.start_dt and main.end_dt are required when weather.enabled is true."
            )

    try:
        new_h5 = fs.new_std_file(str(output_path), authors, overwrite=overwrite)
    except FileExistsError as exc:
        raise click.UsageError(f"Output file already exists: {output_path}") from exc

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SkippedWarning)
        try:
            with h5py.File(obs_path, "r") as obs_h5, Dataset(wrf_path, "r") as wrf_dataset:
                common = (new_h5, obs_h5, wrf_dataset)
                sections = (
                    ("severity", lambda: _run_severity(config, *common, fire_grid_bound, compression_lvl)),
                    (
                        "building_damage",
                        lambda: _run_building_damage(config, *common, fire_grid_bound, compression_lvl),
                    ),
                    ("ravg_cc", lambda: _run_ravg_cc(config, *common, fire_grid_bound, compression_lvl)),
                    (
                        "burn_severity_cbi",
                        lambda: _run_burn_severity_cbi(config, *common, fire_grid_bound, compression_lvl),
                    ),
                    (
                        "perimeters",
                        lambda: _run_perimeters(
                            config, config_dir, *common, sfire_ref_time, fire_grid_bound, overwrite
                        ),
                    ),
                    (
                        "weather",
                        lambda: _run_weather(
                            config,
                            *common,
                            sfire_ref_time,
                            start_dt,
                            end_dt,
                            fire_grid_bound,
                            compression_lvl,
                        ),
                    ),
                )
                shown = 0
                for section_name, run_section in sections:
                    try:
                        run_section()
                    except SECTION_ERRORS as exc:
                        skip(f"Section {section_name}", exc)
                    shown = _echo_skipped(caught, shown)
        finally:
            new_h5.close()

    skipped = [item for item in caught if issubclass(item.category, SkippedWarning)]
    for item in caught:
        if not issubclass(item.category, SkippedWarning):
            warnings.showwarning(item.message, item.category, item.filename, item.lineno)
    click.echo(f"Wrote {output_path}")
    if skipped:
        click.echo(
            f"{len(skipped)} item(s) skipped, listed above as WARNING; the output holds the rest.",
            err=True,
        )


def _echo_skipped(caught: list[warnings.WarningMessage], shown: int) -> int:
    """Echo the skipped items recorded since ``shown``, and return the new count."""
    for item in caught[shown:]:
        if issubclass(item.category, SkippedWarning):
            click.echo(f"WARNING: {item.message}", err=True)
    return len(caught)


def _load_config(config_yml: Path) -> dict[str, Any]:
    try:
        with config_yml.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise click.UsageError(f"Invalid YAML config: {exc}") from exc

    if not isinstance(config, dict):
        raise click.UsageError("Config file must contain a YAML mapping.")
    return config


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise click.UsageError(f"{key} section is required.")
    return value


def _optional_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise click.UsageError(f"{key} section must be a mapping.")
    return value


def _required_str(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise click.UsageError(f"{key} is required.")
    return value


def _optional_str(config: dict[str, Any], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise click.UsageError(f"{key} must be a non-empty string.")
    return value


def _required_path(config: dict[str, Any], key: str, base_dir: Path) -> Path:
    return _resolve_path(Path(_required_str(config, key)), base_dir)


def _resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return base_dir / path


def _reference_path(main_config: dict[str, Any], config_dir: Path) -> Path:
    """The file the output is laid out on: an observation file or a benchmark kit's template."""
    has_obs = main_config.get("obs_path") is not None
    has_template = main_config.get("template_path") is not None
    if has_obs and has_template:
        raise click.UsageError("Give main.obs_path or main.template_path, not both.")
    if not has_obs and not has_template:
        raise click.UsageError(
            "main.template_path (the template.h5 of a benchmark kit) or main.obs_path is required."
        )
    if has_obs:
        obs_path = _required_path(main_config, "obs_path", config_dir)
        if not obs_path.is_file():
            raise click.UsageError(f"Observation file does not exist: {obs_path}")
        return obs_path

    template_path = _required_path(main_config, "template_path", config_dir)
    if not template_path.is_file():
        raise click.UsageError(f"Template file does not exist: {template_path}")
    with h5py.File(template_path, "r") as template_h5:
        if TEMPLATE_ATTRIBUTE not in template_h5.attrs:
            raise click.UsageError(
                f"{template_path} is not a model-output template (no root attribute "
                f"{TEMPLATE_ATTRIBUTE}). Give an observation file as main.obs_path."
            )
    return template_path


def _check_case(config: dict[str, Any], reference_path: Path) -> None:
    """Stop when the reference holds a case definition other than the one the config names."""
    section = _optional_mapping(config, "case")
    expected = section.get("definition_sha256")
    if expected is None:
        return
    try:
        # Imported here so that a FireBench without generic cases still runs an obs_path config.
        # pylint: disable-next=import-outside-toplevel
        from firebench.benchmarks.fire_case_definition import definition_sha256, read_case_definition
    except ImportError as exc:
        raise click.UsageError(
            "case.definition_sha256 needs a FireBench with generic benchmark cases (0.11 or later)."
        ) from exc

    try:
        definition = read_case_definition(reference_path)
    except ValueError as exc:
        raise click.UsageError(f"{reference_path}: {exc}") from exc
    if definition is None:
        return
    if definition_sha256(definition) != expected:
        raise click.UsageError(
            f"This config was made for another version of the case {section.get('id', '')}: "
            f"{reference_path} holds the definition {definition_sha256(definition)}, "
            f"the config names {expected}."
        )


def _optional_datetime(config: dict[str, Any], key: str) -> datetime | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise click.UsageError(f"main.{key} must be an ISO datetime string.") from exc
    else:
        raise click.UsageError(f"main.{key} must be an ISO datetime string.")

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise click.UsageError(f"main.{key} must include a timezone offset.")
    return parsed


def _section_enabled(config: dict[str, Any], section_name: str) -> bool:
    section = _optional_mapping(config, section_name)
    return bool(section.get("enabled", False))


def _enabled_processing_sections(config: dict[str, Any]) -> list[str]:
    return [
        section_name
        for section_name in (
            "severity",
            "building_damage",
            "ravg_cc",
            "burn_severity_cbi",
            "perimeters",
            "weather",
        )
        if _section_enabled(config, section_name)
    ]


def _run_severity(
    config: dict[str, Any],
    new_h5: h5py.File,
    obs_h5: h5py.File,
    wrf_dataset: Dataset,
    fire_grid_bound: int,
    compression_lvl: int,
) -> None:
    section = _optional_mapping(config, "severity")
    if not section.get("enabled", False):
        return

    click.echo("Process Burn Severity")
    severity_thresholds = section.get("severity_thresholds")
    if severity_thresholds is not None:
        severity_thresholds = np.array(severity_thresholds, dtype=np.float64)

    kwargs = {
        "method": section.get("method", "fuel_frac"),
        "fire_grid_bound": int(section.get("fire_grid_bound", fire_grid_bound)),
        "compression_lvl": int(section.get("compression_lvl", compression_lvl)),
    }
    if severity_thresholds is not None:
        kwargs["severity_thresholds"] = severity_thresholds

    get_mtbs_severity_surf(
        new_h5,
        obs_h5,
        _required_str(section, "group_path"),
        wrf_dataset,
        **kwargs,
    )


def _run_building_damage(
    config: dict[str, Any],
    new_h5: h5py.File,
    obs_h5: h5py.File,
    wrf_dataset: Dataset,
    fire_grid_bound: int,
    compression_lvl: int,
) -> None:
    section = _optional_mapping(config, "building_damage")
    if not section.get("enabled", False):
        return

    click.echo("Process Building Damage")
    get_building_damage(
        new_h5,
        obs_h5,
        _required_str(section, "group_path"),
        wrf_dataset,
        method=section.get("method", "fire_area"),
        fire_grid_bound=int(section.get("fire_grid_bound", fire_grid_bound)),
        compression_lvl=int(section.get("compression_lvl", compression_lvl)),
    )


def _run_ravg_cc(
    config: dict[str, Any],
    new_h5: h5py.File,
    obs_h5: h5py.File,
    wrf_dataset: Dataset,
    fire_grid_bound: int,
    compression_lvl: int,
) -> None:
    section = _optional_mapping(config, "ravg_cc")
    if not section.get("enabled", False):
        return

    click.echo("Process RAVG Canopy Cover Loss")
    get_ravg_cc(
        new_h5,
        obs_h5,
        _required_str(section, "group_path"),
        wrf_dataset,
        method=section.get("method", "cfb"),
        fire_grid_bound=int(section.get("fire_grid_bound", fire_grid_bound)),
        compression_lvl=int(section.get("compression_lvl", compression_lvl)),
    )


def _run_burn_severity_cbi(
    config: dict[str, Any],
    new_h5: h5py.File,
    obs_h5: h5py.File,
    wrf_dataset: Dataset,
    fire_grid_bound: int,
    compression_lvl: int,
) -> None:
    section = _optional_mapping(config, "burn_severity_cbi")
    if not section.get("enabled", False):
        return

    click.echo("Process RAVG Composite Burn Index Severity")
    kwargs = {
        "method": section.get("method", "fuel_frac"),
        "fire_grid_bound": int(section.get("fire_grid_bound", fire_grid_bound)),
        "compression_lvl": int(section.get("compression_lvl", compression_lvl)),
    }
    cbi_thresholds = section.get("cbi_thresholds")
    if cbi_thresholds is not None:
        kwargs["cbi_thresholds"] = np.array(cbi_thresholds, dtype=np.float64)

    get_ravg_cbi_severity(
        new_h5,
        obs_h5,
        _required_str(section, "group_path"),
        wrf_dataset,
        **kwargs,
    )


def _run_perimeters(
    config: dict[str, Any],
    config_dir: Path,
    new_h5: h5py.File,
    obs_h5: h5py.File,
    wrf_dataset: Dataset,
    sfire_ref_time: str | None,
    fire_grid_bound: int,
    overwrite: bool,
) -> None:
    section = _optional_mapping(config, "perimeters")
    if not section.get("enabled", False):
        return

    paths = section.get("paths")
    if not isinstance(paths, list) or not paths or not all(isinstance(path, str) for path in paths):
        raise click.UsageError("perimeters.paths must be a non-empty list of group paths.")

    click.echo("Process Perimeters")
    kml_dir = _resolve_path(Path(section.get("kml_dir", DEFAULT_PERIMETER_KML_DIR)), config_dir)
    get_perimeters_from_t_ign_g(
        new_h5,
        obs_h5,
        wrf_dataset,
        paths,
        sfire_ref_time,
        fire_grid_bound=int(section.get("fire_grid_bound", fire_grid_bound)),
        kml_dir=str(kml_dir),
        overwrite=bool(section.get("overwrite", overwrite)),
        extraction_method=section.get("extraction_method", "tign_g"),
        smooth_sigma=float(section.get("smooth_sigma", 0)),
    )


def _run_weather(
    config: dict[str, Any],
    new_h5: h5py.File,
    obs_h5: h5py.File,
    wrf_dataset: Dataset,
    sfire_ref_time: str | None,
    start_dt: datetime | None,
    end_dt: datetime | None,
    fire_grid_bound: int,
    compression_lvl: int,
) -> None:
    section = _optional_mapping(config, "weather")
    if not section.get("enabled", False):
        return

    click.echo("Process WX")
    get_weather_stations(
        new_h5,
        obs_h5,
        wrf_dataset,
        sfire_ref_time,
        start_dt,
        end_dt,
        fire_grid_bound=int(section.get("fire_grid_bound", fire_grid_bound)),
        compression_lvl=int(section.get("compression_lvl", compression_lvl)),
        i_fmc_10h=section.get("i_fmc_10h", 1),
    )
