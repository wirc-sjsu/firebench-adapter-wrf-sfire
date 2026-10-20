from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import h5py
import numpy as np
import yaml

import firebench.standardize as fs
from firebench_adapter_wrf_sfire import (
    get_building_damage,
    get_fire_total,
    get_mtbs_severity_surf,
    get_perimeters_from_t_ign_g,
    get_plume,
    get_ravg_cbi_severity,
    get_ravg_cc,
    get_weather_stations,
)
from firebench_adapter_wrf_sfire._skip import SkippedWarning, skip
from firebench_adapter_wrf_sfire._wrfout import WrfRun, open_run
from firebench_adapter_wrf_sfire.frp import DEFAULT_CONVERSION_FACTOR, DEFAULT_GROUP_PATH
from firebench_adapter_wrf_sfire.perimeters import PERIMETER_EXTRACTION_METHODS
from firebench_adapter_wrf_sfire.plume import DEFAULT_MAX_TIME_GAP, DEFAULT_TRACER, TRACER_NAME

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
@click.option(
    "-o",
    "--overwrite",
    "overwrite_flag",
    is_flag=True,
    help="Replace an existing output file and recompute its KML perimeters (as main.overwrite: true).",
)
def run(config_yml: Path, overwrite_flag: bool) -> None:
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
    overwrite = overwrite_flag or bool(main_config.get("overwrite", False))
    compression_lvl = int(main_config.get("compression_lvl", DEFAULT_COMPRESSION_LVL))
    fire_grid_bound = int(main_config.get("fire_grid_bound", DEFAULT_FIRE_GRID_BOUND))

    if not wrf_path.exists():
        raise click.UsageError(
            f"WRF-SFIRE output does not exist: {wrf_path} (give a wrfout file or the folder of the run)"
        )

    _check_case(config, obs_path)

    sfire_ref_time = _optional_str(main_config, "sfire_ref_time")
    start_dt = _optional_datetime(main_config, "start_dt")
    end_dt = _optional_datetime(main_config, "end_dt")

    if _section_enabled(config, "perimeters"):
        extraction_method = _optional_mapping(config, "perimeters").get("extraction_method", "tign_g")
        if extraction_method not in PERIMETER_EXTRACTION_METHODS:
            raise click.UsageError(
                f"perimeters.extraction_method must be one of {', '.join(PERIMETER_EXTRACTION_METHODS)}."
            )
    _check_band_and_plume(config)
    if _section_enabled(config, "weather"):
        if start_dt is None or end_dt is None:
            raise click.UsageError(
                "main.start_dt and main.end_dt are required when weather.enabled is true."
            )

    if output_path.exists() and not overwrite:
        raise click.UsageError(f"Output file already exists: {output_path}; run with -o to replace it.")
    try:
        run_files = open_run(wrf_path)
    except (OSError, ValueError) as exc:
        raise click.UsageError(f"Cannot read the WRF-SFIRE output {wrf_path}: {exc}") from exc
    with run_files:
        sfire_ref_time = _run_ref_time(run_files, sfire_ref_time, config)
        try:
            new_h5 = fs.new_std_file(str(output_path), authors, overwrite=overwrite)
        except FileExistsError as exc:
            raise click.UsageError(
                f"Output file already exists: {output_path}; run with -o to replace it."
            ) from exc
        caught = _run_sections(
            config,
            config_dir,
            new_h5,
            obs_path,
            run_files,
            sfire_ref_time,
            start_dt,
            end_dt,
            fire_grid_bound,
            compression_lvl,
            overwrite,
        )

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


def _run_ref_time(run_files: WrfRun, sfire_ref_time: str | None, config: dict[str, Any]) -> str | None:
    """Echo the domains of the run, and give sfire_ref_time, from the run when the config has none."""
    for domain in run_files.domains:
        click.echo(f"WRF-SFIRE domain {domain.describe()}")
    if sfire_ref_time is not None:
        return sfire_ref_time
    if not any(_section_enabled(config, name) for name in ("perimeters", "weather", "frp", "plume")):
        return None
    try:
        sfire_ref_time = run_files.start_time()
    except ValueError as exc:
        raise click.UsageError(
            f"main.sfire_ref_time is required: the WRF-SFIRE output does not give it ({exc})."
        ) from exc
    click.echo(f"sfire_ref_time {sfire_ref_time}, the start of the run (SIMULATION_START_DATE)")
    return sfire_ref_time


def _run_sections(  # pylint: disable=too-many-arguments
    config: dict[str, Any],
    config_dir: Path,
    new_h5: h5py.File,
    obs_path: Path,
    wrf_dataset: WrfRun,
    sfire_ref_time: str | None,
    start_dt: datetime | None,
    end_dt: datetime | None,
    fire_grid_bound: int,
    compression_lvl: int,
    overwrite: bool,
) -> list[warnings.WarningMessage]:
    """Run every enabled section into ``new_h5``, close it, and return the warnings they raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SkippedWarning)
        try:
            with h5py.File(obs_path, "r") as obs_h5:
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
                    (
                        "frp",
                        lambda: _run_frp(config, *common, sfire_ref_time, fire_grid_bound, compression_lvl),
                    ),
                    ("plume", lambda: _run_plume(config, *common, sfire_ref_time, compression_lvl)),
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
    return caught


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
            "frp",
            "plume",
        )
        if _section_enabled(config, section_name)
    ]


def _run_severity(
    config: dict[str, Any],
    new_h5: h5py.File,
    obs_h5: h5py.File,
    wrf_dataset: WrfRun,
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
    wrf_dataset: WrfRun,
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
    wrf_dataset: WrfRun,
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


def _numbers(value, count: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == count
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    )


def _check_band_and_plume(config: dict[str, Any]) -> None:
    """Refuse an enabled frp or plume section the run could not use."""
    if _section_enabled(config, "frp"):
        _check_frp_section(_optional_mapping(config, "frp"))
    if _section_enabled(config, "plume"):
        _check_plume_section(_optional_mapping(config, "plume"))


def _check_frp_section(section: dict[str, Any]) -> None:
    """Refuse an frp section the run could not use, before anything is written."""
    factor = section.get("conversion_factor", list(DEFAULT_CONVERSION_FACTOR))
    if not _numbers(factor, 2) or not 0 < factor[0] <= factor[1]:
        raise click.UsageError("frp.conversion_factor must be [smallest, largest], both above 0.")
    frame = section.get("frame")
    if frame is not None and not _numbers(frame, 4):
        raise click.UsageError("frp.frame must be [west, south, east, north].")


def _run_frp(  # pylint: disable=too-many-arguments
    config: dict[str, Any],
    new_h5: h5py.File,
    obs_h5: h5py.File,
    wrf_dataset: WrfRun,
    sfire_ref_time: str | None,
    fire_grid_bound: int,
    compression_lvl: int,
) -> None:
    section = _optional_mapping(config, "frp")
    if not section.get("enabled", False):
        return

    click.echo("Process Fire Radiative Power")
    get_fire_total(
        new_h5,
        obs_h5,
        section.get("group_path", DEFAULT_GROUP_PATH),
        wrf_dataset,
        sfire_ref_time,
        conversion_factor=section.get("conversion_factor", list(DEFAULT_CONVERSION_FACTOR)),
        frame=section.get("frame"),
        fire_grid_bound=int(section.get("fire_grid_bound", fire_grid_bound)),
        compression_lvl=int(section.get("compression_lvl", compression_lvl)),
    )


def _plume_points(section: dict[str, Any]) -> dict[str, Any] | None:
    """The points of the plume for figures: the defaults, the values given, or None when false."""
    points = section.get("points", {})
    if points is False:
        return None
    return {} if points is None or points is True else points


def _check_plume_section(section: dict[str, Any]) -> None:
    """Refuse a plume section the run could not use, before anything is written."""
    tracer = section.get("tracer", DEFAULT_TRACER)
    if not isinstance(tracer, str) or not TRACER_NAME.fullmatch(tracer):
        raise click.UsageError("plume.tracer must be one of tr17_1 to tr17_8.")
    gap = section.get("max_time_gap", DEFAULT_MAX_TIME_GAP)
    if not _numbers([gap], 1) or gap <= 0:
        raise click.UsageError("plume.max_time_gap must be a number of seconds above 0.")
    domain = section.get("domain")
    if domain is not None and (not isinstance(domain, str) or not domain.startswith("d")):
        raise click.UsageError("plume.domain must name a domain, such as d02, or be left out.")
    points = _plume_points(section)
    if points is not None:
        if not isinstance(points, dict) or set(points) - {"max_per_time", "every_s"}:
            raise click.UsageError("plume.points may give max_per_time and every_s, or be false.")
        for key in ("max_per_time", "every_s"):
            if key in points and (not _numbers([points[key]], 1) or points[key] <= 0):
                raise click.UsageError(f"plume.points.{key} must be a number above 0.")


def _run_plume(  # pylint: disable=too-many-arguments
    config: dict[str, Any],
    new_h5: h5py.File,
    obs_h5: h5py.File,
    wrf_dataset: WrfRun,
    sfire_ref_time: str | None,
    compression_lvl: int,
) -> None:
    section = _optional_mapping(config, "plume")
    if not section.get("enabled", False):
        return

    tracer = section.get("tracer", DEFAULT_TRACER)
    click.echo(f"Process Radar Plume ({tracer})")
    written = get_plume(
        new_h5,
        obs_h5.filename,
        wrf_dataset,
        sfire_ref_time,
        tracer=tracer,
        domain=section.get("domain"),
        max_time_gap=float(section.get("max_time_gap", DEFAULT_MAX_TIME_GAP)),
        points=_plume_points(section),
        compression_lvl=int(section.get("compression_lvl", compression_lvl)),
    )
    for target, count in written.items():
        click.echo(f"  {target}: {count} time(s)")


def _run_burn_severity_cbi(
    config: dict[str, Any],
    new_h5: h5py.File,
    obs_h5: h5py.File,
    wrf_dataset: WrfRun,
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
    wrf_dataset: WrfRun,
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
    wrf_dataset: WrfRun,
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
