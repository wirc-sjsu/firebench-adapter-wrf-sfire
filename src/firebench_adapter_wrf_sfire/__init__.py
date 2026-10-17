from importlib.metadata import version, PackageNotFoundError

from ._skip import SkippedWarning
from ._wrfout import WrfRun, open_run

from .building_damage import get_building_damage
from .perimeters import get_perimeters_from_t_ign_g
from .ravg import get_ravg_cc
from .severity_cbi import get_ravg_cbi_severity
from .severity_mtbs import get_mtbs_severity_surf
from .wx import get_weather_stations

try:
    __version__ = version("firebench-adapter-wrf-sfire")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "__version__",
    "SkippedWarning",
    "WrfRun",
    "open_run",
    "get_building_damage",
    "get_perimeters_from_t_ign_g",
    "get_ravg_cc",
    "get_ravg_cbi_severity",
    "get_mtbs_severity_surf",
    "get_weather_stations",
]
