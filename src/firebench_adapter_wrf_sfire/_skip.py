import warnings


class SkippedWarning(UserWarning):
    """Part of the output the simulation cannot give; the adapter writes the rest."""


def skip(item: str, reason) -> None:
    """Warn that ``item`` is left out of the output, and why."""
    warnings.warn(f"{item} skipped: {reason}", SkippedWarning, stacklevel=3)
