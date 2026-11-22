"""Environment-backed feature flags for the web application."""

import os
from collections.abc import Mapping

DEFAULT_FLAGS = {"new_nav": False, "compact_cards": True}
TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def parse_bool(value: str | None, default: bool = False) -> bool:
    """Parse common environment-variable representations of booleans."""
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return default


def load_feature_flags(environ: Mapping[str, str] | None = None) -> dict[str, bool]:
    """Load documented flags from an environment mapping or the process environment."""
    raise NotImplementedError
