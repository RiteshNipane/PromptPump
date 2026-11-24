"""Small per-user settings for PromptPump's current Ollama model."""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_MODEL = "qwen2.5-coder:7b"


def config_path() -> Path:
    """Return the user-level PromptPump settings file path."""
    return Path.home() / ".promptpump" / "config.json"


def get_model_preference(explicit: str | None = None) -> str:
    """Resolve the model option, environment preference, saved choice, then default."""
    if explicit and explicit.strip():
        return explicit.strip()

    try:
        stored = json.loads(config_path().read_text(encoding="utf-8")).get("model")
    except (OSError, ValueError, AttributeError):
        stored = None
    if isinstance(stored, str) and stored.strip():
        return stored.strip()

    from_env = os.getenv("PROMPTPUMP_MODEL", "").strip()
    return from_env or DEFAULT_MODEL


def set_model_preference(model: str) -> Path:
    """Persist a model selected by the user and return the settings file path."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model}, indent=2) + "\n", encoding="utf-8")
    return path
