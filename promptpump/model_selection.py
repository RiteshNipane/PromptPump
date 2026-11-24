"""Choose a usable local Ollama chat model from the models already installed."""

from __future__ import annotations

import re
from collections.abc import Iterable

_NON_CHAT_MARKERS = (
    "embed",
    "embedding",
    "rerank",
    "whisper",
    "tts",
    "text-to-speech",
)

_CODE_MODEL_MARKERS = (
    "coder",
    "codestral",
    "codegemma",
    "starcoder",
    "granite-code",
    "devstral",
)

_GENERAL_MODEL_ORDER = (
    "deepseek-r1",
    "qwen3",
    "qwen2.5",
    "llama3.3",
    "llama3.2",
    "llama3.1",
    "gemma3",
    "phi4",
    "mistral",
    "llama",
    "gemma",
    "phi",
)


def _canonical(name: str) -> str:
    name = name.strip().lower()
    if name.endswith(":latest"):
        return name[:-7]
    return name


def _family(name: str) -> str:
    return _canonical(name).split(":", maxsplit=1)[0]


def _parameter_billions(name: str) -> float | None:
    """Read a conventional size tag such as ``14b`` without assuming every tag has one."""
    match = re.search(r"(?:^|[:.-])(\d+(?:\.\d+)?)b(?:$|[-._])", name.lower())
    return float(match.group(1)) if match else None


def _is_chat_candidate(name: str) -> bool:
    lowered = name.lower()
    return bool(name.strip()) and not any(marker in lowered for marker in _NON_CHAT_MARKERS)


def available_chat_models(installed: Iterable[str]) -> list[str]:
    """Filter obvious embedding/audio-only tags from Ollama's installed model names."""
    return [name.strip() for name in installed if _is_chat_candidate(name)]


def select_installed_model(
    installed: Iterable[str],
    preferred: str,
    *,
    prompt_type: str = "code",
) -> str | None:
    """Return a matching or sensible installed chat model, or ``None`` if none can chat.

    Exact and same-family matches preserve user preferences. If those are absent, coding
    workflows prefer code-tuned models; creative workflows prefer general instruction
    models. All remaining installed chat models are valid last-resort candidates.
    """
    candidates = available_chat_models(installed)
    if not candidates:
        return None

    wanted = _canonical(preferred)
    for name in candidates:
        if _canonical(name) == wanted:
            return name

    same_family = [name for name in candidates if _family(name) == _family(preferred)]
    if same_family:
        wanted_size = _parameter_billions(preferred)
        if wanted_size is not None:
            same_family.sort(
                key=lambda name: abs((_parameter_billions(name) or wanted_size) - wanted_size)
            )
        return same_family[0]

    def category(name: str) -> int:
        lowered = name.lower()
        is_code = any(marker in lowered for marker in _CODE_MODEL_MARKERS)
        if prompt_type == "code" and is_code:
            return 0
        if prompt_type != "code" and not is_code:
            return 0
        return 1

    def family_priority(name: str) -> int:
        lowered = name.lower()
        for index, marker in enumerate(_GENERAL_MODEL_ORDER):
            if marker in lowered:
                return index
        return len(_GENERAL_MODEL_ORDER)

    def size_priority(name: str) -> tuple[float, float]:
        size = _parameter_billions(name)
        if size is None:
            return (1.0, 0.0)
        # Favor useful small/medium local models; exceptionally large tags can still be
        # selected when they are all that is installed, but they are not the first fallback.
        return (0.0 if size <= 32 else 1.0, abs(size - 8.0))

    return min(
        candidates, key=lambda name: (category(name), family_priority(name), *size_priority(name))
    )
