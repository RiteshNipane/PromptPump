"""Token counting with an honest label for how each number was obtained.

The default counter is a cheap heuristic so results are identical on every machine.
``tiktoken`` is an opt-in approximation (it is OpenAI's tokenizer, not Qwen's). The only
*exact* model-side number is Ollama's ``prompt_eval_count``; see ``telemetry.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


def heuristic_count(text: str) -> int:
    """Roughly 4 characters per token; deterministic and dependency-free."""
    return max(1, len(text) // 4) if text else 0


@dataclass(frozen=True)
class TokenCounter:
    """A named token-counting callable."""

    name: str
    count: Callable[[str], int]

    def __call__(self, text: str) -> int:
        return self.count(text)


def _tiktoken_counter() -> TokenCounter | None:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:  # not installed, or the vocabulary could not be loaded offline
        return None
    return TokenCounter(
        "tiktoken cl100k_base (approximation for Qwen)",
        lambda text: len(encoding.encode(text, disallowed_special=())) if text else 0,
    )


def get_counter(kind: str = "heuristic") -> TokenCounter:
    """Return a counter: ``heuristic`` (default), ``tiktoken``, or ``auto``."""
    kind = kind.lower()
    if kind not in {"heuristic", "tiktoken", "auto"}:
        raise ValueError("tokenizer must be one of: heuristic, tiktoken, auto")
    if kind in {"tiktoken", "auto"}:
        counter = _tiktoken_counter()
        if counter is not None:
            return counter
        if kind == "tiktoken":
            raise ValueError("tiktoken is unavailable. Install it: pip install 'promptpump[tokens]'")
    return TokenCounter("heuristic (~4 chars/token)", heuristic_count)
