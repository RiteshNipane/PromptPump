"""Core compiler: builds the payload, talks to Ollama, and post-processes the result."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import requests

from .context import WorkspaceContext
from .templates import (
    REQUIRED_TAGS,
    build_system_prompt,
    build_user_message,
    get_stack,
)

DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_HOST = "http://localhost:11434"

Message = dict[str, str]


class PromptPumpError(Exception):
    """Base class for all PromptPump errors."""


class OllamaError(PromptPumpError):
    """Raised when the local Ollama server cannot satisfy a request."""


class ChatClient(Protocol):
    """Anything that can turn chat messages into a text reply (easy to fake in tests)."""

    def chat(self, messages: list[Message]) -> str: ...


class OllamaClient:
    """Minimal client for Ollama's ``/api/chat`` endpoint (non-streaming)."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        *,
        timeout: float = 300.0,
        temperature: float = 0.2,
        seed: int = 42,
        num_ctx: int = 8192,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        # Low temperature + fixed seed keeps runs as reproducible as a local LLM allows.
        self.options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx}

    def chat(self, messages: list[Message]) -> str:
        """Send messages to the model and return the assistant text."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self.options,
        }
        try:
            resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.host}. Is it running? Start it with `ollama serve`."
            ) from exc
        except requests.Timeout as exc:
            raise OllamaError(f"Ollama timed out after {self.timeout:.0f}s.") from exc

        if resp.status_code == 404:
            raise OllamaError(f"Model '{self.model}' not found. Run `ollama pull {self.model}`.")
        if not resp.ok:
            raise OllamaError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            return str(resp.json()["message"]["content"])
        except (ValueError, KeyError, TypeError) as exc:
            raise OllamaError("Unexpected response format from Ollama.") from exc

    def list_models(self) -> list[str]:
        """Return the names of locally installed models."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.host}.") from exc


# ---------------------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"^```[\w-]*\s*\n(.*?)\n?```\s*$", re.DOTALL)
_CONTEXT_RE = re.compile(r"<codebase_context>.*?</codebase_context>", re.DOTALL)


def clean_response(raw: str) -> str:
    """Strip reasoning blocks, conversational filler, and wrapping code fences."""
    text = _THINK_RE.sub("", raw).strip()

    first = text.find(f"<{REQUIRED_TAGS[0]}>")
    if first > 0:  # drop any preamble such as "Sure! Here's your prompt:"
        text = text[first:]
    closing = f"</{REQUIRED_TAGS[-1]}>"
    last = text.rfind(closing)
    if last != -1:  # drop trailing remarks such as "Let me know if..."
        text = text[: last + len(closing)]

    fenced = _FENCE_RE.match(text.strip())
    if fenced:
        text = fenced.group(1)
    return text.strip()


def find_missing_tags(text: str) -> list[str]:
    """Return required tags whose opening or closing marker is absent."""
    return [t for t in REQUIRED_TAGS if f"<{t}>" not in text or f"</{t}>" not in text]


def ground_codebase_context(prompt: str, summary: str) -> str:
    """Replace the model-written ``<codebase_context>`` with deterministic facts."""
    block = f"<codebase_context>\n{summary}\n</codebase_context>"
    return _CONTEXT_RE.sub(lambda _m: block, prompt, count=1)


# ---------------------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------------------


@dataclass
class CompileResult:
    """Outcome of a compilation run."""

    prompt: str
    raw: str
    model: str
    attempts: int
    grounded: bool
    missing_tags: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.missing_tags


class PromptCompiler:
    """Combines shorthand + workspace + template rules and runs the local model."""

    def __init__(
        self, client: ChatClient, *, ground_context: bool = True, max_repairs: int = 1
    ) -> None:
        self.client = client
        self.ground_context = ground_context
        self.max_repairs = max_repairs

    def build_messages(
        self, shorthand: str, context: WorkspaceContext, stacks: Sequence[str] = ()
    ) -> list[Message]:
        """Assemble the system + user messages (also used by ``--dry-run``)."""
        profiles = list(dict.fromkeys(get_stack(s) for s in stacks))
        return [
            {"role": "system", "content": build_system_prompt(profiles)},
            {"role": "user", "content": build_user_message(shorthand, context.render())},
        ]

    def compile(
        self, shorthand: str, context: WorkspaceContext, stacks: Sequence[str] = ()
    ) -> CompileResult:
        """Compile ``shorthand`` into a validated master prompt."""
        if not shorthand.strip():
            raise PromptPumpError("Shorthand prompt must not be empty.")

        messages = self.build_messages(shorthand, context, stacks)
        raw = self.client.chat(messages)
        prompt = clean_response(raw)
        missing = find_missing_tags(prompt)
        attempts = 1

        # One-shot repair loop: show the model what it missed and ask for a full redo.
        while missing and attempts <= self.max_repairs:
            messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"Your output is missing or has unclosed tags: {', '.join(missing)}. "
                        "Re-output the COMPLETE master prompt with all seven tags in order. "
                        "Output only the prompt."
                    ),
                },
            ]
            raw = self.client.chat(messages)
            prompt = clean_response(raw)
            missing = find_missing_tags(prompt)
            attempts += 1

        grounded = False
        if self.ground_context and "<codebase_context>" in prompt:
            prompt = ground_codebase_context(prompt, context.render_summary())
            grounded = True

        return CompileResult(
            prompt=prompt,
            raw=raw,
            model=str(getattr(self.client, "model", "unknown")),
            attempts=attempts,
            grounded=grounded,
            missing_tags=missing,
        )
