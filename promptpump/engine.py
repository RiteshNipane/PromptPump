"""Core compiler: builds the payload, talks to Ollama, and post-processes the result."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import requests

from .context import WorkspaceContext
from .telemetry import InferenceMetrics
from .verify import Issue, check_references
from .templates import (
    REQUIRED_TAGS,
    PromptMode,
    build_system_prompt,
    build_user_message,
    get_stack,
    get_required_tags,
)

DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_HOST = "http://localhost:11434"

Message = dict[str, str]


class PromptPumpError(Exception):
    """Base class for all PromptPump errors."""


class OllamaError(PromptPumpError):
    """Raised when the local Ollama server cannot satisfy a request."""


_OOM_MARKERS = ("failed to allocate", "out of memory", "cannot be allocated", "insufficient memory")


class ChatClient(Protocol):
    """Anything that can turn chat messages into a text reply (easy to fake in tests)."""

    def chat(self, messages: list[Message]) -> str: ...


class OllamaClient:
    """Client for Ollama's ``/api/chat`` (streamed, so TTFT can be measured)."""

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
        self.last_metrics: InferenceMetrics | None = None

    def _ollama_error(self, text: str, status: int | None = None) -> OllamaError:
        """Turn an Ollama error body into an actionable message."""
        if any(marker in text.lower() for marker in _OOM_MARKERS):
            return OllamaError(
                f"Ollama ran out of memory loading '{self.model}'. Free memory (`ollama ps`, then "
                "`ollama stop <model>`), close heavy apps, lower --num-ctx, or use a smaller "
                "model such as qwen2.5-coder:7b."
            )
        prefix = f"Ollama returned HTTP {status}: " if status else "Ollama error: "
        return OllamaError(prefix + text[:300])

    def chat(self, messages: list[Message]) -> str:
        """Stream a reply, record :attr:`last_metrics`, and return the full text."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": self.options,
        }
        started = time.perf_counter()
        first: float | None = None
        parts: list[str] = []
        final: dict = {}
        try:
            with requests.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout, stream=True
            ) as resp:
                if resp.status_code == 404:
                    raise OllamaError(f"Model '{self.model}' not found. Run `ollama pull {self.model}`.")
                if not resp.ok:
                    raise self._ollama_error(resp.text, resp.status_code)
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except ValueError as exc:
                        raise OllamaError("Unexpected streaming format from Ollama.") from exc
                    if "error" in chunk:
                        raise self._ollama_error(str(chunk["error"]))
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        if first is None:
                            first = time.perf_counter() - started
                        parts.append(piece)
                    if chunk.get("done"):
                        final = chunk
        except requests.ConnectionError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.host}. Is it running? Start it with `ollama serve`."
            ) from exc
        except requests.Timeout as exc:
            raise OllamaError(f"Ollama timed out after {self.timeout:.0f}s.") from exc
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc

        if not parts:
            raise OllamaError("Ollama returned an empty response.")
        self.last_metrics = InferenceMetrics.from_chunk(
            final, model=self.model, ttft_s=first, total_s=time.perf_counter() - started
        )
        return "".join(parts)

    # -- auxiliary endpoints (used by `doctor` and `telemetry`) -------------------------

    def _request(self, method: str, path: str, **kwargs: object) -> dict:
        try:
            resp = requests.request(method, f"{self.host}{path}", timeout=10, **kwargs)  # type: ignore[arg-type]
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.host}.") from exc

    def list_models(self) -> list[str]:
        """Return the names of locally installed models."""
        return [m["name"] for m in self._request("GET", "/api/tags").get("models", [])]

    def version(self) -> str | None:
        return self._request("GET", "/api/version").get("version")

    def show(self) -> dict:
        """Model details (parameter size, quantization)."""
        return self._request("POST", "/api/show", json={"model": self.model, "name": self.model})

    def model_memory(self) -> dict | None:
        """``{"size": bytes, "size_vram": bytes}`` for the loaded model (``/api/ps``), if loaded."""
        for entry in self._request("GET", "/api/ps").get("models", []):
            if str(entry.get("name", "")).startswith(self.model):
                return {"size": entry.get("size"), "size_vram": entry.get("size_vram")}
        return None

    def model_vram_bytes(self) -> int | None:
        """VRAM Ollama reports for the loaded model, if it is loaded."""
        info = self.model_memory()
        return info["size_vram"] if info else None

    def unload_all(self) -> None:
        """Evict every loaded model, so a leftover one cannot contaminate a cold-start measurement."""
        loaded = [e.get("name") for e in self._request("GET", "/api/ps").get("models", [])]
        for name in [n for n in loaded if n] or [self.model]:
            self._request("POST", "/api/generate", json={"model": name, "keep_alive": 0})

    def unload(self) -> None:
        """Evict the model from memory so the next call is a true cold start."""
        self._request("POST", "/api/generate", json={"model": self.model, "keep_alive": 0})


# ---------------------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"^```[\w-]*\s*\n(.*?)\n?```\s*$", re.DOTALL)
_CONTEXT_RE = re.compile(r"<codebase_context>.*?</codebase_context>", re.DOTALL)


def clean_response(raw: str, required_tags: Sequence[str] = REQUIRED_TAGS) -> str:
    """Strip reasoning blocks, conversational filler, and wrapping code fences."""
    text = _THINK_RE.sub("", raw).strip()

    first = text.find(f"<{required_tags[0]}>")
    if first > 0:  # drop any preamble such as "Sure! Here's your prompt:"
        text = text[first:]
    closing = f"</{required_tags[-1]}>"
    last = text.rfind(closing)
    if last != -1:  # drop trailing remarks such as "Let me know if..."
        text = text[: last + len(closing)]

    fenced = _FENCE_RE.match(text.strip())
    if fenced:
        text = fenced.group(1)
    return text.strip()


def find_missing_tags(
    text: str, required_tags: Sequence[str] = REQUIRED_TAGS
) -> list[str]:
    """Return required tags whose opening or closing marker is absent."""
    return [t for t in required_tags if f"<{t}>" not in text or f"</{t}>" not in text]


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
    metrics: list[InferenceMetrics] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.missing_tags


class PromptCompiler:
    """Combines shorthand + workspace + template rules and runs the local model."""

    def __init__(
        self, client: ChatClient, *, ground_context: bool = True, max_repairs: int = 1,
        verify: bool = True,
    ) -> None:
        self.client = client
        self.verify = verify
        self.ground_context = ground_context
        self.max_repairs = max_repairs

    def build_messages(
        self, shorthand: str, context: WorkspaceContext, stacks: Sequence[str] = (),
        mode: PromptMode | str = PromptMode.CODE,
    ) -> list[Message]:
        """Assemble the system + user messages (also used by ``--dry-run``)."""
        selected_mode = PromptMode(mode)
        if stacks and selected_mode is not PromptMode.CODE:
            raise PromptPumpError("--stack profiles are only available for code prompts.")
        profiles = list(dict.fromkeys(get_stack(s) for s in stacks))
        return [
            {"role": "system", "content": build_system_prompt(profiles, selected_mode)},
            {
                "role": "user",
                "content": build_user_message(
                    shorthand, context.render(), selected_mode
                ),
            },
        ]

    def compile(
        self, shorthand: str, context: WorkspaceContext, stacks: Sequence[str] = (),
        mode: PromptMode | str = PromptMode.CODE,
    ) -> CompileResult:
        """Compile ``shorthand`` into a validated prompt for the selected mode."""
        if not shorthand.strip():
            raise PromptPumpError("Shorthand prompt must not be empty.")

        selected_mode = PromptMode(mode)
        required_tags = get_required_tags(selected_mode)
        messages = self.build_messages(shorthand, context, stacks, selected_mode)
        metrics: list[InferenceMetrics] = []

        def call(msgs: list[Message]) -> str:
            reply = self.client.chat(msgs)
            last = getattr(self.client, "last_metrics", None)
            if last is not None:
                metrics.append(last)
            return reply

        raw = call(messages)
        prompt = clean_response(raw)
        missing = find_missing_tags(prompt, required_tags)
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
                        f"Re-output the COMPLETE prompt with all seven required tags in this "
                        f"order: {', '.join(required_tags)}. Output only the prompt."
                    ),
                },
            ]
            raw = call(messages)
            prompt = clean_response(raw)
            missing = find_missing_tags(prompt, required_tags)
            attempts += 1

        grounded = False
        if (
            selected_mode is PromptMode.CODE
            and self.ground_context
            and "<codebase_context>" in prompt
        ):
            prompt = ground_codebase_context(prompt, context.render_summary())
            grounded = True

        return CompileResult(
            prompt=prompt,
            raw=raw,
            model=str(getattr(self.client, "model", "unknown")),
            attempts=attempts,
            grounded=grounded,
            missing_tags=missing,
            metrics=metrics,
            issues=(
                check_references(prompt, context, shorthand)
                if self.verify and selected_mode is PromptMode.CODE
                else []
            ),
        )
