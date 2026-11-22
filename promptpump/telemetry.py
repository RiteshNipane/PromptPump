"""Inference telemetry: TTFT, tokens/sec, token counts, and peak GPU memory.

Latency and token figures come from Ollama itself (``eval_count``/``eval_duration`` etc.)
plus a wall-clock timer around the streamed response. GPU memory is sampled from
``nvidia-smi``; on machines without an NVIDIA GPU it is reported as unavailable, never guessed.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

START_MARKER = "<!-- promptpump:telemetry:start -->"
END_MARKER = "<!-- promptpump:telemetry:end -->"


# ---------------------------------------------------------------------------------------
# Per-request metrics
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceMetrics:
    """Timing and token counts for one model call."""

    model: str
    ttft_s: float | None  # wall clock: request sent -> first content token (includes model load)
    total_s: float
    prompt_tokens: int | None  # Ollama prompt_eval_count (absent when the prompt was cached)
    output_tokens: int | None  # Ollama eval_count
    tokens_per_s: float | None  # eval_count / eval_duration (decode speed)
    prompt_tokens_per_s: float | None
    load_s: float | None
    prompt_eval_s: float | None

    @classmethod
    def from_chunk(
        cls, final: dict[str, Any], *, model: str, ttft_s: float | None, total_s: float
    ) -> "InferenceMetrics":
        """Build metrics from Ollama's final (``done: true``) streaming chunk."""

        def seconds(key: str) -> float | None:
            value = final.get(key)
            return value / 1e9 if isinstance(value, (int, float)) else None

        eval_s, prompt_s = seconds("eval_duration"), seconds("prompt_eval_duration")
        out, prompt = final.get("eval_count"), final.get("prompt_eval_count")
        return cls(
            model=model,
            ttft_s=ttft_s,
            total_s=total_s,
            prompt_tokens=prompt,
            output_tokens=out,
            tokens_per_s=out / eval_s if out and eval_s else None,
            prompt_tokens_per_s=prompt / prompt_s if prompt and prompt_s else None,
            load_s=seconds("load_duration"),
            prompt_eval_s=prompt_s,
        )


# ---------------------------------------------------------------------------------------
# GPU memory
# ---------------------------------------------------------------------------------------


def _run_nvidia_smi(fields: str) -> list[list[str]] | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return [[c.strip() for c in line.split(",")] for line in proc.stdout.strip().splitlines()]


def query_nvidia_smi() -> int | None:
    """Total GPU memory in use across all NVIDIA GPUs, in MiB (``None`` if unavailable)."""
    rows = _run_nvidia_smi("memory.used")
    if not rows:
        return None
    try:
        return sum(int(r[0]) for r in rows)
    except (ValueError, IndexError):
        return None


def query_gpu_info() -> tuple[str, int] | None:
    """(name, total MiB) of the first NVIDIA GPU, or ``None``."""
    rows = _run_nvidia_smi("name,memory.total")
    if not rows:
        return None
    try:
        return rows[0][0], int(rows[0][1])
    except (ValueError, IndexError):
        return None


@dataclass(frozen=True)
class VramReading:
    """Peak GPU memory during a run. ``baseline`` is the idle reading taken just before."""

    available: bool
    baseline_mib: int | None = None
    peak_mib: int | None = None

    @property
    def delta_mib(self) -> int | None:
        if self.baseline_mib is None or self.peak_mib is None:
            return None
        return max(0, self.peak_mib - self.baseline_mib)


class VramSampler:
    """Polls GPU memory in a background thread and records the peak."""

    def __init__(
        self, query: Callable[[], int | None] = query_nvidia_smi, interval: float = 0.1
    ) -> None:
        self._query, self.interval = query, interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline: int | None = None
        self._peak: int | None = None
        self.reading = VramReading(available=False)

    def start(self) -> "VramSampler":
        self._baseline = self._peak = self._query()
        self._stop.clear()
        if self._baseline is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._observe()

    def _observe(self) -> None:
        value = self._query()
        if value is not None and (self._peak is None or value > self._peak):
            self._peak = value

    def stop(self) -> VramReading:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._observe()
        self.reading = VramReading(self._baseline is not None, self._baseline, self._peak)
        return self.reading

    def __enter__(self) -> "VramSampler":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


# ---------------------------------------------------------------------------------------
# Hardware description
# ---------------------------------------------------------------------------------------


def total_ram_gb() -> float | None:
    """Installed physical RAM in GiB, or ``None`` if it cannot be determined."""
    try:
        if sys.platform == "win32":  # pragma: no cover - exercised on Windows only

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            status = _MemStatus()
            status.dwLength = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
            return round(status.ullTotalPhys / 2**30, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (OSError, ValueError, AttributeError):
        return None


def cpu_name() -> str:
    """Human-readable CPU model (registry on Windows, /proc/cpuinfo on Linux)."""
    if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def collect_hardware(client: Any) -> dict[str, Any]:
    """Best-effort description of the machine and model, for reproducibility."""

    def probe(fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception:  # every probe is optional
            return None

    gpu = query_gpu_info()
    memory = (probe(client.model_memory) if hasattr(client, "model_memory") else None) or {}
    details = (probe(client.show) or {}).get("details", {}) if hasattr(client, "show") else {}
    return {
        "os": platform.platform(),
        "cpu": cpu_name(),
        "ram_gb": total_ram_gb(),
        "gpu": gpu[0] if gpu else None,
        "gpu_total_mib": gpu[1] if gpu else None,
        "ollama_version": probe(client.version) if hasattr(client, "version") else None,
        "model": getattr(client, "model", None),
        "parameter_size": details.get("parameter_size"),
        "quantization": details.get("quantization_level"),
        "model_size_mib": _mib(memory.get("size")),
        "model_vram_mib": _mib(memory.get("size_vram")),
    }


def _mib(value: int | None) -> int | None:
    return round(value / 2**20) if value else None


# ---------------------------------------------------------------------------------------
# Repeated runs and reporting
# ---------------------------------------------------------------------------------------


@dataclass
class RunRecord:
    index: int
    cold: bool
    metrics: InferenceMetrics
    vram: VramReading


@dataclass
class TelemetryReport:
    hardware: dict[str, Any]
    prompt: str
    records: list[RunRecord] = field(default_factory=list)


def _settle(
    query: Callable[[], int | None], timeout: float, interval: float, tolerance_mib: int = 16
) -> None:
    """Wait until GPU memory stops changing (a model has finished unloading)."""
    previous = query()
    if previous is None:
        return
    stable, deadline = 0, time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(interval)
        current = query()
        if current is None:
            return
        stable = stable + 1 if abs(current - previous) <= tolerance_mib else 0
        previous = current
        if stable >= 2:
            return


def run_telemetry(
    *,
    client: Any,
    compile_fn: Callable[[], Any],
    runs: int,
    unload: bool = True,
    vram_query: Callable[[], int | None] = query_nvidia_smi,
    interval: float = 0.1,
    progress: Callable[[int, int], None] | None = None,
    settle_timeout: float = 10.0,
    settle_interval: float = 0.5,
) -> list[RunRecord]:
    """Run ``compile_fn`` ``runs`` times; run 0 is a cold start (model unloaded first)."""
    records: list[RunRecord] = []
    idle_mib: int | None = None  # GPU memory before the model was loaded (run 0 only)
    for i in range(runs):
        if progress:
            progress(i + 1, runs)
        if i == 0 and unload:
            evict = getattr(client, "unload_all", None) or getattr(client, "unload", None)
            if evict is not None:
                evict()
                _settle(vram_query, settle_timeout, settle_interval)
        sampler = VramSampler(vram_query, interval)
        with sampler:
            result = compile_fn()
        if not getattr(result, "metrics", None):
            raise RuntimeError("The client returned no inference metrics.")
        reading = sampler.reading
        if i == 0:
            idle_mib = reading.baseline_mib
        elif reading.available and idle_mib is not None:
            # Warm runs start with the model already resident, so their own baseline would
            # hide it. Measure every run against the pre-load idle reading instead.
            reading = VramReading(True, idle_mib, reading.peak_mib)
        records.append(RunRecord(i, i == 0, result.metrics[0], reading))
    return records


def percentile(values: Sequence[float], pct: float) -> float | None:
    """Linear-interpolated percentile (``None`` for empty input)."""
    data = sorted(values)
    if not data:
        return None
    k = (len(data) - 1) * pct / 100
    low, high = int(k), min(int(k) + 1, len(data) - 1)
    return data[low] + (data[high] - data[low]) * (k - low)


def _fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.{digits}f}" if isinstance(value, float) else f"{value:,}"


def render_markdown(report: TelemetryReport) -> str:
    """Render the telemetry table (wrapped in README markers)."""
    recs = report.records
    cold, warm = recs[0], recs[1:]
    hw = report.hardware

    def col(getter: Callable[[RunRecord], float | int | None]) -> list[float]:
        return [v for v in (getter(r) for r in warm) if v is not None]

    def med(vals: list[float]) -> float | None:
        return statistics.median(vals) if vals else None

    ttft, tps = col(lambda r: r.metrics.ttft_s), col(lambda r: r.metrics.tokens_per_s)
    total = col(lambda r: r.metrics.total_s)
    in_tok, out_tok = col(lambda r: r.metrics.prompt_tokens), col(lambda r: r.metrics.output_tokens)
    peaks = col(lambda r: r.vram.peak_mib)
    deltas = col(lambda r: r.vram.delta_mib)
    m = cold.metrics

    rows = [
        ("TTFT (s)", _fmt(m.ttft_s), _fmt(med(ttft)), _fmt(percentile(ttft, 95))),
        ("Decode speed (tokens/s)", _fmt(m.tokens_per_s, 1), _fmt(med(tps), 1), _fmt(percentile(tps, 5), 1)),
        ("Time for the model call (s)", _fmt(m.total_s), _fmt(med(total)), _fmt(percentile(total, 95))),
        ("Input tokens (model-side)", _fmt(m.prompt_tokens), _fmt(med(in_tok), 0), "-"),
        ("Output tokens", _fmt(m.output_tokens), _fmt(med(out_tok), 0), "-"),
        (
            "Peak GPU memory, absolute (MiB)",
            _fmt(cold.vram.peak_mib), _fmt(med(peaks), 0), _fmt(max(peaks) if peaks else None),
        ),
        (
            "Peak GPU memory, delta over pre-load idle (MiB)",
            _fmt(cold.vram.delta_mib), _fmt(med(deltas), 0), _fmt(max(deltas) if deltas else None),
        ),
    ]
    table = "\n".join(
        ["| Metric | Cold start (run 1) | Warm median | Warm worst case |", "| --- | ---: | ---: | ---: |"]
        + [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    )
    gpu = f"{hw['gpu']} ({hw['gpu_total_mib']:,} MiB)" if hw.get("gpu") else "none detected (GPU memory N/A)"
    model = " ".join(str(x) for x in (hw.get("model"), hw.get("parameter_size"), hw.get("quantization")) if x)
    placement = ""
    if hw.get("model_size_mib") and hw.get("model_vram_mib") is not None:
        pct = 100 * hw["model_vram_mib"] / hw["model_size_mib"]
        placement = (
            f"- **Model placement:** {hw['model_vram_mib']:,} of {hw['model_size_mib']:,} MiB "
            f"on GPU ({pct:.0f}%), remainder on CPU/RAM\n"
        )
    hardware = "\n".join(
        [
            f"- **Model:** {model or 'unknown'}",
            f"- **GPU:** {gpu}",
            *([placement.rstrip("\n")] if placement else []),
            f"- **CPU / RAM:** {hw.get('cpu') or 'unknown'} / {_fmt(hw.get('ram_gb'), 1)} GB",
            f"- **OS / Ollama:** {hw.get('os')} / {hw.get('ollama_version') or 'unknown'}",
            f"- **Runs:** {len(recs)} ({len(warm)} warm), shorthand: `{report.prompt}`",
        ]
    )
    notes = (
        "_TTFT is wall-clock time from request to first streamed token (cold start includes model "
        "load). Decode speed is Ollama's `eval_count / eval_duration`; its worst case is the 5th "
        "percentile, latency worst cases are the 95th. GPU memory is read from `nvidia-smi` for the "
        "whole GPU, so use the delta column when other programs share it._"
    )
    return f"{START_MARKER}\n{hardware}\n\n{table}\n\n{notes}\n{END_MARKER}"


def update_readme(path: Path, markdown: str) -> None:
    """Replace the marked telemetry block in ``path`` (raises if the markers are missing)."""
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"{path} has no {START_MARKER} ... {END_MARKER} block to update.")
    path.write_text(pattern.sub(lambda _m: markdown, text, count=1), encoding="utf-8")
