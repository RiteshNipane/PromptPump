"""Tests for inference telemetry; a fake Ollama server stands in for the real one."""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from promptpump.cli import app
from promptpump.engine import OllamaClient, OllamaError
from promptpump.telemetry import (
    END_MARKER,
    START_MARKER,
    InferenceMetrics,
    TelemetryReport,
    VramReading,
    VramSampler,
    _settle,
    collect_hardware,
    percentile,
    render_markdown,
    run_telemetry,
    update_readme,
)

FINAL = {
    "message": {"role": "assistant", "content": ""}, "done": True,
    "total_duration": 2_000_000_000, "load_duration": 500_000_000,
    "prompt_eval_count": 120, "prompt_eval_duration": 400_000_000,
    "eval_count": 40, "eval_duration": 800_000_000,
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass

    def _json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/version":
            self._json({"version": "0.9.9-test"})
        elif self.path == "/api/ps":
            self._json({"models": [{"name": "fake:1b", "size": 4 * 2**30, "size_vram": 3 * 2**30}]})
        elif self.path == "/api/tags":
            self._json({"models": [{"name": "fake:1b"}]})
        else:
            self._json({}, 404)

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/api/chat":
            if body.get("model") == "oom":
                return self._json({"error": "llama-server process has terminated: failed to allocate buffer of size 4292825088"}, 500)
            if body.get("model") == "missing":
                return self._json({"error": "nope"}, 404)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for piece in ("<role_definition>", "hello"):
                chunk = {"message": {"role": "assistant", "content": piece}, "done": False}
                self.wfile.write((json.dumps(chunk) + "\n").encode())
                self.wfile.flush()
                time.sleep(0.05)
            self.wfile.write((json.dumps(FINAL) + "\n").encode())
        elif self.path == "/api/show":
            self._json({"details": {"parameter_size": "1B", "quantization_level": "Q4_K_M"}})
        else:
            self._json({"done": True})


@pytest.fixture()
def server() -> str:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_metrics_from_chunk() -> None:
    m = InferenceMetrics.from_chunk(FINAL, model="m", ttft_s=0.7, total_s=2.0)
    assert m.tokens_per_s == pytest.approx(50.0) and m.prompt_tokens_per_s == pytest.approx(300.0)
    assert m.load_s == pytest.approx(0.5) and m.prompt_tokens == 120 and m.output_tokens == 40
    empty = InferenceMetrics.from_chunk({}, model="m", ttft_s=None, total_s=1.0)
    assert empty.tokens_per_s is None and empty.prompt_tokens is None


def test_streaming_chat_records_metrics(server: str) -> None:
    client = OllamaClient(host=server, model="fake:1b")
    assert client.chat([{"role": "user", "content": "hi"}]) == "<role_definition>hello"
    m = client.last_metrics
    assert m is not None and m.tokens_per_s == pytest.approx(50.0)
    assert m.ttft_s is not None and 0 <= m.ttft_s < m.total_s


def test_chat_errors(server: str) -> None:
    with pytest.raises(OllamaError, match="not found"):
        OllamaClient(host=server, model="missing").chat([{"role": "user", "content": "x"}])
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(OllamaError, match="Cannot reach"):
        OllamaClient(host=f"http://127.0.0.1:{port}", model="m").chat([{"role": "user", "content": "x"}])


def test_collect_hardware(server: str) -> None:
    hw = collect_hardware(OllamaClient(host=server, model="fake:1b"))
    assert hw["ollama_version"] == "0.9.9-test" and hw["quantization"] == "Q4_K_M"
    assert hw["model_vram_mib"] == 3072 and hw["model_size_mib"] == 4096


def test_vram_sampler_tracks_peak() -> None:
    values = iter([100, 150, 400, 300])
    last = {"v": 300}

    def query() -> int:
        last["v"] = next(values, last["v"])
        return last["v"]

    with VramSampler(query, interval=0.005) as sampler:
        time.sleep(0.15)
    r = sampler.reading
    assert r.available and r.baseline_mib == 100 and r.peak_mib == 400 and r.delta_mib == 300


def test_vram_unavailable_is_reported_not_guessed() -> None:
    with VramSampler(lambda: None) as sampler:
        pass
    assert not sampler.reading.available and sampler.reading.delta_mib is None


def fake_result(i: int = 0) -> SimpleNamespace:
    m = InferenceMetrics("m", 0.5 + i / 10, 2.0, 100, 40, 50.0 - i, 300.0, 0.1, 0.3)
    return SimpleNamespace(metrics=[m])


def test_run_telemetry_marks_cold_and_unloads() -> None:
    client = SimpleNamespace(unloaded=0)
    client.unload = lambda: setattr(client, "unloaded", client.unloaded + 1)
    calls = iter(range(10))
    records = run_telemetry(client=client, compile_fn=lambda: fake_result(next(calls)),
                            runs=4, vram_query=lambda: None)
    assert [r.cold for r in records] == [True, False, False, False] and client.unloaded == 1
    with pytest.raises(RuntimeError):
        run_telemetry(client=client, compile_fn=lambda: SimpleNamespace(metrics=[]), runs=2,
                      vram_query=lambda: None)


def test_percentile() -> None:
    assert percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)
    assert percentile([], 50) is None and percentile([7], 95) == 7


def test_render_and_update_readme(tmp_path: Path) -> None:
    calls = iter(range(10))
    records = run_telemetry(client=SimpleNamespace(), compile_fn=lambda: fake_result(next(calls)),
                            runs=4, unload=False, vram_query=lambda: None)
    md = render_markdown(TelemetryReport({"gpu": None, "model": "fake:1b", "os": "test"}, "p", records))
    assert md.startswith(START_MARKER) and md.endswith(END_MARKER)
    assert "Decode speed" in md and "N/A" in md and "none detected" in md

    readme = tmp_path / "README.md"
    readme.write_text(f"before\n{START_MARKER}\nstale-text\n{END_MARKER}\nafter\n")
    update_readme(readme, md)
    text = readme.read_text()
    assert text.startswith("before") and text.endswith("after\n") and "stale-text" not in text
    bare = tmp_path / "bare.md"
    bare.write_text("no markers")
    with pytest.raises(ValueError):
        update_readme(bare, md)


def test_cli_telemetry_end_to_end(server: str, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "app.py").write_text("def main(): ...\n")
    readme = tmp_path / "README.md"
    readme.write_text(f"{START_MARKER}\nplaceholder\n{END_MARKER}\n")
    result = CliRunner().invoke(app, [
        "telemetry", "--path", str(ws), "--host", server, "--model", "fake:1b",
        "--runs", "3", "--write-readme", str(readme),
    ])
    assert result.exit_code == 0, result.output
    assert "Cold start" in readme.read_text() and "placeholder" not in readme.read_text()


def test_warm_vram_delta_is_measured_against_pre_load_idle() -> None:
    calls = {"n": 0}

    def query() -> int:
        calls["n"] += 1
        return 100 if calls["n"] == 1 else 900  # idle once, then the model is resident

    results = iter(range(10))

    def compile_fn() -> SimpleNamespace:
        time.sleep(0.03)
        return fake_result(next(results))

    records = run_telemetry(client=SimpleNamespace(), compile_fn=compile_fn, runs=3,
                            unload=False, vram_query=query, interval=0.005)
    assert [r.vram.delta_mib for r in records] == [800, 800, 800]


def test_render_shows_model_placement() -> None:
    records = run_telemetry(client=SimpleNamespace(), compile_fn=lambda: fake_result(),
                            runs=2, unload=False, vram_query=lambda: None)
    hw = {"gpu": None, "model": "m", "model_size_mib": 9000, "model_vram_mib": 4500}
    assert "4,500 of 9,000 MiB on GPU (50%)" in render_markdown(TelemetryReport(hw, "p", records))


def test_settle_waits_for_memory_to_stop_changing() -> None:
    readings = iter([5000, 3000, 600, 590, 585, 585, 585])
    calls = {"n": 0}

    def query() -> int:
        calls["n"] += 1
        return next(readings, 585)

    _settle(query, timeout=5, interval=0.001)
    assert 4 <= calls["n"] <= 7  # returned once stable, not after the full timeout
    _settle(lambda: None, timeout=5, interval=0.001)  # no GPU: returns immediately


def test_cold_start_prefers_unload_all() -> None:
    log: list[str] = []
    client = SimpleNamespace(unload=lambda: log.append("one"), unload_all=lambda: log.append("all"))
    run_telemetry(client=client, compile_fn=lambda: fake_result(), runs=2, vram_query=lambda: None)
    assert log == ["all"]


def test_unload_all_hits_every_loaded_model(server: str) -> None:
    OllamaClient(host=server, model="fake:1b").unload_all()  # must not raise


def test_out_of_memory_gets_an_actionable_message(server: str) -> None:
    with pytest.raises(OllamaError, match="ran out of memory.*ollama stop"):
        OllamaClient(host=server, model="oom").chat([{"role": "user", "content": "x"}])


def test_num_ctx_is_sent_to_ollama() -> None:
    assert OllamaClient(num_ctx=4096).options["num_ctx"] == 4096


def test_run_writes_output_and_log_files_as_utf8(server: str, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    (ws / "pkg").mkdir(parents=True)
    (ws / "pkg" / "a.py").write_text("def f(): ...\n")
    out, log = tmp_path / "out.md", tmp_path / "notes.txt"
    result = CliRunner().invoke(app, [
        "run", "add x", "--path", str(ws), "--host", server, "--model", "fake:1b",
        "--no-copy", "--output", str(out), "--log", str(log),
    ])
    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8").startswith("<role_definition>hello")
    assert "Compiled" in log.read_text(encoding="utf-8")


def test_dry_run_output_file_keeps_the_unicode_tree(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    (ws / "pkg").mkdir(parents=True)
    (ws / "pkg" / "a.py").write_text("def f(): ...\n")
    out = tmp_path / "payload.txt"
    result = CliRunner().invoke(app, ["run", "add x", "--path", str(ws), "--dry-run", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert "\u2514\u2500\u2500" in out.read_text(encoding="utf-8")  # the tree's box characters
