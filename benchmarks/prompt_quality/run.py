"""Run a small, test-scored prompt-quality pilot against a local Ollama model.

This is a development benchmark, not part of PromptPump's installed package.
Each task is generated from the same model and scored by its fixture's pytest suite.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from promptpump.context import scan_workspace  # noqa: E402
from promptpump.engine import OllamaClient, OllamaError, PromptCompiler  # noqa: E402

ROOT = Path(__file__).resolve().parent
TASKS = ROOT / "tasks"
DEFAULT_MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"
CONDITIONS = ("raw_request", "repository_dump", "promptpump")

GENERATION_SYSTEM = """\
You are a careful Python coding assistant modifying a small existing project.
Implement the user's task using the supplied project context when present.
The benchmark target is exactly main.py. Do not change tests or create files.
Return ONLY the complete replacement source for main.py inside one ```python code
block, even if an embedded prompt suggests a diff or another output format. Do not
include explanations before or after the code block.
"""


def _project_dump(project: Path) -> str:
    sections: list[str] = []
    for path in sorted(project.rglob("*")):
        if not path.is_file() or any(
            part in {"__pycache__", ".pytest_cache"} for part in path.parts
        ):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(project).as_posix()
        content = path.read_text(encoding="utf-8")
        sections.append(f"### {relative}\n```text\n{content.rstrip()}\n```")
    return "\n\n".join(sections)


def _model_source(text: str) -> str:
    blocks = re.findall(r"```[^\r\n]*\r?\n(.*?)```", text, flags=re.DOTALL)
    if len(blocks) != 1 or not blocks[0].strip():
        raise ValueError("response must contain exactly one non-empty source code block")
    return blocks[0].strip() + "\n"


def _evaluate_generation(
    project: Path,
    result_files: dict[str, str],
    allowed_files: set[str],
) -> tuple[bool, str]:
    unexpected = set(result_files) - allowed_files
    if unexpected:
        raise ValueError(f"model returned disallowed paths: {', '.join(sorted(unexpected))}")
    if set(result_files) != allowed_files:
        missing = allowed_files - set(result_files)
        raise ValueError(f"model omitted required files: {', '.join(sorted(missing))}")

    with tempfile.TemporaryDirectory(prefix="promptpump-ab-") as temporary:
        worktree = Path(temporary) / "project"
        shutil.copytree(project, worktree)
        for relative, content in result_files.items():
            target = worktree / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        try:
            process = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--disable-warnings", "--maxfail=1"],
                cwd=worktree,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + "\npytest timed out after 120 seconds"
            return False, output[-1_200:]
        output = f"{process.stdout}\n{process.stderr}".strip()
        return process.returncode == 0, output[-1_200:]


def _task_dirs() -> list[Path]:
    return sorted(path for path in TASKS.iterdir() if path.is_dir())


def run(model: str, host: str, timeout: float, max_context_tokens: int) -> dict[str, Any]:
    client = OllamaClient(model=model, host=host, timeout=timeout)
    ollama_version = client.version()
    compiler = PromptCompiler(client)
    records: list[dict[str, Any]] = []

    for task_dir in _task_dirs():
        request = (task_dir / "request.txt").read_text(encoding="utf-8").strip()
        metadata = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        project = task_dir / "project"
        allowed_files = set(metadata["allowed_files"])
        dump = _project_dump(project)
        context = scan_workspace(
            project, query=request, max_tokens=max_context_tokens
        )

        for condition in CONDITIONS:
            record: dict[str, Any] = {
                "task": task_dir.name,
                "condition": condition,
                "passed": False,
                "generation_seconds": None,
                "compile_seconds": None,
                "error": None,
                "test_output": "",
            }
            try:
                if condition == "raw_request":
                    user_content = request
                elif condition == "repository_dump":
                    user_content = f"{request}\n\nRepository dump:\n{dump}"
                else:
                    compile_started = time.perf_counter()
                    compiled = compiler.compile(request, context)
                    record["compile_seconds"] = round(
                        time.perf_counter() - compile_started, 2
                    )
                    if not compiled.is_valid:
                        raise OllamaError(
                            "PromptPump output is missing required sections: "
                            + ", ".join(compiled.missing_tags)
                        )
                    record["fallback_tags"] = compiled.fallback_tags
                    user_content = compiled.prompt

                generation_started = time.perf_counter()
                response = client.chat(
                    [
                        {"role": "system", "content": GENERATION_SYSTEM},
                        {"role": "user", "content": user_content},
                    ]
                )
                record["generation_seconds"] = round(
                    time.perf_counter() - generation_started, 2
                )
                files = {next(iter(allowed_files)): _model_source(response)}
                passed, test_output = _evaluate_generation(project, files, allowed_files)
                record["passed"] = passed
                record["test_output"] = test_output
            except (OllamaError, ValueError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            status = "PASS" if record["passed"] else "FAIL"
            detail = record["error"] or ""
            if not record["passed"] and not detail and record["test_output"]:
                detail = record["test_output"].splitlines()[-1]
            suffix = f" — {detail}" if detail else ""
            print(f"{task_dir.name:16} {condition:16} {status}{suffix}", flush=True)

    totals = {
        condition: {
            "passed": sum(
                record["passed"]
                for record in records
                if record["condition"] == condition
            ),
            "total": sum(record["condition"] == condition for record in records),
        }
        for condition in CONDITIONS
    }
    return {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "ollama_version": ollama_version,
        "tasks": [path.name for path in _task_dirs()],
        "scoring": "Each condition generates code once per task; pass requires all fixture pytest tests to pass.",
        "totals": totals,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-context-tokens", type=int, default=2500)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "0.2.0-pilot.json",
    )
    args = parser.parse_args()
    report = run(args.model, args.host, args.timeout, args.max_context_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nCondition scores:")
    for condition, score in report["totals"].items():
        print(f"{condition:16} {score['passed']}/{score['total']} tests passed")
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
