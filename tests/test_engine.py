"""Unit tests for PromptPump (no Ollama server required)."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from promptpump.cli import app
from promptpump.context import outline_python, scan_workspace
from promptpump.engine import (
    PromptCompiler,
    PromptPumpError,
    clean_response,
    find_missing_tags,
)
from promptpump.templates import (
    REQUIRED_TAGS,
    UnknownStackError,
    build_system_prompt,
    get_stack,
)


def make_prompt(body: str = "content") -> str:
    return "\n".join(f"<{t}>\n{body}\n</{t}>" for t in REQUIRED_TAGS)


class FakeClient:
    """Returns canned replies in order and records every call."""

    model = "fake-model"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.replies.pop(0)


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_text("# Demo\nA demo project.")
    (tmp_path / "app.py").write_text('def run(x: int) -> int:\n    """Run it."""\n    return x\n')
    (tmp_path / ".gitignore").write_text("secret_dir/\n*.log\n")
    (tmp_path / "secret_dir").mkdir()
    (tmp_path / "secret_dir" / "hidden.py").write_text("x = 1")
    (tmp_path / "debug.log").write_text("log")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("x")
    (tmp_path / ".env").write_text("API_KEY=supersecret")
    return tmp_path


# -- templates ---------------------------------------------------------------------------


def test_system_prompt_contains_all_tags_and_stack_rules() -> None:
    prompt = build_system_prompt([get_stack("fastapi")])
    for tag in REQUIRED_TAGS:
        assert f"<{tag}>" in prompt
    assert "Pydantic" in prompt


def test_system_prompt_is_deterministic() -> None:
    assert build_system_prompt([get_stack("py")]) == build_system_prompt([get_stack("python")])


def test_unknown_stack_raises() -> None:
    with pytest.raises(UnknownStackError):
        get_stack("cobol")


# -- context -----------------------------------------------------------------------------


def test_scanner_respects_gitignore_defaults_and_secrets(workspace: Path) -> None:
    ctx = scan_workspace(workspace)
    tree = ctx.tree
    assert "app.py" in tree and "README.md" in tree
    assert "hidden.py" not in tree and "debug.log" not in tree
    assert "node_modules" not in tree
    assert ".env" not in tree
    assert "supersecret" not in ctx.render()


def test_scanner_collects_key_files_and_outline(workspace: Path) -> None:
    ctx = scan_workspace(workspace)
    assert [kf.path for kf in ctx.key_files][0] == "README.md"
    assert "app.py" in ctx.outlines
    assert "def run(x: int) -> int" in ctx.outlines["app.py"][0]


def test_outline_python_handles_classes_and_bad_syntax() -> None:
    src = "class A(B):\n    def go(self): ...\n    def _hidden(self): ...\n"
    assert outline_python(src) == ["class A(B):", "    def go(self)"]
    assert outline_python("def (:") == []


# -- engine ------------------------------------------------------------------------------


def test_clean_response_strips_filler_and_fences() -> None:
    raw = f"Sure! Here is your prompt:\n```xml\n{make_prompt()}\n```\nLet me know if you need more."
    cleaned = clean_response(raw)
    assert cleaned.startswith("<role_definition>")
    assert cleaned.endswith("</expected_output_format>")
    assert "```" not in cleaned and "Sure" not in cleaned


def test_find_missing_tags() -> None:
    partial = "<role_definition>x</role_definition>"
    assert find_missing_tags(partial) == list(REQUIRED_TAGS[1:])
    assert find_missing_tags(make_prompt()) == []


def test_compile_grounds_codebase_context(workspace: Path) -> None:
    client = FakeClient(make_prompt("model text"))
    result = PromptCompiler(client).compile("add caching", scan_workspace(workspace), ["python"])
    assert result.is_valid and result.grounded and result.attempts == 1
    assert "app.py" in result.prompt  # real tree facts injected
    assert "<codebase_context>\nmodel text" not in result.prompt
    assert result.model == "fake-model"
    assert "Python 3.10+" in client.calls[0][0]["content"]


def test_compile_can_skip_grounding(workspace: Path) -> None:
    client = FakeClient(make_prompt("model text"))
    result = PromptCompiler(client, ground_context=False).compile("x", scan_workspace(workspace))
    assert "<codebase_context>\nmodel text" in result.prompt


def test_compile_retries_when_tags_missing(workspace: Path) -> None:
    client = FakeClient("<role_definition>only</role_definition>", make_prompt())
    result = PromptCompiler(client).compile("x", scan_workspace(workspace))
    assert result.attempts == 2 and result.is_valid
    assert "missing" in client.calls[1][-1]["content"]


def test_compile_reports_missing_after_retries(workspace: Path) -> None:
    client = FakeClient("junk", "still junk")
    result = PromptCompiler(client).compile("x", scan_workspace(workspace))
    assert not result.is_valid and result.missing_tags == list(REQUIRED_TAGS)


def test_empty_shorthand_rejected(workspace: Path) -> None:
    with pytest.raises(PromptPumpError):
        PromptCompiler(FakeClient()).compile("   ", scan_workspace(workspace))


# -- cli ---------------------------------------------------------------------------------


def test_cli_dry_run(workspace: Path) -> None:
    result = CliRunner().invoke(
        app, ["run", "add auth", "--stack", "fastapi", "--path", str(workspace), "--dry-run"]
    )
    assert result.exit_code == 0
    assert "<shorthand_request>" in result.output and "add auth" in result.output


def test_cli_rejects_unknown_stack(workspace: Path) -> None:
    result = CliRunner().invoke(app, ["run", "x", "--stack", "cobol", "--path", str(workspace)])
    assert result.exit_code == 2
