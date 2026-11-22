"""Unit tests for PromptPump (no Ollama server required)."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

import promptpump.cli as cli_module
from promptpump.cli import app
from promptpump.context import outline_python, scan_workspace
from promptpump.engine import (
    PromptCompiler,
    PromptPumpError,
    clean_response,
    find_missing_tags,
)
from promptpump.stack_profiles import resolve_stacks
from promptpump.templates import (
    REQUIRED_TAGS,
    PromptMode,
    UnknownStackError,
    build_system_prompt,
    get_required_tags,
    get_stack,
    list_stacks,
)


def make_prompt(body: str = "content", tags: tuple[str, ...] = REQUIRED_TAGS) -> str:
    return "\n".join(f"<{t}>\n{body}\n</{t}>" for t in tags)


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


def test_creative_system_prompts_use_distinct_schemas() -> None:
    image = build_system_prompt(mode=PromptMode.IMAGE)
    content = build_system_prompt(mode=PromptMode.CONTENT)
    assert "composition_and_lighting" in image and "IMAGE PROMPT RULES" in image
    assert "content_structure" in content and "stories, scripts" in content
    assert "<codebase_context>" not in image + content
    assert get_required_tags(PromptMode.IMAGE) != get_required_tags(PromptMode.CONTENT)


def test_unknown_stack_raises() -> None:
    with pytest.raises(UnknownStackError):
        get_stack("cobol")


def test_stack_aliases_are_case_normalized_and_deduplicated() -> None:
    profiles = resolve_stacks([
        "C++", "cpp", "Next", "nextjs", "NODE", "nodejs", "Py", "PYTHON",
    ])
    assert [profile.name for profile in profiles] == ["cpp", "nextjs", "node", "python"]
    prompt = build_system_prompt(profiles)
    assert prompt.count("[C++]") == 1
    assert prompt.count("[Next.js]") == 1
    assert prompt.count("[Node.js]") == 1
    assert prompt.count("[Python]") == 1


def test_templates_keeps_stack_public_imports() -> None:
    from promptpump import templates

    assert templates.get_stack("PY").name == "python"
    assert "python" in templates.STACKS
    assert templates.STACK_ALIASES["py"] == "python"


def test_mainstream_stack_profiles_are_available_and_have_rules() -> None:
    profiles = list_stacks()
    names = {profile.name for profile in profiles}
    assert len(profiles) >= 40
    assert {
        "python", "javascript", "typescript", "java", "kotlin", "csharp", "go", "rust",
        "c", "cpp", "php", "ruby", "swift", "dart", "scala", "elixir", "r", "sql",
        "bash", "spring", "aspnet", "angular", "vue", "svelte", "express", "nestjs",
        "flask", "pydantic", "sqlalchemy", "redis", "celery", "graphql", "grpc",
        "react-native", "electron", "postgresql", "mysql", "mongodb", "pandas", "scikit-learn", "tensorflow", "transformers",
        "pytest", "playwright", "docker", "terraform",
    } <= names
    assert all(profile.rules for profile in profiles)
    assert all(get_stack(alias).name == canonical for alias, canonical in {
        "js": "javascript", "ts": "typescript", "golang": "go", "c++": "cpp",
        "c#": "csharp", "dotnet": "csharp", "postgres": "postgresql", "mongo": "mongodb",
        "sklearn": "scikit-learn",
    }.items())


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


def test_compile_image_prompt_uses_its_schema_without_code_grounding(workspace: Path) -> None:
    client = FakeClient(make_prompt("cinematic scene", get_required_tags(PromptMode.IMAGE)))
    result = PromptCompiler(client).compile(
        "a fox in a snowy forest", scan_workspace(workspace), mode=PromptMode.IMAGE
    )
    assert result.is_valid and not result.grounded and result.issues == []
    assert "<subject_and_scene>" in result.prompt
    assert "<codebase_context>" not in client.calls[0][0]["content"]
    assert "<workspace>" not in client.calls[0][1]["content"]


def test_compile_rejects_stack_profiles_for_creative_modes(workspace: Path) -> None:
    compiler = PromptCompiler(FakeClient())
    with pytest.raises(PromptPumpError, match="only available for code"):
        compiler.build_messages("draw a fox", scan_workspace(workspace), ["python"], "image")


def test_compile_retries_when_tags_missing(workspace: Path) -> None:
    client = FakeClient("<role_definition>only</role_definition>", make_prompt())
    result = PromptCompiler(client).compile("x", scan_workspace(workspace))
    assert result.attempts == 2 and result.is_valid
    assert "missing" in client.calls[1][-1]["content"]


def test_compile_reports_missing_after_retries(workspace: Path) -> None:
    client = FakeClient("junk", "still junk", "still invalid")
    result = PromptCompiler(client).compile("x", scan_workspace(workspace))
    assert not result.is_valid and result.missing_tags == list(REQUIRED_TAGS)


def test_compile_recovers_missing_output_format_after_retries(workspace: Path) -> None:
    incomplete = make_prompt().replace(
        "<expected_output_format>\ncontent\n</expected_output_format>", ""
    )
    client = FakeClient(incomplete, incomplete, incomplete)
    result = PromptCompiler(client).compile("add a feature", scan_workspace(workspace))
    assert result.is_valid
    assert result.fallback_tags == ["expected_output_format"]
    assert "<expected_output_format>" in result.prompt
    assert "Follow the output format" in result.prompt


def test_compile_normalizes_an_unclosed_output_format_section(workspace: Path) -> None:
    incomplete = make_prompt().replace(
        "<expected_output_format>\ncontent\n</expected_output_format>",
        "<expected_output_format >\nReturn full source for main.py",
    )
    client = FakeClient(incomplete, incomplete, incomplete)
    result = PromptCompiler(client).compile("add a feature", scan_workspace(workspace))
    assert result.is_valid
    assert result.fallback_tags == ["expected_output_format"]
    assert "<expected_output_format>\nReturn full source for main.py\n</expected_output_format>" in result.prompt


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


def test_cli_dry_run_emits_each_canonical_stack_once(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["run", "add a game loop", "--type", "code", "--stack", "C++", "--stack", "cpp",
         "--path", str(tmp_path), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert result.output.count("[C++]") == 1


def test_cli_image_dry_run_uses_image_schema_without_workspace_scan(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["run", "a fox in a snowy forest", "--type", "image", "--path", str(tmp_path), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "<subject_and_scene>" in result.output
    assert "codebase_context" not in result.output


def test_cli_ignores_stack_with_warning_for_creative_mode(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    result = CliRunner().invoke(
        app,
        ["run", "a fox in a snowy forest", "--type", "image", "--stack", "not-a-stack",
         "--path", str(tmp_path), "--dry-run", "--log", str(log)],
    )
    assert result.exit_code == 0, result.output
    assert "warning" in result.output and "Ignoring --stack" in result.output
    assert "STACK RULES" not in result.output
    assert "Ignoring --stack" in log.read_text(encoding="utf-8")


def test_no_args_menu_selects_content_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_module,
        "OllamaClient",
        lambda **_kwargs: FakeClient(make_prompt("a short adventure", get_required_tags("content"))),
    )
    result = CliRunner().invoke(app, [], input="3\na short adventure story\n")
    assert result.exit_code == 0, result.output
    assert "Content generation" in result.output
    assert "<content_structure>" in result.output


def test_menu_command_selects_image_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_module,
        "OllamaClient",
        lambda **_kwargs: FakeClient(make_prompt("night scene", get_required_tags("image"))),
    )
    result = CliRunner().invoke(app, ["menu"], input="2\na moonlit night scene\n")
    assert result.exit_code == 0, result.output
    assert "<subject_and_scene>" in result.output


def test_stacks_command_and_version() -> None:
    runner = CliRunner()
    stacks = runner.invoke(app, ["stacks"])
    version = runner.invoke(app, ["--version"])
    assert stacks.exit_code == 0 and "typescript" in stacks.output and "react-native" in stacks.output
    assert version.exit_code == 0 and version.output.startswith("promptpump ")


def test_doctor_command_uses_mocked_local_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOllama:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def list_models(self) -> list[str]:
            return ["qwen2.5-coder:7b"]

    monkeypatch.setattr(cli_module, "OllamaClient", FakeOllama)
    result = CliRunner().invoke(app, ["doctor", "--model", "qwen2.5-coder:7b"])
    assert result.exit_code == 0, result.output
    assert "Ollama reachable" in result.output and "is installed" in result.output


def test_telemetry_command_with_mocked_components(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(
        cli_module, "OllamaClient", lambda **_kwargs: SimpleNamespace(model="fake-model")
    )
    monkeypatch.setattr(cli_module, "run_telemetry", lambda **_kwargs: [])
    monkeypatch.setattr(cli_module, "collect_hardware", lambda _client: {})
    monkeypatch.setattr(cli_module, "render_markdown", lambda _report: "# mocked telemetry")
    result = CliRunner().invoke(
        app, ["telemetry", "--path", str(tmp_path), "--runs", "2"]
    )
    assert result.exit_code == 0, result.output
    assert "# mocked telemetry" in result.output


def test_cli_rejects_unknown_stack(workspace: Path) -> None:
    result = CliRunner().invoke(app, ["run", "x", "--stack", "cobol", "--path", str(workspace)])
    assert result.exit_code == 2
