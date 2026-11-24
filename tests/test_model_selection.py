from typer.testing import CliRunner

from promptpump import config
from promptpump.cli import app
from promptpump.engine import OllamaClient
from promptpump.model_selection import available_chat_models, select_installed_model


def test_missing_default_falls_back_to_installed_deepseek() -> None:
    assert (
        select_installed_model(["deepseek-r1:14b"], "qwen2.5-coder:7b", prompt_type="code")
        == "deepseek-r1:14b"
    )


def test_available_tag_in_preferred_family_wins() -> None:
    assert (
        select_installed_model(
            ["deepseek-r1:8b", "qwen2.5-coder:14b"],
            "qwen2.5-coder:7b",
        )
        == "qwen2.5-coder:14b"
    )


def test_embedding_models_are_not_selected_as_chat_models() -> None:
    assert (
        select_installed_model(["nomic-embed-text:latest", "llama3.2:3b"], "qwen2.5-coder:7b")
        == "llama3.2:3b"
    )


def test_no_chat_models_returns_none() -> None:
    assert select_installed_model(["nomic-embed-text:latest"], "qwen2.5-coder:7b") is None


def test_default_client_resolves_to_installed_model(monkeypatch) -> None:
    client = OllamaClient()
    monkeypatch.setattr(client, "list_models", lambda: ["deepseek-r1:14b"])
    assert client.resolve_model() == "deepseek-r1:14b"
    assert client.model_was_fallback


def test_configured_model_preference_round_trips(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "config.json")
    monkeypatch.delenv("PROMPTPUMP_MODEL", raising=False)
    assert config.get_model_preference() == "qwen2.5-coder:7b"
    config.set_model_preference("deepseek-r1:14b")
    assert config.get_model_preference() == "deepseek-r1:14b"


def test_model_command_saves_interactive_selection(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "config.json")
    monkeypatch.delenv("PROMPTPUMP_MODEL", raising=False)

    class FakeOllama:
        def __init__(self, **_kwargs):
            pass

        def list_models(self):
            return ["qwen2.5-coder:7b", "deepseek-r1:14b"]

    from promptpump import cli as cli_module

    monkeypatch.setattr(cli_module, "OllamaClient", FakeOllama)
    result = CliRunner().invoke(app, ["model"], input="2\n")
    assert result.exit_code == 0, result.output
    assert "Current model set to deepseek-r1:14b" in result.output
    assert config.get_model_preference() == "deepseek-r1:14b"


def test_chat_model_list_excludes_embedding_tags() -> None:
    assert available_chat_models(["nomic-embed-text:latest", "deepseek-r1:8b"]) == [
        "deepseek-r1:8b"
    ]
