"""Tests for the compiled-prompt reference checker."""

from pathlib import Path

import pytest

from promptpump.context import scan_workspace
from promptpump.engine import PromptCompiler
from promptpump.templates import REQUIRED_TAGS
from promptpump.verify import check_references


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    for rel, body in {
        "evaluation/generation_eval.py": "def extract_citations(a): pass\ndef citation_metrics(a, n): pass\nclass LLMJudge:\n    pass\n",
        "eval/metrics.py": "def extract_citations(a): pass\ndef citation_precision(a, c): pass\n",
        "api/main.py": "class SourceDoc:\n    pass\nclass QueryResponse:\n    pass\ndef query_rag(r): pass\n",
        "retrieval/generation.py": "def build_prompt(q, c): pass\n",
        "eval/run_eval.py": "def main(): pass\n",
        "evaluation/run_eval.py": "def main(): pass\n",
        "eval/__init__.py": "",
        "api/__init__.py": "",
        "tests/test_generation_eval.py": "def test_x(): pass\n",
    }.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(body)
    return tmp_path


def messages(prompt: str, repo: Path) -> list[str]:
    return [i.message for i in check_references(prompt, scan_workspace(repo, query="citations"))]


def test_flags_symbol_attributed_to_the_wrong_file(repo: Path) -> None:
    prompt = (
        "- Update the `citation_metrics` function in `eval/metrics.py`.\n"
        "- Assume `SourceDoc` and `LLMJudge` exist in `api/main.py`.\n"
    )
    found = messages(prompt, repo)
    assert any("citation_metrics" in m and "evaluation/generation_eval.py" in m for m in found)
    assert any("LLMJudge" in m for m in found)
    assert not any("SourceDoc" in m for m in found)


def test_correct_references_raise_nothing(repo: Path) -> None:
    prompt = (
        "- Reuse `extract_citations` in `evaluation/generation_eval.py`.\n"
        "- Change `build_prompt` in `retrieval/generation.py` and `QueryResponse` in `api/main.py`.\n"
    )
    assert messages(prompt, repo) == []  # extract_citations is defined in both files: either is fine


def test_file_block_supplies_the_expected_path(repo: Path) -> None:
    prompt = '<file path="eval/metrics.py">\n- Update the `citation_metrics` function.\n</file>\n'
    assert any("citation_metrics" in m for m in messages(prompt, repo))
    outside = '<file path="eval/metrics.py">\n</file>\nUpdate the `citation_metrics` function.\n'
    assert messages(outside, repo) == []  # block closed: no expected path any more


def test_flags_missing_paths_but_not_basenames_that_resolve(repo: Path) -> None:
    found = messages("See `retrieval/nope.py` and `generation.py`.", repo)
    assert found == ["retrieval/nope.py does not exist in the repo (fine if the plan creates it)"]


def test_compiler_attaches_issues_and_can_disable_them(repo: Path) -> None:
    body = "- Update `citation_metrics` in `eval/metrics.py`."
    prompt = "\n".join(f"<{t}>\n{body}\n</{t}>" for t in REQUIRED_TAGS)

    class Fake:
        model = "fake"

        def chat(self, messages: list) -> str:
            return prompt

    ctx = scan_workspace(repo, query="citations")
    assert PromptCompiler(Fake(), ground_context=False).compile("x", ctx).issues
    assert PromptCompiler(Fake(), ground_context=False, verify=False).compile("x", ctx).issues == []


def scope_messages(prompt: str, repo: Path, shorthand: str) -> list[str]:
    ctx = scan_workspace(repo, query=shorthand)
    return [i.message for i in check_references(prompt, ctx, shorthand) if i.kind == "scope"]


def test_scope_warning_for_unrequested_eval_edits(repo: Path) -> None:
    plan = "<implementation_steps>\n1. Update `evaluate_answer` in `evaluation/generation_eval.py`.\n</implementation_steps>"
    assert scope_messages(plan, repo, "enforce citations in the generation prompt") == [
        "the plan edits evaluation/generation_eval.py, but your request does not mention evaluation or benchmarks"]
    assert scope_messages(plan, repo, "improve the evaluation metrics") == []


def test_scope_ignores_context_sections_and_normal_files(repo: Path) -> None:
    prompt = ("<codebase_context>\n- evaluation/generation_eval.py\n</codebase_context>\n"
              "<implementation_steps>\n1. Edit `build_prompt` in `retrieval/generation.py`.\n</implementation_steps>")
    assert scope_messages(prompt, repo, "enforce citations") == []


def test_ambiguous_basenames_are_not_reported_missing(repo: Path) -> None:
    assert messages("Update `run_eval.py` and `__init__.py`.", repo) == []


def test_ambiguous_basename_still_gets_the_scope_warning_when_every_match_is_eval_code(repo: Path) -> None:
    plan = "<implementation_steps>\n1. Edit `run_eval.py` and `__init__.py`.\n</implementation_steps>"
    found = scope_messages(plan, repo, "enforce citations")
    assert len(found) == 1 and "eval/run_eval.py" in found[0] and "evaluation/run_eval.py" in found[0]


def test_editing_tests_is_not_a_scope_problem(repo: Path) -> None:
    plan = "<implementation_steps>\n1. Add cases to `tests/test_generation_eval.py`.\n</implementation_steps>"
    assert scope_messages(plan, repo, "enforce citations") == []
