"""Tests for outlines, BM25 ranking, token budgeting, and context statistics."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from promptpump.cli import app
from promptpump.context import scan_workspace
from promptpump.outline import outline_for, outline_go, outline_js
from promptpump.rank import BM25, rank_files, tokenize
from promptpump.stats import compute_stats, gold_file_recall
from promptpump.tokens import get_counter, heuristic_count


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_text("# Demo\n" * 40)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    for i in range(30):
        body = f"def func_{i}(x):\n    '''doc {i}'''\n" + "    y = x + 1\n" * 200
        (pkg / f"mod{i}.py").write_text(body)
    (pkg / "citations.py").write_text(
        "def extract_citations(answer: str) -> set:\n    '''Find citation markers.'''\n    return set()\n"
    )
    return tmp_path


# -- tokenizer & ranking -----------------------------------------------------------------


def test_tokenize_splits_identifiers_and_stems() -> None:
    assert tokenize("extractCitations in query_rag.py") == ["extract", "citation", "query", "rag", "py"]
    assert tokenize("HTTPServer") == ["http", "server"]


def test_bm25_prefers_matching_document() -> None:
    docs = [["citation", "answer", "rag"], ["pause", "menu", "game"], ["login", "auth"]]
    scores = BM25(docs).scores(["citation"])
    assert scores.index(max(scores)) == 0 and scores[1] == 0 == scores[2]


def test_rank_files_path_boost() -> None:
    ranked = rank_files({"api/citations.py": "x = 1", "api/models.py": "x = 1"}, "citations")
    assert ranked[0][0] == "api/citations.py" and ranked[0][1] > ranked[1][1]


def test_rank_files_empty() -> None:
    assert rank_files({}, "anything") == []


# -- outlines ----------------------------------------------------------------------------


def test_outline_js_and_go() -> None:
    js = "export async function load(a, b) {}\nclass Foo {}\nexport const f = (x) => x\ninterface Bar {}\n  function nested() {}\n"
    assert outline_js(js) == ["function load(a, b)", "class Foo", "const f = (x) =>", "interface Bar"]
    go = "func Public(a int) {}\nfunc private() {}\nfunc (s *Srv) Run() {}\ntype Srv struct {\n"
    assert outline_go(go) == ["func Public(a int)", "func (s *Srv) Run()", "type Srv struct"]
    assert outline_for("x.rb", "def a; end") == []


# -- budgeting ---------------------------------------------------------------------------


def test_budget_is_enforced_and_ranking_orders_outlines(repo: Path) -> None:
    ctx = scan_workspace(repo, max_tokens=800, query="fix extract citations")
    assert heuristic_count(ctx.render()) <= 800
    assert list(ctx.outlines)[0] == "pkg/citations.py"
    assert ctx.ranked[0][0] == "pkg/citations.py"


def test_budget_holds_under_a_different_counter(repo: Path) -> None:
    ctx = scan_workspace(repo, max_tokens=2000, counter=len, query="citations")  # 1 char = 1 token
    assert len(ctx.render()) <= 2000


def test_no_query_keeps_directory_order(repo: Path) -> None:
    ctx = scan_workspace(repo, max_tokens=3000)
    assert ctx.ranked == [] and ctx.outlines


# -- stats -------------------------------------------------------------------------------


def test_reduction_math_and_baseline_scope(repo: Path) -> None:
    deep = repo / "a" / "b" / "c" / "d" / "e" / "f"
    deep.mkdir(parents=True)
    (deep / "deep.py").write_text("def deep(): pass\n")  # beyond tree depth, still in baseline
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "x.js").write_text("x" * 5000)  # ignored

    ctx = scan_workspace(repo, max_tokens=800, query="citations")
    stats = compute_stats(ctx)
    assert stats.baseline_files == 33  # README + 30 mods + citations + deep.py
    assert stats.baseline_complete and stats.packed_tokens == heuristic_count(ctx.render())
    expected = (1 - stats.packed_tokens / stats.baseline_tokens) * 100
    assert stats.reduction_pct == pytest.approx(expected) and stats.reduction_pct > 80
    assert stats.tree_tokens + stats.key_file_tokens + stats.outline_tokens <= stats.packed_tokens


def test_baseline_cap_is_reported(repo: Path) -> None:
    stats = compute_stats(scan_workspace(repo, max_tokens=800), max_files=5)
    assert stats.baseline_files == 5 and not stats.baseline_complete


def test_gold_file_recall(repo: Path) -> None:
    ctx = scan_workspace(repo, max_tokens=800, query="citations")
    assert gold_file_recall(ctx, ["pkg/citations.py", "does/not/exist.py"]) == 0.5
    assert gold_file_recall(ctx, []) == 1.0


def test_get_counter() -> None:
    assert get_counter().name.startswith("heuristic")
    with pytest.raises(ValueError):
        get_counter("bogus")


def test_cli_stats_dry_run(repo: Path) -> None:
    result = CliRunner().invoke(app, ["run", "fix citations", "--path", str(repo), "--stats", "--dry-run"])
    assert result.exit_code == 0
    assert "Input reduction" in result.output and "<shorthand_request>" in result.output


def test_defining_a_symbol_outranks_mentioning_it() -> None:
    long_body = "we use the ollama server here. " * 200
    ranked = rank_files(
        {"client.py": "x = 1", "notes.py": long_body},
        "retry logic for the ollama client",
        symbols={"client.py": "class OllamaClient: def chat(self)"},
    )
    assert ranked[0][0] == "client.py"


# -- relevance-aware packing -------------------------------------------------------------

from promptpump.context import is_test_path  # noqa: E402
from promptpump.outline import trim_outline  # noqa: E402
from promptpump.stats import gold_symbol_recall  # noqa: E402


def test_is_test_path() -> None:
    assert is_test_path("tests/test_x.py") and is_test_path("pkg/conftest.py")
    assert is_test_path("web/__tests__/a.js") and is_test_path("svc/handler_test.go")
    assert not is_test_path("pkg/contest.py") and not is_test_path("api/main.py")


def test_trim_outline_keeps_matching_symbols_and_class_header() -> None:
    lines = ["class Svc:", "    def a(self)", "    def retry_policy(self)", "    def b(self)",
             "def helper_one()", "def helper_two()"]
    assert trim_outline(lines, set(tokenize("retry policy")), 2) == ["class Svc:", "    def retry_policy(self)"]
    assert trim_outline(lines, set(), 2) == ["class Svc:", "    def a(self)"]  # no query: first entries
    assert trim_outline(lines, {"x"}, 99) == lines


def test_test_files_rank_below_source_unless_prompt_mentions_tests(tmp_path: Path) -> None:
    body = "def handle_widgets(): pass\n"
    (tmp_path / "tests").mkdir()
    (tmp_path / "zsrc").mkdir()
    (tmp_path / "tests" / "test_widgets.py").write_text(body)
    (tmp_path / "zsrc" / "widgets.py").write_text(body)
    plain = [p for p, _ in scan_workspace(tmp_path, query="handle widgets").ranked]
    assert plain.index("zsrc/widgets.py") < plain.index("tests/test_widgets.py")
    with_tests = [p for p, _ in scan_workspace(tmp_path, query="handle widgets tests").ranked]
    assert with_tests.index("tests/test_widgets.py") < with_tests.index("zsrc/widgets.py")


def test_relevant_symbol_survives_a_tight_budget_deep_in_a_large_file(tmp_path: Path) -> None:
    funcs = [f"def fn_{i}(a, b):\n    return a + b\n" for i in range(80)]
    funcs.insert(70, "def build_prompt(question, sources):\n    return question\n")
    (tmp_path / "gen.py").write_text("\n".join(funcs))
    ctx = scan_workspace(tmp_path, max_tokens=500, query="build prompt with sources")
    assert any("build_prompt" in line for line in ctx.outlines["gen.py"])
    assert gold_symbol_recall(ctx, {"gen.py": ["build_prompt"]}) == 1.0
    assert gold_symbol_recall(ctx, {"gen.py": ["build_prompt", "nope_missing"]}) == 0.5
    assert gold_symbol_recall(ctx, {}) == 1.0


def test_key_files_leave_room_for_outlines(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("readme words " * 3000)
    (tmp_path / "Dockerfile").write_text("FROM python\n" * 500)
    for i in range(5):
        (tmp_path / f"m{i}.py").write_text(f"def feature_{i}(x):\n    return x\n")
    ctx = scan_workspace(tmp_path, max_tokens=1500, query="feature")
    assert len(ctx.outlines) >= 4
    assert heuristic_count(ctx.render()) <= 1500


def test_code_baseline_excludes_data_files(repo: Path) -> None:
    (repo / "data.json").write_text('{"k": "v"}\n' * 5000)
    (repo / "package.json").write_text('{"name": "x"}\n')  # config: stays in the code baseline
    stats = compute_stats(scan_workspace(repo, max_tokens=800, query="citations"))
    assert stats.code_baseline_tokens < stats.baseline_tokens
    assert stats.baseline_tokens - stats.code_baseline_tokens > 10_000
    assert stats.code_reduction_pct < stats.reduction_pct


def test_tests_only_get_outline_budget_after_source_files(tmp_path: Path) -> None:
    body = "def handle_widgets(): pass\n"
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_widgets.py").write_text(body)
    (tmp_path / "zsrc").mkdir()
    (tmp_path / "zsrc" / "widgets.py").write_text(body)
    assert list(scan_workspace(tmp_path, query="handle widgets").outlines) == [
        "zsrc/widgets.py", "tests/test_widgets.py"]
    assert list(scan_workspace(tmp_path, query="handle widgets tests").outlines)[0] == "tests/test_widgets.py"


def test_unrelated_files_are_not_padded_into_the_outline(tmp_path: Path) -> None:
    (tmp_path / "citations.py").write_text("def extract_citations(answer): pass\n")
    for name in ("scraper_a", "scraper_b", "scraper_c"):
        (tmp_path / f"{name}.py").write_text("def download_pages(url): pass\n")
    assert list(scan_workspace(tmp_path, query="source citations").outlines) == ["citations.py"]
    # a query matching nothing falls back to including files rather than an empty outline
    assert scan_workspace(tmp_path, query="zzzunmatched").outlines


def test_eval_and_benchmark_code_is_secondary_unless_mentioned(tmp_path: Path) -> None:
    body = "def extract_citations(answer): pass\n"
    for d in ("evaluation", "retrieval"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "citations.py").write_text(body)
    plain = scan_workspace(tmp_path, query="citations")
    assert list(plain.outlines) == ["retrieval/citations.py", "evaluation/citations.py"]
    assert plain.scores["retrieval/citations.py"] > plain.scores["evaluation/citations.py"]
    asked = scan_workspace(tmp_path, query="citations evaluation")
    assert asked.scores["evaluation/citations.py"] >= asked.scores["retrieval/citations.py"]


def test_docstring_prose_does_not_masquerade_as_a_symbol_match(tmp_path: Path) -> None:
    (tmp_path / "a_scraper.py").write_text(
        'def download_pages(url):\n    """Return citations. Citations are listed."""\n    return url\n')
    (tmp_path / "z_cites.py").write_text("def extract_citations(answer):\n    return answer\n")
    ranked = scan_workspace(tmp_path, query="citations").ranked
    assert ranked[0][0] == "z_cites.py"  # the function *named* for it beats docstring prose
