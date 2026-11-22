"""Local workspace scanner: file tree, key files, and relevance-ranked symbol outlines.

The scanner is deliberately conservative: it never follows symlinks, skips heavy
directories and secret files, honours the root ``.gitignore``, and packs everything into
a token budget (measured with a pluggable counter) so it fits a small model's context.
When a ``query`` is given, outlines are packed in BM25 relevance order instead of
directory order, so the budget goes to the files that matter for the request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import pathspec

from .outline import (  # noqa: F401  (re-exported for backwards compatibility)
    MAX_OUTLINE_LINES_PER_FILE,
    OUTLINE_SUFFIXES,
    outline_for,
    outline_python,
    trim_outline,
)
from .rank import rank_files, tokenize
from .tokens import heuristic_count

# Heavy / generated directories that are never worth scanning.
DEFAULT_IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git", ".hg", ".svn", "node_modules", "venv", ".venv", "env", "__pycache__",
        ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".idea", ".vscode",
        "dist", "build", ".next", ".nuxt", "target", "coverage", ".eggs", ".cache",
        "site-packages",
    }
)

# Non-text files that would only waste tokens.
BINARY_SUFFIXES: frozenset[str] = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg", ".pdf", ".zip", ".tar",
        ".gz", ".7z", ".rar", ".exe", ".dll", ".so", ".dylib", ".pyc", ".pyo", ".class",
        ".jar", ".bin", ".woff", ".woff2", ".ttf", ".mp3", ".mp4", ".mov", ".sqlite",
        ".db", ".lock", ".pt", ".pth", ".onnx", ".parquet", ".npy", ".npz",
    }
)

SECRET_SUFFIXES: frozenset[str] = frozenset({".pem", ".key", ".p12", ".pfx"})
ENV_TEMPLATE_NAMES: frozenset[str] = frozenset({".env.example", ".env.sample", ".env.template"})

# Priority-ordered file names treated as "key files" (first match wins the budget).
KEY_FILE_NAMES: tuple[str, ...] = (
    "README.md", "README.rst", "README.txt", "README",
    "pyproject.toml", "package.json", "requirements.txt", "setup.py", "setup.cfg", "Pipfile",
    "tsconfig.json", "vite.config.ts", "vite.config.js", "next.config.js",
    "Dockerfile", "docker-compose.yml", "Makefile", ".env.example",
    "main.py", "app.py", "manage.py", "__main__.py", "cli.py",
    "index.js", "index.ts", "server.js", "main.tsx", "App.tsx", "App.jsx",
)

MAX_FILE_BYTES = 1_000_000
MAX_OUTLINE_FILES = 25
KEY_FILE_BUDGET_FRACTION = 0.3  # README/config get at most this share of what is left after the tree
SCAN_OUTLINE_LINES = 300  # read generously, then trim by relevance
MIN_RELATIVE_SCORE = 0.25  # outlines only for files scoring at least this share of the best match
TEST_SCORE_FACTOR = 0.5  # tests and eval/benchmark code rank lower unless the prompt mentions them
SECONDARY_DIRS = frozenset({"eval", "evals", "evaluation", "benchmark", "benchmarks", "bench"})
SECONDARY_QUERY_TOKENS = frozenset({"test", "eval", "evaluation", "benchmark", "bench"})
WRAPPER_OVERHEAD_TOKENS = 12  # <file path="..."> ... </file> tags around key-file content

Counter_ = Callable[[str], int]

_DEF_RE = re.compile(r"^\s*(?:async\s+def|def|class|function|interface|type|enum|func)\s+(?:\([^)]*\)\s*)?(\w+)")
_CONST_RE = re.compile(r"^const\s+(\w+)")


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 characters per token); kept for backwards compatibility."""
    return heuristic_count(text)


@dataclass(frozen=True)
class KeyFile:
    """A key project file and its (possibly truncated) content."""

    path: str
    content: str
    truncated: bool = False


@dataclass
class WorkspaceContext:
    """Snapshot of the local workspace handed to the compiler."""

    root: Path
    tree_lines: list[str] = field(default_factory=list)
    key_files: list[KeyFile] = field(default_factory=list)
    outlines: dict[str, list[str]] = field(default_factory=dict)
    ranked: list[tuple[str, float]] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    query: str | None = None
    scanner: Any = field(default=None, repr=False, compare=False)

    @property
    def tree(self) -> str:
        """The file tree as printable text, rooted at the project directory name."""
        return "\n".join([f"{self.root.name}/", *self.tree_lines])

    def sections(self) -> dict[str, str]:
        """Rendered context blocks by name (used for rendering and token accounting)."""
        parts = {"file_tree": f"<file_tree>\n{self.tree}\n</file_tree>"}
        if self.key_files:
            files = "\n".join(
                f'<file path="{kf.path}"{" truncated" if kf.truncated else ""}>\n'
                f"{kf.content}\n</file>"
                for kf in self.key_files
            )
            parts["key_files"] = f"<key_files>\n{files}\n</key_files>"
        if self.outlines:
            mods = "\n".join(
                f'<module path="{path}">\n' + "\n".join(lines) + "\n</module>"
                for path, lines in self.outlines.items()
            )
            parts["code_outline"] = f"<code_outline>\n{mods}\n</code_outline>"
        return parts

    def render(self) -> str:
        """Full context block sent to the model."""
        return "\n\n".join(self.sections().values())

    def included_paths(self) -> set[str]:
        """Files whose content or symbol outline is present in the context."""
        return {kf.path for kf in self.key_files} | set(self.outlines)

    def render_summary(self, max_tree_lines: int = 60) -> str:
        """Compact, deterministic summary used to ground ``<codebase_context>``."""
        tree = self.tree.splitlines()
        if len(tree) > max_tree_lines:
            hidden = len(tree) - max_tree_lines
            tree = [*tree[:max_tree_lines], f"... ({hidden} more entries)"]
        lines = ["File tree:", "```text", *tree, "```"]
        if self.key_files:
            lines += ["", "Key files: " + ", ".join(kf.path for kf in self.key_files)]
        if self.outlines:
            lines += ["", "Code outline (public API):"]
            for path, outline in self.outlines.items():
                lines.append(f"- {path}")
                lines.extend(f"  {entry}" for entry in outline[:12])
        return "\n".join(lines)


def load_gitignore(root: Path) -> pathspec.GitIgnoreSpec | None:
    """Parse the root ``.gitignore`` if present (nested ignore files are not read)."""
    gitignore = root / ".gitignore"
    if not gitignore.is_file():
        return None
    try:
        lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    return pathspec.GitIgnoreSpec.from_lines(lines)


def is_test_path(path: str) -> bool:
    """True for test files and test directories (pytest, jest, go-style names)."""
    parts = path.lower().split("/")
    name = parts[-1]
    return (
        any(p in {"tests", "test", "__tests__", "spec"} for p in parts[:-1])
        or name.startswith("test_")
        or name == "conftest.py"
        or name.endswith(("_test.py", "_test.go", ".test.ts", ".test.js", ".spec.ts", ".spec.js"))
    )


def is_secondary_path(path: str) -> bool:
    """Tests plus evaluation/benchmark code: rarely where a feature request lands."""
    parts = path.lower().split("/")
    return is_test_path(path) or any(p in SECONDARY_DIRS for p in parts[:-1])


def _is_secret(name: str) -> bool:
    """Heuristic for files that may contain credentials; never read or list them."""
    if name == ".env" or (name.startswith(".env.") and name not in ENV_TEMPLATE_NAMES):
        return True
    return (
        Path(name).suffix.lower() in SECRET_SUFFIXES
        or name.startswith("id_rsa")
        or name == "credentials.json"
    )


def _read_text(path: Path, max_chars: int) -> tuple[str, bool] | None:
    """Read up to ``max_chars`` of a text file; return ``None`` for binary/unreadable."""
    try:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            return None
        with path.open("rb") as fh:
            data = fh.read(max_chars * 4 + 1)
    except OSError:
        return None
    if b"\x00" in data[:1024]:
        return None
    text = data.decode("utf-8", errors="replace")
    truncated = len(text) > max_chars or size > len(data)
    return text[:max_chars], truncated


class WorkspaceScanner:
    """Crawls a directory and builds a budgeted :class:`WorkspaceContext`."""

    def __init__(
        self,
        root: Path,
        *,
        max_tokens: int = 3000,
        max_depth: int = 4,
        max_entries: int = 400,
        extra_ignore: Iterable[str] = (),
        counter: Counter_ = heuristic_count,
        query: str | None = None,
        max_rank_files: int = 3000,
    ) -> None:
        self.root = root.resolve()
        self.max_tokens = max_tokens
        self.max_depth = max_depth
        self.max_entries = max_entries
        self.ignore_dirs = DEFAULT_IGNORE_DIRS | frozenset(extra_ignore)
        self.counter = counter
        self.query = query
        self.max_rank_files = max_rank_files
        self.gitignore = load_gitignore(self.root)
        self._lines: list[str] = []
        self._files: list[Path] = []
        self._count = 0
        self._symbols: dict[str, list[str]] | None = None
        self._paths: set[str] | None = None

    # -- filtering ---------------------------------------------------------------------

    def _skip(self, entry: Path, is_dir: bool) -> bool:
        if entry.is_symlink():  # avoids loops and escaping the root
            return True
        name = entry.name
        if is_dir:
            if name in self.ignore_dirs or name.endswith(".egg-info"):
                return True
        elif _is_secret(name) or entry.suffix.lower() in BINARY_SUFFIXES:
            return True
        if self.gitignore is not None:
            rel = entry.relative_to(self.root).as_posix()
            return self.gitignore.match_file(rel + "/" if is_dir else rel)
        return False

    def iter_files(self) -> Iterator[Path]:
        """Yield every non-ignored file (relative paths), with no depth or count limit."""
        stack = [self.root]
        while stack:
            directory = stack.pop()
            try:
                entries = sorted(directory.iterdir(), key=lambda p: p.name.lower())
            except OSError:
                continue
            subdirs: list[Path] = []
            for entry in entries:
                try:
                    is_dir = entry.is_dir()
                except OSError:
                    continue
                if self._skip(entry, is_dir):
                    continue
                if is_dir:
                    subdirs.append(entry)
                else:
                    yield entry.relative_to(self.root)
            stack.extend(reversed(subdirs))

    def all_paths(self) -> set[str]:
        """Every non-ignored file path (posix, relative to the root); computed once."""
        if self._paths is None:
            self._paths = {p.as_posix() for p in self.iter_files()}
        return self._paths

    def symbol_index(self) -> dict[str, list[str]]:
        """Map each class/function name to the files that define it (computed once)."""
        if self._symbols is None:
            index: dict[str, set[str]] = {}
            for i, rel in enumerate(self.iter_files()):
                if i >= self.max_rank_files:
                    break
                if rel.suffix.lower() not in OUTLINE_SUFFIXES:
                    continue
                result = _read_text(self.root / rel, 200_000)
                if result is None:
                    continue
                for line in outline_for(rel.name, result[0], 10_000):
                    match = _DEF_RE.match(line) or _CONST_RE.match(line)
                    if match:
                        index.setdefault(match.group(1), set()).add(rel.as_posix())
            self._symbols = {name: sorted(files) for name, files in index.items()}
        return self._symbols

    # -- tree --------------------------------------------------------------------------

    def _walk(self, directory: Path, depth: int, prefix: str) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        visible = [(e, e.is_dir()) for e in entries]
        visible = [(e, d) for e, d in visible if not self._skip(e, d)]
        for i, (entry, is_dir) in enumerate(visible):
            if self._count >= self.max_entries:
                return
            last = i == len(visible) - 1
            self._lines.append(f"{prefix}{'└── ' if last else '├── '}{entry.name}{'/' if is_dir else ''}")
            self._count += 1
            if is_dir:
                if depth < self.max_depth:
                    self._walk(entry, depth + 1, prefix + ("    " if last else "│   "))
            else:
                self._files.append(entry.relative_to(self.root))

    def _fit_lines(self, lines: list[str], token_budget: int) -> list[str]:
        kept: list[str] = []
        used = 0
        for i, line in enumerate(lines):
            cost = self.counter(line) + 1
            if used + cost > token_budget:
                kept.append(f"... ({len(lines) - i} more entries truncated)")
                break
            kept.append(line)
            used += cost
        return kept

    # -- key files ---------------------------------------------------------------------

    def _select_key_files(self, token_budget: int) -> list[KeyFile]:
        ranked: list[tuple[int, int, str, Path]] = []
        for rel in self._files:
            if len(rel.parts) <= 3 and rel.name in KEY_FILE_NAMES:
                ranked.append((KEY_FILE_NAMES.index(rel.name), len(rel.parts), rel.as_posix(), rel))
        ranked.sort()

        selected: list[KeyFile] = []
        remaining = token_budget
        per_file_cap = max(200, token_budget // 2)
        for *_, rel in ranked:
            file_budget = min(remaining, per_file_cap) - WRAPPER_OVERHEAD_TOKENS
            if file_budget < 50:
                break
            result = _read_text(self.root / rel, file_budget * 4)
            if result is None:
                continue
            text, truncated = result
            while text and self.counter(text) > file_budget:
                text, truncated = text[: int(len(text) * 0.85)], True
            text = text.rstrip()
            if not text:
                continue
            selected.append(KeyFile(rel.as_posix(), text, truncated))
            remaining -= self.counter(text) + WRAPPER_OVERHEAD_TOKENS
        return selected

    # -- outlines ----------------------------------------------------------------------

    @staticmethod
    def _module_block(path: str, lines: list[str]) -> str:
        return f'<module path="{path}">\n' + "\n".join(lines) + "\n</module>"

    def _default_outline_candidates(self) -> Iterator[tuple[str, list[str]]]:
        """Directory-order candidates (shallow first) used when no query is given."""
        for rel in sorted((p for p in self._files if p.suffix.lower() in OUTLINE_SUFFIXES),
                          key=lambda p: (len(p.parts), p.as_posix())):
            result = _read_text(self.root / rel, 200_000)
            if result is None:
                continue
            outline = outline_for(rel.name, result[0], SCAN_OUTLINE_LINES)
            if outline:
                yield rel.as_posix(), outline

    def _rank(self) -> tuple[list[tuple[str, float]], dict[str, list[str]]]:
        """BM25-rank every readable file against the query; also cache their outlines."""
        docs: dict[str, str] = {}
        symbols: dict[str, str] = {}
        outlines: dict[str, list[str]] = {}
        for i, rel in enumerate(self.iter_files()):
            if i >= self.max_rank_files:
                break
            result = _read_text(self.root / rel, 200_000)
            if result is None:
                continue
            text, _ = result
            key = rel.as_posix()
            outline = outline_for(rel.name, text, SCAN_OUTLINE_LINES)
            if outline:
                outlines[key] = outline
                # symbol *names* carry intent; docstring prose ("  # ...") only counts as body text
                symbols[key] = " ".join(line.split("  # ", 1)[0] for line in outline)
            docs[key] = text[:20_000]
        ranked = rank_files(docs, self.query or "", symbols)
        if not SECONDARY_QUERY_TOKENS & set(tokenize(self.query or "")):
            ranked = [(p, sc * TEST_SCORE_FACTOR if is_secondary_path(p) else sc) for p, sc in ranked]
            ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked, outlines

    def _pack_outlines(
        self, candidates: Iterable[tuple[str, list[str]]], token_budget: int
    ) -> dict[str, list[str]]:
        query_tokens = set(tokenize(self.query)) if self.query else set()
        packed: dict[str, list[str]] = {}
        remaining = token_budget
        for path, outline in candidates:
            if len(packed) >= MAX_OUTLINE_FILES or remaining < 40:
                break
            keep = min(len(outline), MAX_OUTLINE_LINES_PER_FILE)
            lines = trim_outline(outline, query_tokens, keep)
            cost = self.counter(self._module_block(path, lines))
            while keep > 1 and cost > remaining:  # shrink, keeping the best-matching symbols
                keep = max(1, keep // 2)
                lines = trim_outline(outline, query_tokens, keep)
                cost = self.counter(self._module_block(path, lines))
            if cost > remaining:
                continue
            packed[path] = lines
            remaining -= cost
        return packed

    def _enforce_budget(self, ctx: WorkspaceContext) -> None:
        """Final guarantee: drop lowest-priority material until the render fits."""
        while self.counter(ctx.render()) > self.max_tokens and ctx.outlines:
            ctx.outlines.popitem()
        while self.counter(ctx.render()) > self.max_tokens and ctx.key_files:
            ctx.key_files.pop()

    # -- public ------------------------------------------------------------------------

    def scan(self) -> WorkspaceContext:
        """Run the scan and return a context that fits ``max_tokens``."""
        self._walk(self.root, 1, "")
        budget = self.max_tokens
        ctx = WorkspaceContext(
            self.root,
            self._fit_lines(self._lines, int(budget * 0.3)),
            query=self.query,
            scanner=self,
        )
        remaining = budget - self.counter(ctx.sections()["file_tree"])
        ctx.key_files = self._select_key_files(int(max(remaining, 0) * KEY_FILE_BUDGET_FRACTION))
        remaining -= sum(self.counter(kf.content) + WRAPPER_OVERHEAD_TOKENS for kf in ctx.key_files)

        if self.query and self.query.strip():
            ranked, cache = self._rank()
            ctx.ranked = ranked[:20]
            ctx.scores = dict(ranked)
            scores = dict(ranked)
            ordered = [path for path, _ in ranked if path in cache]
            best = max((scores[p] for p in ordered), default=0.0)
            if best > 0:  # skip files unrelated to the request instead of padding the budget
                ordered = [p for p in ordered if scores[p] >= MIN_RELATIVE_SCORE * best]
            if not SECONDARY_QUERY_TOKENS & set(tokenize(self.query)):
                ordered.sort(key=is_secondary_path)  # stable: runtime code first, tests/evals last
            candidates: Iterable[tuple[str, list[str]]] = ((path, cache[path]) for path in ordered)
        else:
            candidates = self._default_outline_candidates()
        ctx.outlines = self._pack_outlines(candidates, max(remaining, 0))
        self._enforce_budget(ctx)
        return ctx


def scan_workspace(root: Path | str | None = None, **kwargs: Any) -> WorkspaceContext:
    """Convenience wrapper: scan ``root`` (default: current directory)."""
    return WorkspaceScanner(Path(root or Path.cwd()), **kwargs).scan()
