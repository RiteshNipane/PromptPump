"""Local workspace scanner: file tree, key files, and Python AST outlines.

The scanner is deliberately conservative: it never follows symlinks, skips heavy
directories and secret files, honours the root ``.gitignore``, and keeps the total
output under a token budget so it fits a small local model's context window.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pathspec

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
MAX_OUTLINE_FILES = 15
MAX_OUTLINE_LINES_PER_FILE = 40


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 characters per token); good enough for budgeting."""
    return max(1, len(text) // 4) if text else 0


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

    @property
    def tree(self) -> str:
        """The file tree as printable text, rooted at the project directory name."""
        return "\n".join([f"{self.root.name}/", *self.tree_lines])

    def render(self) -> str:
        """Full context block sent to the model."""
        parts = [f"<file_tree>\n{self.tree}\n</file_tree>"]
        if self.key_files:
            files = "\n".join(
                f'<file path="{kf.path}"{" truncated" if kf.truncated else ""}>\n'
                f"{kf.content}\n</file>"
                for kf in self.key_files
            )
            parts.append(f"<key_files>\n{files}\n</key_files>")
        if self.outlines:
            mods = "\n".join(
                f'<module path="{path}">\n' + "\n".join(lines) + "\n</module>"
                for path, lines in self.outlines.items()
            )
            parts.append(f"<code_outline>\n{mods}\n</code_outline>")
        return "\n\n".join(parts)

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
            lines += ["", "Python modules (public API):"]
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


def outline_python(source: str, max_lines: int = MAX_OUTLINE_LINES_PER_FILE) -> list[str]:
    """Extract classes, functions, and public methods from Python source via ``ast``."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []

    def sig(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        return f"{prefix} {node.name}({ast.unparse(node.args)}){ret}"

    def doc(node: ast.AST) -> str:
        text = ast.get_docstring(node)  # type: ignore[arg-type]
        return f"  # {text.splitlines()[0][:80]}" if text else ""

    out: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = f"({', '.join(ast.unparse(b) for b in node.bases)})" if node.bases else ""
            out.append(f"class {node.name}{bases}:{doc(node)}")
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not sub.name.startswith("_") or sub.name == "__init__":
                        out.append(f"    {sig(sub)}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                out.append(f"{sig(node)}{doc(node)}")
    return out[:max_lines]


def _fit_lines(lines: list[str], token_budget: int) -> list[str]:
    """Trim ``lines`` so their estimated token total fits the budget."""
    kept: list[str] = []
    used = 0
    for i, line in enumerate(lines):
        cost = estimate_tokens(line) + 1
        if used + cost > token_budget:
            kept.append(f"... ({len(lines) - i} more entries truncated)")
            break
        kept.append(line)
        used += cost
    return kept


class WorkspaceScanner:
    """Crawls a directory and builds a :class:`WorkspaceContext`."""

    def __init__(
        self,
        root: Path,
        *,
        max_tokens: int = 3000,
        max_depth: int = 4,
        max_entries: int = 400,
        extra_ignore: Iterable[str] = (),
    ) -> None:
        self.root = root.resolve()
        self.max_tokens = max_tokens
        self.max_depth = max_depth
        self.max_entries = max_entries
        self.ignore_dirs = DEFAULT_IGNORE_DIRS | frozenset(extra_ignore)
        self.gitignore = load_gitignore(self.root)
        self._lines: list[str] = []
        self._files: list[Path] = []
        self._count = 0

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

    # -- key files & outlines ----------------------------------------------------------

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
            if remaining < 100:
                break
            budget = min(remaining, per_file_cap)
            result = _read_text(self.root / rel, budget * 4)
            if result is None:
                continue
            text, truncated = result
            if not text.strip():
                continue
            selected.append(KeyFile(rel.as_posix(), text.rstrip(), truncated))
            remaining -= estimate_tokens(text)
        return selected

    def _build_outlines(self, token_budget: int) -> dict[str, list[str]]:
        candidates = sorted(
            (p for p in self._files if p.suffix == ".py"), key=lambda p: (len(p.parts), p.as_posix())
        )
        outlines: dict[str, list[str]] = {}
        remaining = token_budget
        for rel in candidates:
            if len(outlines) >= MAX_OUTLINE_FILES or remaining < 50:
                break
            result = _read_text(self.root / rel, 200_000)
            if result is None:
                continue
            outline = outline_python(result[0])
            if not outline:
                continue
            outlines[rel.as_posix()] = outline
            remaining -= sum(estimate_tokens(line) + 1 for line in outline)
        return outlines

    # -- public ------------------------------------------------------------------------

    def scan(self) -> WorkspaceContext:
        """Run the scan and return a budgeted :class:`WorkspaceContext`."""
        self._walk(self.root, 1, "")
        tree_lines = _fit_lines(self._lines, int(self.max_tokens * 0.3))
        key_files = self._select_key_files(int(self.max_tokens * 0.5))
        used = sum(estimate_tokens(kf.content) for kf in key_files)
        outline_budget = max(0, self.max_tokens - int(self.max_tokens * 0.3) - used)
        outlines = self._build_outlines(outline_budget)
        return WorkspaceContext(self.root, tree_lines, key_files, outlines)


def scan_workspace(root: Path | str | None = None, **kwargs: object) -> WorkspaceContext:
    """Convenience wrapper: scan ``root`` (default: current directory)."""
    return WorkspaceScanner(Path(root or Path.cwd()), **kwargs).scan()  # type: ignore[arg-type]
