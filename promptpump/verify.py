"""Reference checker: flag files and symbols in a compiled prompt that do not match the repo.

Small local models often attribute a function to the wrong file. The scanner knows where
every function and class is defined, so those slips can be caught deterministically.
Findings are warnings to review; they never alter the prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .context import SECONDARY_QUERY_TOKENS, WorkspaceContext, is_secondary_path, is_test_path
from .rank import tokenize

_PATH_RE = re.compile(
    r"[\w.\-]+(?:/[\w.\-]+)*\.(?:py|js|jsx|ts|tsx|go|md|toml|json|yml|yaml|txt|cfg|ini|html|css|sh)\b"
)
_TICK_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)(?:\(\))?`")
_FILE_OPEN_RE = re.compile(r'<file\s+path="([^"]+)"\s*>')
PLAN_TAGS = ("<architectural_specifications>", "<implementation_steps>")
PLAN_CLOSE = ("</architectural_specifications>", "</implementation_steps>")
EVAL_QUERY_TOKENS = SECONDARY_QUERY_TOKENS - {"test"}  # asking for tests is fine; eval edits need asking
MAX_DEFINING_FILES = 3  # names defined in many files (``main``, ``run``) are too ambiguous to check


@dataclass(frozen=True)
class Issue:
    """One suspicious reference in a compiled prompt."""

    kind: str  # "wrong_file", "missing_path" or "scope"
    message: str
    line: int


def _matches(path: str, files: set[str]) -> list[str]:
    """Every repo file a mentioned path could refer to (a bare basename can match several)."""
    return sorted(f for f in files if f == path or f.endswith("/" + path))


def check_references(
    prompt: str, context: WorkspaceContext, shorthand: str | None = None
) -> list[Issue]:
    """Compare files and ``backticked`` symbols named in ``prompt`` against the scanned repo.

    Also warns when the plan edits tests or evaluation code that the request never mentioned.
    """
    scanner = context.scanner
    if scanner is None:
        return []
    index, files = scanner.symbol_index(), scanner.all_paths()
    issues: list[Issue] = []
    seen: set[tuple] = set()
    current: str | None = None  # path of the enclosing <file path="..."> block
    request = shorthand if shorthand is not None else context.query
    check_scope = request is not None and not EVAL_QUERY_TOKENS & set(tokenize(request))
    in_plan = False

    for number, line in enumerate(prompt.splitlines(), 1):
        opened = _FILE_OPEN_RE.search(line)
        if opened:
            current = opened.group(1)

        if any(tag in line for tag in PLAN_TAGS):
            in_plan = True
        expected: list[str] = []
        for path in _PATH_RE.findall(line):
            matches = _matches(path, files)
            if not matches:
                key = ("missing", path)
                if key not in seen:
                    seen.add(key)
                    issues.append(Issue("missing_path", f"{path} does not exist in the repo (fine if the plan creates it)", number))
                continue
            if len(matches) == 1:  # a bare basename shared by several files stays ambiguous
                expected.append(matches[0])
            # Tests are expected to change with a feature; evaluation/benchmark code is not.
            if check_scope and in_plan and all(is_secondary_path(m) and not is_test_path(m) for m in matches):
                key = ("scope", tuple(matches))
                if key not in seen:
                    seen.add(key)
                    issues.append(Issue(
                        "scope",
                        f"the plan edits {', '.join(matches)}, but your request does not mention evaluation or benchmarks",
                        number,
                    ))
        if not expected and current:
            expected = [current]

        if expected:
            for name in _TICK_RE.findall(line):
                defined = index.get(name, [])
                if not defined or len(defined) > MAX_DEFINING_FILES or set(defined) & set(expected):
                    continue
                key = ("wrong", name, tuple(expected))
                if key not in seen:
                    seen.add(key)
                    issues.append(Issue(
                        "wrong_file",
                        f"`{name}` is defined in {', '.join(defined)}, not in {', '.join(expected)}",
                        number,
                    ))
        if "</file>" in line:
            current = None
        if any(tag in line for tag in PLAN_CLOSE):
            in_plan = False
    return issues
