"""Symbol outlines: Python via ``ast``; JS/TS and Go via lightweight top-level regexes."""

from __future__ import annotations

import ast
import re
from pathlib import PurePosixPath
from typing import Collection

from .rank import tokenize

MAX_OUTLINE_LINES_PER_FILE = 40
OUTLINE_SUFFIXES: frozenset[str] = frozenset(
    {".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".go"}
)


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


_JS_FUNC = re.compile(r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(\w+)\s*\(([^)]*)\)")
_JS_CLASS = re.compile(r"^(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)")
_JS_ARROW = re.compile(
    r"^(?:export\s+)?(?:const|let)\s+(\w+)\s*=\s*(?:async\s*)?(\([^)]*\)|\w+)"
    r"\s*(?::\s*[\w<>\[\],.| ]+)?\s*=>"
)
_JS_TYPE = re.compile(r"^(?:export\s+)?(?:declare\s+)?(interface|type|enum)\s+(\w+)")


def outline_js(source: str, max_lines: int = MAX_OUTLINE_LINES_PER_FILE) -> list[str]:
    """Top-level functions, classes, arrow-function consts, and types in JS/TS (regex-based)."""
    out: list[str] = []
    for line in source.splitlines():
        if not line or line[0].isspace():  # top-level declarations only
            continue
        if m := _JS_FUNC.match(line):
            out.append(f"function {m.group(1)}({m.group(2).strip()})")
        elif m := _JS_CLASS.match(line):
            out.append(f"class {m.group(1)}")
        elif m := _JS_ARROW.match(line):
            out.append(f"const {m.group(1)} = {m.group(2)} =>")
        elif m := _JS_TYPE.match(line):
            out.append(f"{m.group(1)} {m.group(2)}")
        if len(out) >= max_lines:
            break
    return out


_GO_FUNC = re.compile(r"^func\s+(?:\(([^)]*)\)\s*)?(\w+)\s*\(([^)]*)\)")
_GO_TYPE = re.compile(r"^type\s+(\w+)\s+(struct|interface)\b")


def outline_go(source: str, max_lines: int = MAX_OUTLINE_LINES_PER_FILE) -> list[str]:
    """Exported Go funcs/methods and struct/interface types (regex-based)."""
    out: list[str] = []
    for line in source.splitlines():
        if m := _GO_FUNC.match(line):
            recv, name, args = m.groups()
            if name[0].isupper() or name == "main":
                prefix = f"func ({recv.strip()}) " if recv else "func "
                out.append(f"{prefix}{name}({args.strip()})")
        elif m := _GO_TYPE.match(line):
            out.append(f"type {m.group(1)} {m.group(2)}")
        if len(out) >= max_lines:
            break
    return out


def outline_for(filename: str, source: str, max_lines: int = MAX_OUTLINE_LINES_PER_FILE) -> list[str]:
    """Dispatch to the right outliner by file extension; empty list if unsupported."""
    suffix = PurePosixPath(filename).suffix.lower()
    if suffix == ".py":
        return outline_python(source, max_lines)
    if suffix in {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}:
        return outline_js(source, max_lines)
    if suffix == ".go":
        return outline_go(source, max_lines)
    return []


def trim_outline(lines: list[str], query_tokens: Collection[str], max_lines: int) -> list[str]:
    """Keep the ``max_lines`` outline entries that best match the query, in file order.

    A kept method always brings its class header along, so the outline stays readable.
    With no query tokens this degrades to "the first ``max_lines`` entries".
    """
    if len(lines) <= max_lines:
        return list(lines)
    wanted = set(query_tokens)
    header_of: list[int] = []
    current = 0
    scored: list[tuple[int, int]] = []
    for i, line in enumerate(lines):
        if not line.startswith(" "):
            current = i
        header_of.append(current)
        scored.append((-len(wanted & set(tokenize(line))), i))
    keep: set[int] = set()
    for _, i in sorted(scored):
        if len(keep) >= max_lines:
            break
        keep.add(i)
        keep.add(header_of[i])
    return [lines[i] for i in sorted(keep)]
