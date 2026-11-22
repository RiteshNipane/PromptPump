"""Context statistics: exact token accounting and reduction versus pasting the whole repo."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from .context import (
    KEY_FILE_NAMES,
    MAX_FILE_BYTES,
    WorkspaceContext,
    WorkspaceScanner,
    _read_text,
)
from .tokens import heuristic_count

MAX_BASELINE_FILES = 20_000
# Data, logs and reports: pasting them is unrealistic, so a second baseline excludes them.
DATA_SUFFIXES: frozenset[str] = frozenset({".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".log", ".xml"})


@dataclass(frozen=True)
class ContextStats:
    """Token accounting for one scan. All counts use the same ``counter_name``."""

    counter_name: str
    budget_tokens: int
    baseline_tokens: int
    code_baseline_tokens: int
    baseline_files: int
    baseline_complete: bool
    packed_tokens: int
    tree_tokens: int
    key_file_tokens: int
    outline_tokens: int
    files_in_context: int

    @property
    def reduction_pct(self) -> float:
        """Percent of input tokens saved versus the whole-repo baseline."""
        if self.baseline_tokens <= 0:
            return 0.0
        return max(0.0, (1 - self.packed_tokens / self.baseline_tokens) * 100)

    @property
    def code_reduction_pct(self) -> float:
        """Percent saved versus a baseline that leaves out data/log/report files."""
        if self.code_baseline_tokens <= 0:
            return 0.0
        return max(0.0, (1 - self.packed_tokens / self.code_baseline_tokens) * 100)


def compute_stats(
    context: WorkspaceContext,
    counter: Callable[[str], int] = heuristic_count,
    counter_name: str = "heuristic (~4 chars/token)",
    max_files: int = MAX_BASELINE_FILES,
) -> ContextStats:
    """Measure the context against a baseline of *every* non-ignored text file.

    The baseline is what you would send by pasting the whole repository; the packed figure
    is the rendered context. Both use the same counter, so the percentage is consistent.
    """
    scanner = context.scanner or WorkspaceScanner(context.root)
    total = code_total = files = 0
    complete = True
    for rel in scanner.iter_files():
        if files >= max_files:
            complete = False
            break
        result = _read_text(context.root / rel, MAX_FILE_BYTES)
        if result is None:
            continue
        tokens = counter(result[0])
        total += tokens
        if rel.suffix.lower() not in DATA_SUFFIXES or rel.name in KEY_FILE_NAMES:
            code_total += tokens
        files += 1

    sections = context.sections()
    return ContextStats(
        counter_name=counter_name,
        budget_tokens=scanner.max_tokens,
        baseline_tokens=total,
        code_baseline_tokens=code_total,
        baseline_files=files,
        baseline_complete=complete,
        packed_tokens=counter(context.render()),
        tree_tokens=counter(sections.get("file_tree", "")),
        key_file_tokens=counter(sections.get("key_files", "")),
        outline_tokens=counter(sections.get("code_outline", "")),
        files_in_context=len(context.included_paths()),
    )


def gold_file_recall(context: WorkspaceContext, gold_files: Iterable[str]) -> float:
    """Fraction of ``gold_files`` whose content or outline made it into the context.

    A big token reduction is only a win if this stays high: dropping the file the change
    actually touches is a regression, not an optimisation.
    """
    gold = {g.replace("\\", "/") for g in gold_files}
    if not gold:
        return 1.0
    return len(gold & context.included_paths()) / len(gold)


def gold_symbol_recall(context: WorkspaceContext, gold: Mapping[str, Iterable[str]]) -> float:
    """Fraction of ``{file: [symbol, ...]}`` whose symbol name appears in that file's context.

    Stricter than :func:`gold_file_recall`: a file can be "included" yet have lost the one
    function that matters, which this catches.
    """
    total = hits = 0
    for path, names in gold.items():
        path = path.replace("\\", "/")
        text = "\n".join(context.outlines.get(path, []))
        text += "\n" + "\n".join(kf.content for kf in context.key_files if kf.path == path)
        for name in names:
            total += 1
            hits += bool(re.search(rf"\b{re.escape(name)}\b", text))
    return hits / total if total else 1.0


def format_stats(stats: ContextStats) -> list[tuple[str, str]]:
    """Label/value rows for display."""
    note = "" if stats.baseline_complete else f" (stopped at {MAX_BASELINE_FILES:,} files)"
    return [
        ("Token counter", stats.counter_name),
        ("Whole-repo baseline", f"{stats.baseline_tokens:,} tokens in {stats.baseline_files:,} files{note}"),
        ("Baseline without data/log files", f"{stats.code_baseline_tokens:,} tokens"),
        ("Packed context", f"{stats.packed_tokens:,} tokens (budget {stats.budget_tokens:,})"),
        (
            "  breakdown",
            f"tree {stats.tree_tokens:,} / key files {stats.key_file_tokens:,} / "
            f"outlines {stats.outline_tokens:,}",
        ),
        ("Input reduction vs whole repo", f"{stats.reduction_pct:.1f}%"),
        ("Input reduction vs code and docs", f"{stats.code_reduction_pct:.1f}%"),
        ("Files with content or outline", f"{stats.files_in_context} of {stats.baseline_files:,}"),
    ]
