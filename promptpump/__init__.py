"""PromptPump: compile shorthand prompts into dense, structured master prompts."""

from .context import WorkspaceContext, scan_workspace
from .engine import CompileResult, OllamaClient, PromptCompiler, PromptPumpError
from .stats import ContextStats, compute_stats, gold_file_recall, gold_symbol_recall
from .telemetry import InferenceMetrics

__version__ = "0.2.0.dev0"

__all__ = [
    "CompileResult",
    "ContextStats",
    "InferenceMetrics",
    "compute_stats",
    "gold_file_recall",
    "gold_symbol_recall",
    "OllamaClient",
    "PromptCompiler",
    "PromptPumpError",
    "WorkspaceContext",
    "scan_workspace",
    "__version__",
]
