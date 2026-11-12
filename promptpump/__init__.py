"""PromptPump: compile shorthand prompts into dense, structured master prompts."""

from .context import WorkspaceContext, scan_workspace
from .engine import CompileResult, OllamaClient, PromptCompiler, PromptPumpError

__version__ = "0.1.0"

__all__ = [
    "CompileResult",
    "OllamaClient",
    "PromptCompiler",
    "PromptPumpError",
    "WorkspaceContext",
    "scan_workspace",
    "__version__",
]
