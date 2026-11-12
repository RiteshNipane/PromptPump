"""Typer CLI entry point: ``promptpump run "shorthand" --stack python --copy``."""

import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__
from .context import scan_workspace
from .engine import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    OllamaClient,
    PromptCompiler,
    PromptPumpError,
)
from .templates import UnknownStackError, get_stack, list_stacks

app = typer.Typer(
    name="promptpump",
    help="Compile a short shorthand prompt into a dense, production-ready master prompt "
    "using local codebase context and a local Ollama model.",
    no_args_is_help=True,
    add_completion=False,
)

# Progress and diagnostics go to stderr so stdout stays clean for piping the prompt.
err = Console(stderr=True)


def copy_to_clipboard(text: str) -> bool:
    """Copy ``text`` to the system clipboard; return False if unavailable."""
    try:
        import pyperclip

        pyperclip.copy(text)
        return True
    except Exception:  # pyperclip raises PyperclipException when no backend exists
        return False


def _force_utf8() -> None:
    """Avoid UnicodeEncodeError on Windows when output is redirected (cp1252 default)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"promptpump {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = None,
) -> None:
    """PromptPump: deterministic context engineering for coding prompts."""
    _force_utf8()


@app.command()
def run(
    prompt: Annotated[str, typer.Argument(help="Shorthand prompt, e.g. 'add jwt auth'.")],
    stack: Annotated[
        Optional[list[str]],
        typer.Option("--stack", "-s", help="Stack rules to inject (repeatable). See `stacks`."),
    ] = None,
    copy: Annotated[
        bool, typer.Option("--copy/--no-copy", help="Copy the result to the clipboard.")
    ] = True,
    path: Annotated[
        Path,
        typer.Option("--path", "-p", exists=True, file_okay=False, resolve_path=True,
                     help="Project directory to scan."),
    ] = Path("."),
    model: Annotated[
        str, typer.Option("--model", "-m", envvar="PROMPTPUMP_MODEL", help="Ollama model name.")
    ] = DEFAULT_MODEL,
    host: Annotated[
        str, typer.Option("--host", envvar="PROMPTPUMP_HOST", help="Ollama base URL.")
    ] = DEFAULT_HOST,
    max_context_tokens: Annotated[
        int, typer.Option("--max-context-tokens", min=500, help="Workspace context budget.")
    ] = 3000,
    ground: Annotated[
        bool,
        typer.Option("--ground/--no-ground",
                     help="Overwrite <codebase_context> with deterministic scan facts."),
    ] = True,
    timeout: Annotated[float, typer.Option(help="Seconds to wait for the model.")] = 300.0,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the payload; do not call the model.")
    ] = False,
) -> None:
    """Compile PROMPT into a structured master prompt."""
    try:
        profiles = [get_stack(s) for s in stack or []]
    except UnknownStackError as exc:
        err.print(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc

    with err.status("Scanning workspace..."):
        context = scan_workspace(path, max_tokens=max_context_tokens)
    err.print(
        f"[dim]Scanned {context.root.name}/: {len(context.tree_lines)} tree entries, "
        f"{len(context.key_files)} key files, {len(context.outlines)} outlined modules[/dim]"
    )

    compiler = PromptCompiler(
        OllamaClient(host=host, model=model, timeout=timeout), ground_context=ground
    )
    stack_names = [p.name for p in profiles]

    if dry_run:
        for message in compiler.build_messages(prompt, context, stack_names):
            typer.echo(f"===== {message['role'].upper()} =====\n{message['content']}\n")
        raise typer.Exit()

    start = time.perf_counter()
    try:
        with err.status(f"Compiling with {escape(model)} (local)..."):
            result = compiler.compile(prompt, context, stack_names)
    except PromptPumpError as exc:
        err.print(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    elapsed = time.perf_counter() - start

    if result.missing_tags:
        err.print(
            f"[yellow]warning:[/yellow] model output is missing tags: "
            f"{', '.join(result.missing_tags)}. Try a larger model or rephrase."
        )
    err.print(f"[green]Compiled[/green] in {elapsed:.1f}s ({result.attempts} attempt(s))")

    typer.echo(result.prompt)

    if copy:
        if copy_to_clipboard(result.prompt):
            err.print("[green]Copied to clipboard.[/green]")
        else:
            err.print("[yellow]Clipboard unavailable (install xclip/xsel on Linux).[/yellow]")


@app.command()
def stacks() -> None:
    """List available --stack profiles."""
    table = Table("Name", "Label", "Rules")
    for profile in list_stacks():
        table.add_row(profile.name, profile.label, str(len(profile.rules)))
    Console().print(table)


@app.command()
def doctor(
    model: Annotated[str, typer.Option("--model", "-m", envvar="PROMPTPUMP_MODEL")] = DEFAULT_MODEL,
    host: Annotated[str, typer.Option("--host", envvar="PROMPTPUMP_HOST")] = DEFAULT_HOST,
) -> None:
    """Check that Ollama is reachable and the model is installed."""
    client = OllamaClient(host=host, model=model)
    try:
        installed = client.list_models()
    except PromptPumpError as exc:
        err.print(f"[red]x[/red] {escape(str(exc))} Start it with `ollama serve`.")
        raise typer.Exit(1) from exc
    err.print(f"[green]ok[/green] Ollama reachable at {host}")
    if any(name == model or name.split(":")[0] == model for name in installed):
        err.print(f"[green]ok[/green] Model '{model}' is installed")
    else:
        err.print(f"[red]x[/red] Model '{model}' missing. Run: ollama pull {model}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
