"""Typer CLI entry point: ``promptpump run "shorthand" --stack python --copy``."""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from . import __version__
from .context import WorkspaceContext, scan_workspace
from .engine import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    CompileResult,
    OllamaClient,
    PromptCompiler,
    PromptPumpError,
)
from .stack_profiles import resolve_stacks
from .stats import compute_stats, format_stats
from .telemetry import (
    TelemetryReport,
    collect_hardware,
    render_markdown,
    run_telemetry,
    update_readme,
)
from .templates import PromptMode, UnknownStackError, list_stacks
from .tokens import TokenCounter, get_counter

DEFAULT_TELEMETRY_PROMPT = "add input validation and clear error messages to the main entry point"

app = typer.Typer(
    name="promptpump",
    help="Compile a short shorthand prompt into a dense, production-ready master prompt "
    "using the selected prompt mode and a local Ollama model.",
    no_args_is_help=False,
    invoke_without_command=True,
    add_completion=False,
)

# Progress and diagnostics go to stderr so stdout stays clean for piping the prompt.
err = Console(stderr=True)
_notes: list[str] = []  # plain-text copy of what `note` prints, for --log


def note(markup: str) -> None:
    """Print a message to stderr and remember its plain text for ``--log``."""
    err.print(markup)
    _notes.append(Text.from_markup(markup).plain)


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


def _emit(text: str, output: Optional[Path]) -> None:
    """Print ``text``, or write it as UTF-8 to ``output`` (avoids shell re-encoding)."""
    if output is None:
        typer.echo(text)
        return
    output.write_text(text.rstrip("\n") + "\n", encoding="utf-8")
    note(f"[green]Wrote[/green] {escape(str(output))}")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"promptpump {__version__}")
        raise typer.Exit()


def _counter_or_exit(kind: str) -> TokenCounter:
    try:
        return get_counter(kind)
    except ValueError as exc:
        note(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc


def _print_stats(context: WorkspaceContext, counter: TokenCounter) -> None:
    stats = compute_stats(context, counter, counter.name)
    table = Table(title="Context budget", show_header=False, title_justify="left")
    for label, value in format_stats(stats):
        table.add_row(label, value)
        _notes.append(f"{label}: {value}")
    err.print(table)
    if context.outlines:
        shown = ", ".join(f"{p} ({context.scores.get(p, 0.0):.1f})" for p in context.outlines)
        note(f"[dim]Outlined (relevance): {escape(shown)}[/dim]")
    if context.ranked and context.query:
        top = ", ".join(f"{p} ({s:.1f})" for p, s in context.ranked[:5])
        note(f"[dim]Top relevance for the prompt: {escape(top)}[/dim]")


def _print_model_stats(result: CompileResult) -> None:
    if not result.metrics:
        return
    m = result.metrics[-1]

    def f(value: Optional[float], digits: int = 2) -> str:
        return "N/A" if value is None else f"{value:.{digits}f}"

    note(
        f"[dim]Model-side: {m.prompt_tokens if m.prompt_tokens is not None else 'N/A'} input tokens "
        f"(Ollama prompt_eval_count), {m.output_tokens} output tokens, TTFT {f(m.ttft_s)}s, "
        f"{f(m.tokens_per_s, 1)} tokens/s[/dim]"
    )


@app.callback()
def main(
    ctx: typer.Context,
    version: Annotated[
        Optional[bool],
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = None,
) -> None:
    """PromptPump: locally enhance code, image, and content generation prompts."""
    _force_utf8()
    if ctx.invoked_subcommand is None:
        _interactive_menu(ctx)


@app.command()
def run(
    ctx: typer.Context,
    prompt: Annotated[str, typer.Argument(help="Shorthand prompt, e.g. 'add jwt auth'.")],
    prompt_type: Annotated[
        PromptMode,
        typer.Option("--type", help="Prompt type: code, image, or content."),
    ] = PromptMode.CODE,
    stack: Annotated[
        Optional[list[str]],
        typer.Option("--stack", "-s", help="Code-mode stack rules to inject (repeatable). See `stacks`."),
    ] = None,
    copy: Annotated[
        bool, typer.Option("--copy/--no-copy", help="Copy the result to the clipboard.")
    ] = True,
    path: Annotated[
        Path,
        typer.Option("--path", "-p", exists=True, file_okay=False, resolve_path=True,
                     help="Project directory to scan in code mode."),
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
    rank: Annotated[
        bool,
        typer.Option("--rank/--no-rank", help="Pack the budget by relevance to the prompt (BM25)."),
    ] = True,
    stats: Annotated[
        bool, typer.Option("--stats", help="Show token budget, reduction %, and model telemetry.")
    ] = False,
    tokenizer: Annotated[
        str, typer.Option("--tokenizer", help="Token counter: heuristic (default), tiktoken, auto.")
    ] = "heuristic",
    ground: Annotated[
        bool,
        typer.Option("--ground/--no-ground",
                     help="In code mode, overwrite <codebase_context> with deterministic scan facts."),
    ] = True,
    timeout: Annotated[float, typer.Option(help="Seconds to wait for the model.")] = 300.0,
    num_ctx: Annotated[
        int,
        typer.Option("--num-ctx", envvar="PROMPTPUMP_NUM_CTX", min=1024,
                     help="Model context window; lower it to save memory (needs input + ~1k)."),
    ] = 8192,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the payload; do not call the model.")
    ] = False,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", dir_okay=False, resolve_path=True,
                     help="Write the prompt to this file (UTF-8) instead of printing it."),
    ] = None,
    log: Annotated[
        Optional[Path],
        typer.Option("--log", dir_okay=False, resolve_path=True,
                     help="Also save the warnings and stats shown on stderr to this file (UTF-8)."),
    ] = None,
) -> None:
    """Enhance PROMPT into a structured prompt for the selected generation type."""
    _notes.clear()
    if log:
        ctx.call_on_close(lambda: log.write_text("\n".join(_notes) + "\n", encoding="utf-8"))
    if stack and prompt_type is not PromptMode.CODE:
        note(
            f"[yellow]warning:[/yellow] Ignoring --stack for --type {prompt_type.value}; "
            "stack profiles apply only to code prompts."
        )
        profiles = []
    else:
        try:
            profiles = resolve_stacks(stack or [])
        except UnknownStackError as exc:
            note(f"[red]error:[/red] {escape(str(exc))}")
            raise typer.Exit(2) from exc
    counter = _counter_or_exit(tokenizer)

    if prompt_type is PromptMode.CODE:
        with err.status("Scanning workspace..."):
            context = scan_workspace(
                path, max_tokens=max_context_tokens, counter=counter,
                query=prompt if rank else None,
            )
        note(
            f"[dim]Scanned {context.root.name}/: {len(context.tree_lines)} tree entries, "
            f"{len(context.key_files)} key files, {len(context.outlines)} outlined modules[/dim]"
        )
    else:
        context = WorkspaceContext(root=path)
        note(f"[dim]Using {prompt_type.value} prompt mode; workspace scanning is not needed.[/dim]")
    if stats and prompt_type is PromptMode.CODE:
        with err.status("Measuring whole-repo baseline..."):
            _print_stats(context, counter)
    elif stats:
        note("[dim]Workspace token statistics are available only for code prompts.[/dim]")

    compiler = PromptCompiler(
        OllamaClient(host=host, model=model, timeout=timeout, num_ctx=num_ctx), ground_context=ground
    )
    stack_names = [p.name for p in profiles]

    if dry_run:
        payload = "\n".join(
            f"===== {m['role'].upper()} =====\n{m['content']}\n"
            for m in compiler.build_messages(prompt, context, stack_names, prompt_type)
        )
        _emit(payload, output)
        raise typer.Exit()

    start = time.perf_counter()
    try:
        with err.status(f"Compiling with {escape(model)} (local)..."):
            result = compiler.compile(prompt, context, stack_names, prompt_type)
    except PromptPumpError as exc:
        note(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    elapsed = time.perf_counter() - start

    if result.missing_tags:
        note(
            f"[yellow]warning:[/yellow] model output is missing tags: "
            f"{', '.join(result.missing_tags)}. Try a larger model or rephrase."
        )
    if result.issues:
        note("[yellow]Check these references in the generated prompt (possible model mistakes):[/yellow]")
        for issue in result.issues[:10]:
            note(f"  [yellow]-[/yellow] {escape(issue.message)}")
    note(f"[green]Compiled[/green] in {elapsed:.1f}s with {escape(model)} ({result.attempts} attempt(s))")
    if stats:
        _print_model_stats(result)

    _emit(result.prompt, output)

    if copy:
        if copy_to_clipboard(result.prompt):
            note("[green]Copied to clipboard.[/green]")
        else:
            note("[yellow]Clipboard unavailable (install xclip/xsel on Linux).[/yellow]")


def _interactive_menu(ctx: typer.Context) -> None:
    """Show the entry menu and invoke the selected enhancement workflow."""
    err.print("[bold cyan]PromptPump[/bold cyan] — choose what kind of prompt to enhance")
    err.print("  [bold]1[/bold]  Code generation")
    err.print("  [bold]2[/bold]  Image generation")
    err.print("  [bold]3[/bold]  Content generation (stories, scripts, articles, etc.)")
    selected = typer.prompt("Selection", type=int, default=1)
    modes = {1: PromptMode.CODE, 2: PromptMode.IMAGE, 3: PromptMode.CONTENT}
    if selected not in modes:
        err.print("[red]Choose 1, 2, or 3.[/red]")
        raise typer.Exit(2)
    mode = modes[selected]
    shorthand = typer.prompt("Describe what you want the prompt to produce")
    ctx.invoke(run, ctx=ctx, prompt=shorthand, prompt_type=mode)


@app.command()
def menu(ctx: typer.Context) -> None:
    """Open the interactive prompt-type selector."""
    _interactive_menu(ctx)


@app.command()
def telemetry(
    path: Annotated[
        Path,
        typer.Option("--path", "-p", exists=True, file_okay=False, resolve_path=True,
                     help="Project directory used as the benchmark workload."),
    ] = Path("."),
    model: Annotated[str, typer.Option("--model", "-m", envvar="PROMPTPUMP_MODEL")] = DEFAULT_MODEL,
    host: Annotated[str, typer.Option("--host", envvar="PROMPTPUMP_HOST")] = DEFAULT_HOST,
    runs: Annotated[int, typer.Option(min=2, help="Total runs (first is a cold start).")] = 10,
    prompt: Annotated[str, typer.Option(help="Shorthand prompt to compile.")] = DEFAULT_TELEMETRY_PROMPT,
    stack: Annotated[Optional[list[str]], typer.Option("--stack", "-s")] = None,
    max_context_tokens: Annotated[int, typer.Option("--max-context-tokens", min=500)] = 3000,
    write_readme: Annotated[
        Optional[Path],
        typer.Option(help="Update the marked telemetry block in this README."),
    ] = None,
    json_out: Annotated[Optional[Path], typer.Option(help="Also save raw results as JSON.")] = None,
    timeout: Annotated[float, typer.Option(help="Seconds to wait per model call.")] = 600.0,
    num_ctx: Annotated[int, typer.Option("--num-ctx", envvar="PROMPTPUMP_NUM_CTX", min=1024)] = 8192,
) -> None:
    """Measure TTFT, tokens/sec, token counts, and peak GPU memory on this machine."""
    try:
        stack_names = [profile.name for profile in resolve_stacks(stack or [])]
    except UnknownStackError as exc:
        err.print(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc

    client = OllamaClient(host=host, model=model, timeout=timeout, num_ctx=num_ctx)
    compiler = PromptCompiler(client, verify=False)
    context = scan_workspace(path, max_tokens=max_context_tokens, query=prompt)
    try:
        with err.status("Starting...") as status:
            records = run_telemetry(
                client=client,
                compile_fn=lambda: compiler.compile(prompt, context, stack_names),
                runs=runs,
                progress=lambda i, n: status.update(
                    f"Run {i}/{n}" + (" (cold start)" if i == 1 else "")
                ),
            )
    except (PromptPumpError, RuntimeError) as exc:
        err.print(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    report = TelemetryReport(collect_hardware(client), prompt, records)  # model is loaded now
    markdown = render_markdown(report)
    typer.echo(markdown)
    if json_out:
        json_out.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        err.print(f"[green]Saved[/green] {json_out}")
    if write_readme:
        try:
            update_readme(write_readme, markdown)
        except (OSError, ValueError) as exc:
            err.print(f"[red]error:[/red] {escape(str(exc))}")
            raise typer.Exit(1) from exc
        err.print(f"[green]Updated[/green] {write_readme}")


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
