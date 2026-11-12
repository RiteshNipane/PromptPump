"""Structural schema, system prompt, and stack-specific rules.

Everything in this module is static data plus pure functions, so the same inputs
always produce the same system prompt (the deterministic half of PromptPump).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# The seven tags every compiled master prompt must contain, in this exact order.
REQUIRED_TAGS: tuple[str, ...] = (
    "role_definition",
    "task_objective",
    "codebase_context",
    "architectural_specifications",
    "implementation_steps",
    "constraints_and_guardrails",
    "expected_output_format",
)


class UnknownStackError(ValueError):
    """Raised when a --stack value does not match a known profile."""


@dataclass(frozen=True)
class StackProfile:
    """Domain-specific rules injected into the system prompt."""

    name: str
    label: str
    rules: tuple[str, ...]


STACKS: dict[str, StackProfile] = {
    p.name: p
    for p in (
        StackProfile(
            "python",
            "Python",
            (
                "Target Python 3.10+; use type hints on every public function and `pathlib` for paths.",
                "Prefer the standard library; justify any new third-party dependency.",
                "Follow PEP 8, use dataclasses or typed models for structured data, and raise specific exceptions.",
                "Require pytest tests for new behaviour, including at least one failure-path test.",
            ),
        ),
        StackProfile(
            "fastapi",
            "FastAPI",
            (
                "Define request/response bodies as Pydantic models; never return raw dicts from routes.",
                "Split code into routers, schemas, services, and dependencies; keep route handlers thin.",
                "Use dependency injection (`Depends`) for DB sessions, auth, and settings.",
                "Use async handlers only when the underlying I/O is async; declare explicit status codes and error responses.",
                "Add tests with `TestClient` or `httpx.AsyncClient` covering success and validation errors.",
            ),
        ),
        StackProfile(
            "django",
            "Django",
            (
                "Keep business logic in models/services, not views; use class-based or function views consistently with the project.",
                "Every model change requires a migration; never edit applied migrations.",
                "Use the ORM with `select_related`/`prefetch_related` to avoid N+1 queries.",
                "Read configuration from settings/environment; never hard-code secrets.",
            ),
        ),
        StackProfile(
            "react",
            "React",
            (
                "Use function components and hooks only; keep components small and single-purpose.",
                "Colocate state with the lowest common parent; lift or use context only when needed.",
                "Type props explicitly (TypeScript if the project uses it) and handle loading, error, and empty states.",
                "Avoid unnecessary re-renders: stable keys, `useMemo`/`useCallback` only where measured or obviously needed.",
                "Cover behaviour with React Testing Library tests that query by role/label, not implementation details.",
            ),
        ),
        StackProfile(
            "nextjs",
            "Next.js",
            (
                "Respect the project's router (App Router vs Pages Router) and do not mix them.",
                "Default to Server Components; add `use client` only where interactivity requires it.",
                "Fetch data on the server where possible; validate all inputs in route handlers.",
            ),
        ),
        StackProfile(
            "node",
            "Node.js",
            (
                "Match the project's module system (ESM or CommonJS) and package manager.",
                "Use async/await with centralized error handling; never swallow rejected promises.",
                "Validate external input at the boundary and read config from environment variables.",
            ),
        ),
        StackProfile(
            "pygame",
            "Pygame",
            (
                "Structure the game around a single main loop: handle events, update state, then render, in that order.",
                "Use delta time (`clock.tick(fps) / 1000`) for movement; never tie speed to frame rate.",
                "Separate game state/logic from rendering; model entities as classes (or `pygame.sprite.Sprite` subclasses).",
                "Load assets once at startup, never inside the loop; call `pygame.quit()` on exit.",
                "Keep constants (screen size, FPS, colours) in one config location.",
            ),
        ),
        StackProfile(
            "pytorch",
            "PyTorch",
            (
                "Separate dataset, model, training loop, and evaluation into distinct modules.",
                "Seed all RNGs, set `model.train()`/`model.eval()` explicitly, and wrap inference in `torch.no_grad()`.",
                "Make device placement explicit and configurable; log metrics per epoch and checkpoint the best model.",
            ),
        ),
    )
}

# Friendly aliases accepted on the command line.
STACK_ALIASES: dict[str, str] = {
    "py": "python",
    "fast-api": "fastapi",
    "reactjs": "react",
    "next": "nextjs",
    "next.js": "nextjs",
    "nodejs": "node",
    "torch": "pytorch",
}


def get_stack(name: str) -> StackProfile:
    """Resolve a stack name or alias (case-insensitive) to its profile."""
    key = name.strip().lower()
    key = STACK_ALIASES.get(key, key)
    try:
        return STACKS[key]
    except KeyError:
        available = ", ".join(sorted(STACKS))
        raise UnknownStackError(f"Unknown stack '{name}'. Available stacks: {available}") from None


def list_stacks() -> list[StackProfile]:
    """Return all known stack profiles sorted by name."""
    return [STACKS[k] for k in sorted(STACKS)]


SYSTEM_PROMPT_TEMPLATE = """\
You are PromptPump, a deterministic prompt compiler for software engineers.

Your job: convert a short shorthand request plus workspace context into ONE high-density \
"master prompt" that another coding LLM will execute. You do NOT write the implementation \
and you do NOT answer the request yourself.

OUTPUT CONTRACT (strict):
1. Output ONLY the master prompt. No greeting, no explanation, no code fences around the \
whole output, no closing remarks.
2. The output MUST contain exactly these seven XML tags, once each, in this order:
{{TAG_LIST}}
3. Inside each tag use terse, imperative Markdown (bullets or numbered lists). No filler.
4. Ground everything in the <workspace> block. Reference only files, modules, and \
dependencies that appear there. If something is unknown, record it as an explicit \
assumption inside <constraints_and_guardrails>. Never invent file paths or APIs.
5. Do not copy the shorthand verbatim. Expand it into concrete, testable requirements.
6. REUSE BEFORE CREATE: if the workspace outline already contains a function or class that \
does part of the job, name it and extend or call it. Never propose a duplicate module or helper.
7. NO IMPLEMENTATION CODE: name files, functions, and signatures, but do not write function \
bodies or test bodies inside the prompt.
8. TARGET THE MECHANISM: state HOW the behaviour is achieved (for example: change the \
generation prompt vs post-process the answer), not just what the end result should look like.
9. UNKNOWNS: the workspace snapshot may be truncated. Any file, test, model, or API you cannot \
see must be listed under "Assumptions (verify before coding)" in <constraints_and_guardrails>.
10. STAY IN SCOPE: add no features, flags, files, or dependencies beyond the request. If the \
request contains contradictory or ambiguous parts, resolve them explicitly in the assumptions \
instead of copying the contradiction.

SECTION RULES:
<role_definition>: One short paragraph assigning the executing LLM a precise senior-engineer \
persona matched to the languages and frameworks visible in the workspace.
<task_objective>: One unambiguous, measurable objective, followed by success criteria and \
explicit out-of-scope items.
<codebase_context>: The facts from the workspace that matter: language, frameworks, entry \
points, and the files most likely to change.
<architectural_specifications>: Design decisions: files/modules to create or modify, data \
flow, key interfaces and signatures, naming and style conventions that match existing code.
<implementation_steps>: Ordered, numbered, small, verifiable steps; each names its target file.
<constraints_and_guardrails>: Hard rules: dependencies, compatibility, error handling, \
security, performance, testing, files not to touch, then an "Assumptions (verify before \
coding)" list.
<expected_output_format>: Exactly how the executing LLM must reply (for example: full file \
contents per path in fenced blocks, then a brief test plan). Nothing else.

{{STACK_SECTION}}"""


def render_tag_list() -> str:
    """Render the required tags as a numbered list for the system prompt."""
    return "\n".join(f"   {i}. <{tag}>...</{tag}>" for i, tag in enumerate(REQUIRED_TAGS, 1))


def render_stack_section(stacks: Sequence[StackProfile]) -> str:
    """Render stack-specific rules, or an explicit 'none' marker."""
    if not stacks:
        return "STACK RULES: none specified. Infer conventions from the workspace only."
    lines = [
        "STACK RULES (defaults: apply them in <architectural_specifications> and "
        "<constraints_and_guardrails>, but skip any that conflict with conventions "
        "visible in the workspace):"
    ]
    for profile in stacks:
        lines.append(f"[{profile.label}]")
        lines.extend(f"- {rule}" for rule in profile.rules)
    return "\n".join(lines)


def build_system_prompt(stacks: Sequence[StackProfile] = ()) -> str:
    """Assemble the full system prompt for the local model."""
    return SYSTEM_PROMPT_TEMPLATE.replace("{{TAG_LIST}}", render_tag_list()).replace(
        "{{STACK_SECTION}}", render_stack_section(stacks)
    )


def build_user_message(shorthand: str, workspace_text: str) -> str:
    """Wrap the shorthand request and workspace snapshot into the user turn."""
    return (
        f"<shorthand_request>\n{shorthand.strip()}\n</shorthand_request>\n\n"
        f"<workspace>\n{workspace_text.strip()}\n</workspace>\n\n"
        "Compile the master prompt now. Output only the seven XML-tagged sections."
    )
