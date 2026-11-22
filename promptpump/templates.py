"""Structural schema, system prompt, and stack-specific rules.

Everything in this module is static data plus pure functions, so the same inputs
always produce the same system prompt (the deterministic half of PromptPump).
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence

from .stack_profiles import (
    STACK_ALIASES,
    STACKS,
    StackProfile,
    UnknownStackError,
    get_stack,
    list_stacks,
)

__all__ = [
    "MODE_TAGS",
    "PromptMode",
    "REQUIRED_TAGS",
    "STACKS",
    "STACK_ALIASES",
    "StackProfile",
    "UnknownStackError",
    "build_system_prompt",
    "build_user_message",
    "get_required_tags",
    "get_stack",
    "list_stacks",
    "render_stack_section",
    "render_tag_list",
]

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


class PromptMode(str, Enum):
    """Supported prompt-enhancement workflows."""

    CODE = "code"
    IMAGE = "image"
    CONTENT = "content"


MODE_TAGS: dict[PromptMode, tuple[str, ...]] = {
    PromptMode.CODE: REQUIRED_TAGS,
    PromptMode.IMAGE: (
        "role_definition", "task_objective", "subject_and_scene", "visual_style",
        "composition_and_lighting", "constraints_and_guardrails", "expected_output_format",
    ),
    PromptMode.CONTENT: (
        "role_definition", "task_objective", "audience_and_context", "content_structure",
        "voice_and_style", "constraints_and_guardrails", "expected_output_format",
    ),
}


def get_required_tags(mode: PromptMode | str = PromptMode.CODE) -> tuple[str, ...]:
    """Return the output schema for a prompt-enhancement mode."""
    try:
        return MODE_TAGS[PromptMode(mode)]
    except ValueError:
        choices = ", ".join(item.value for item in PromptMode)
        raise ValueError(f"Unknown prompt type '{mode}'. Choose from: {choices}") from None


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
instead of copying the contradiction. Do not invent restrictions the request did not state \
(such as "only modify directory X"); name the files that should change and the existing \
files to reuse or read.
11. COMPLETE THE DATA FLOW: when the change adds or alters a field, parameter, or output, \
name every function that must produce or pass it through (for example the route handler that \
builds the response), not only the model that declares it. If the request says enforce, \
validate, or ensure, include a step that checks the result at runtime, reusing validation code \
from the outline when it exists, and state what happens when the check fails.

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
<expected_output_format>: Exactly how the executing LLM must reply (unified diffs for \
existing files, full contents only for new files, then a brief test plan). Nothing else.

{{STACK_SECTION}}"""

_CREATIVE_PROMPT_TEMPLATE = """\
You are PromptPump, a deterministic prompt enhancer for creative work.

Your job: turn a short request into ONE clear, detailed prompt for another generative AI. \
You do NOT create the image, story, script, or other final work yourself.

OUTPUT CONTRACT (strict):
1. Output only the enhanced prompt, with no greeting, explanation, or outer code fence.
2. Include exactly the seven XML sections below, once each, in this order:
{{TAG_LIST}}
3. Make every section concrete, useful, and faithful to the user's intent. Do not pad with filler.
4. Do not invent user preferences, facts, references, or source material. If a critical detail is \
unknown, preserve flexibility or identify it as an assumption in constraints.
5. Preserve named people, characters, settings, facts, and requested outcomes from the input.
6. Resolve ambiguity only when a safe, ordinary assumption is clear; otherwise keep the prompt \
open-ended and call out the choice the downstream generator should make.
7. Stay within the requested content type and scope. Do not add unrelated sections or deliverables.

{{MODE_RULES}}"""

_CREATIVE_MODE_RULES: dict[PromptMode, str] = {
    PromptMode.IMAGE: """IMAGE PROMPT RULES:
- Describe the subject, action, setting, and important visual details in concrete terms.
- Specify requested medium, art direction, palette, mood, composition, camera/viewpoint, and lighting.
- Include aspect ratio, resolution, text rendering, or negative constraints only when relevant or requested.
- Keep visual directions internally consistent and do not add unwanted objects, logos, or text.""",
    PromptMode.CONTENT: """CONTENT PROMPT RULES:
- Support stories, scripts, articles, speeches, posts, and other written or spoken content.
- Define audience, purpose, genre, voice, tone, point of view, and length when specified or useful.
- Give an appropriate structure: for example, story beats, script scenes/dialogue, or article sections.
- Preserve requested facts and themes; do not fabricate citations, quotations, or factual claims.
- State formatting and content boundaries clearly, including any requested rating or material to avoid.""",
    PromptMode.CODE: "",
}


def render_tag_list(tags: Sequence[str] = REQUIRED_TAGS) -> str:
    """Render the required tags as a numbered list for the system prompt."""
    return "\n".join(f"   {i}. <{tag}>...</{tag}>" for i, tag in enumerate(tags, 1))


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


def build_system_prompt(
    stacks: Sequence[StackProfile] = (), mode: PromptMode | str = PromptMode.CODE
) -> str:
    """Assemble the mode-specific system prompt for the local model."""
    selected = PromptMode(mode)
    tags = MODE_TAGS[selected]
    if selected is PromptMode.CODE:
        template = SYSTEM_PROMPT_TEMPLATE
        template = template.replace("{{STACK_SECTION}}", render_stack_section(stacks))
    else:
        template = _CREATIVE_PROMPT_TEMPLATE.replace(
            "{{MODE_RULES}}", _CREATIVE_MODE_RULES[selected]
        )
    return template.replace("{{TAG_LIST}}", render_tag_list(tags))


def build_user_message(
    shorthand: str, workspace_text: str = "", mode: PromptMode | str = PromptMode.CODE
) -> str:
    """Wrap the user request and, for code mode, its workspace snapshot."""
    selected = PromptMode(mode)
    if selected is PromptMode.CODE:
        return (
            f"<shorthand_request>\n{shorthand.strip()}\n</shorthand_request>\n\n"
            f"<workspace>\n{workspace_text.strip()}\n</workspace>\n\n"
            "Compile the master prompt now. Output only the seven XML-tagged sections."
        )
    return (
        f"<shorthand_request>\n{shorthand.strip()}\n</shorthand_request>\n\n"
        "Enhance this request as a prompt. Output only the seven XML-tagged sections."
    )
