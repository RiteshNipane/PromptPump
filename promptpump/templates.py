"""Structural schema, system prompt, and stack-specific rules.

Everything in this module is static data plus pure functions, so the same inputs
always produce the same system prompt (the deterministic half of PromptPump).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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
        StackProfile("javascript", "JavaScript", (
            "Match the repository's JavaScript version and module system; preserve its formatting and lint rules.",
            "Validate untrusted input at boundaries and handle rejected promises explicitly.",
            "Add focused tests using the test runner already configured in the project.",
        )),
        StackProfile("typescript", "TypeScript", (
            "Preserve strict compiler settings and use explicit types at public boundaries; avoid `any` where practical.",
            "Narrow unknown input with runtime checks or the validation library already used by the project.",
            "Keep async errors observable and test behavior with the configured test runner.",
        )),
        StackProfile("java", "Java", (
            "Follow the Java version and build tool configured by the repository; preserve package conventions.",
            "Prefer immutable values and clear interfaces; use try-with-resources for owned resources.",
            "Add JUnit tests for behavior and failure paths; do not introduce a second build system.",
        )),
        StackProfile("spring", "Spring Boot", (
            "Use constructor injection and keep controllers thin, with business logic in services.",
            "Use the project's existing Spring configuration and validation conventions; never hard-code secrets.",
            "Cover endpoints and service behavior with the existing test stack, including error responses.",
        )),
        StackProfile("kotlin", "Kotlin", (
            "Follow the repository's Kotlin and Gradle conventions; prefer null-safe types and immutable data.",
            "Use coroutines only where the surrounding APIs are suspendable and preserve structured concurrency.",
            "Add tests with the test framework already configured in the project.",
        )),
        StackProfile("csharp", ".NET / C#", (
            "Match the target framework and nullable-reference settings in the project files.",
            "Use async/await for asynchronous I/O, pass cancellation tokens through, and dispose owned resources.",
            "Follow existing dependency injection and configuration patterns; add tests with the configured test framework.",
        )),
        StackProfile("aspnet", "ASP.NET Core", (
            "Keep endpoint/controller responsibilities small and move domain behavior into services.",
            "Use dependency injection, options-based configuration, model validation, and structured logging.",
            "Propagate cancellation tokens and cover success, validation, and authorization paths with existing tests.",
        )),
        StackProfile("go", "Go", (
            "Follow gofmt and the module's current Go version; keep packages small and APIs explicit.",
            "Handle errors at the call site with context; avoid panics for expected failures.",
            "Pass context.Context through request-scoped work and add table-driven tests with the standard test package.",
        )),
        StackProfile("rust", "Rust", (
            "Follow the edition and dependency versions in Cargo.toml; use rustfmt and existing lint settings.",
            "Represent recoverable failures with Result and meaningful error types; avoid unnecessary cloning.",
            "Add unit or integration tests and run the repository's existing Cargo checks.",
        )),
        StackProfile("c", "C", (
            "Follow the project's C standard and build system; keep headers and source declarations consistent.",
            "Check allocation and I/O failures, define ownership clearly, and release resources on every path.",
            "Avoid undefined behavior and add tests using the repository's existing test harness.",
        )),
        StackProfile("cpp", "C++", (
            "Follow the configured C++ standard and build system; preserve existing compiler warning levels.",
            "Prefer RAII, standard-library types, and explicit ownership over manual resource management.",
            "Add tests through the project's current test framework and cover error paths.",
        )),
        StackProfile("php", "PHP", (
            "Follow the PHP version, Composer dependencies, and PSR conventions already configured.",
            "Use strict types where consistent with the codebase and validate data at external boundaries.",
            "Add tests with the existing PHPUnit or Pest setup; do not add duplicate dependencies.",
        )),
        StackProfile("laravel", "Laravel", (
            "Follow the installed Laravel version; keep controllers thin and use framework validation and policies.",
            "Use migrations for schema changes and Eloquent relationships/eager loading consistently.",
            "Cover behavior with the existing PHPUnit or Pest feature tests.",
        )),
        StackProfile("ruby", "Ruby", (
            "Follow the Ruby version, Bundler dependencies, and style conventions in the repository.",
            "Prefer small methods, explicit error handling, and the established application layers.",
            "Add tests with the configured RSpec or Minitest suite.",
        )),
        StackProfile("rails", "Ruby on Rails", (
            "Follow the installed Rails version and established MVC conventions; keep controllers focused.",
            "Use migrations for schema changes, strong parameters for input, and eager loading where needed.",
            "Cover behavior with the repository's existing RSpec or Rails test suite.",
        )),
        StackProfile("swift", "Swift", (
            "Follow the Swift version and package/build configuration; preserve the project's API availability targets.",
            "Use value types where appropriate, handle errors explicitly, and avoid force unwraps on uncertain values.",
            "Add tests using the test framework already configured for the target platform.",
        )),
        StackProfile("dart", "Dart", (
            "Follow the SDK constraints and analyzer settings in pubspec.yaml; use sound null safety.",
            "Keep asynchronous work typed and handle errors at the appropriate boundary.",
            "Add tests with the existing Dart test setup and preserve project formatting.",
        )),
        StackProfile("flutter", "Flutter", (
            "Follow the repository's Flutter/Dart versions and existing state-management approach.",
            "Keep widgets focused, preserve accessibility and responsive layouts, and avoid business logic in build methods.",
            "Add widget or unit tests using the existing Flutter test setup.",
        )),
        StackProfile("scala", "Scala", (
            "Follow the Scala version and build tool in the repository; do not mix incompatible APIs.",
            "Use the codebase's established functional or object-oriented style and model failures explicitly.",
            "Add tests with the configured Scala test framework.",
        )),
        StackProfile("elixir", "Elixir", (
            "Follow the Elixir and OTP versions and existing Mix project conventions.",
            "Keep processes supervised, use pattern matching for expected outcomes, and avoid blocking the BEAM scheduler.",
            "Add ExUnit tests for normal and failure cases.",
        )),
        StackProfile("r", "R", (
            "Follow the package or project structure and R version already in use; preserve reproducible dependency management.",
            "Use vectorized operations where clear and validate assumptions about missing or malformed data.",
            "Add tests with the project's existing testthat setup when present.",
        )),
        StackProfile("sql", "SQL / Database", (
            "Use the database dialect and migration tool detected in the project; parameterize values to prevent injection.",
            "Preserve transaction boundaries, constraints, and indexes; consider query plans for data-intensive changes.",
            "Test migrations and queries against the configured test database or existing fixtures.",
        )),
        StackProfile("postgresql", "PostgreSQL", (
            "Use the PostgreSQL version and migration conventions configured by the project.",
            "Preserve transaction semantics and use indexes, constraints, and parameterized queries deliberately.",
            "Test schema and query changes against the project's PostgreSQL test setup where available.",
        )),
        StackProfile("mysql", "MySQL", (
            "Follow the configured MySQL or MariaDB version and migration conventions.",
            "Use parameterized queries, explicit transaction boundaries, and indexes suited to the query patterns.",
            "Verify migrations and behavior with the project's existing database test setup.",
        )),
        StackProfile("mongodb", "MongoDB", (
            "Follow the installed MongoDB driver and existing sync/async access patterns.",
            "Validate document shape at the boundary, design indexes for actual query paths, and handle retries safely.",
            "Test repository behavior with the project's existing fixture or a mocked client.",
        )),
        StackProfile("bash", "Shell scripting", (
            "Use portable syntax compatible with the declared shell and quote expansions to preserve argument boundaries.",
            "Check command exit codes, use safe temporary-file handling, and avoid exposing secrets in logs.",
            "Add tests with the repository's existing shell test tooling or provide a clear manual verification command.",
        )),
        StackProfile("angular", "Angular", (
            "Follow the installed Angular version and established standalone/module architecture.",
            "Use typed services and reactive forms or signals according to project conventions; clean up subscriptions.",
            "Add tests with the configured Angular test runner and cover accessible user behavior.",
        )),
        StackProfile("vue", "Vue", (
            "Follow the installed Vue version and project conventions for Composition API or Options API.",
            "Use typed props and emits where TypeScript is enabled; keep state and side effects explicit.",
            "Add component tests with the existing Vue test tooling.",
        )),
        StackProfile("svelte", "Svelte", (
            "Follow the installed Svelte/SvelteKit version and existing routing and data-loading conventions.",
            "Keep component state reactive and validate form or endpoint input at its boundary.",
            "Add tests with the project's configured Svelte tooling.",
        )),
        StackProfile("express", "Express", (
            "Keep route handlers small, centralize error middleware, and forward asynchronous failures correctly.",
            "Validate and normalize external input before use; do not trust request payloads.",
            "Add endpoint tests using the test client and framework already present.",
        )),
        StackProfile("nestjs", "NestJS", (
            "Use modules, providers, and constructor injection consistently with the application architecture.",
            "Validate DTOs at request boundaries and centralize exception handling using existing filters.",
            "Add unit or e2e tests with the configured Nest testing utilities.",
        )),
        StackProfile("flask", "Flask", (
            "Use the app factory and blueprint patterns already present; keep route handlers focused.",
            "Validate external input and use the configured database/session lifecycle safely.",
            "Add tests with Flask's test client and the project's existing pytest fixtures.",
        )),
        StackProfile("sqlalchemy", "SQLAlchemy", (
            "Match the installed SQLAlchemy version and synchronous or asynchronous session style.",
            "Use explicit session lifetimes, parameterized expressions, and migrations through the existing tool.",
            "Test persistence behavior without leaking state between tests.",
        )),
        StackProfile("pydantic", "Pydantic", (
            "Follow the installed Pydantic major version and existing model configuration conventions.",
            "Use explicit field types and validators for boundary data; distinguish missing values from explicit nulls.",
            "Cover valid, invalid, and serialization cases without depending on network services.",
        )),
        StackProfile("redis", "Redis", (
            "Follow the project's client and sync/async conventions; set explicit timeouts and connection lifecycle.",
            "Use key namespaces and deliberate TTLs; make cache misses and Redis outages safe for the application.",
            "Test cache behavior with the existing test fixture or a mocked client.",
        )),
        StackProfile("celery", "Celery", (
            "Keep tasks idempotent where possible and configure retries only for transient failures.",
            "Pass serializable task arguments, set explicit time limits, and avoid logging sensitive payloads.",
            "Test task behavior eagerly or with the project's existing worker integration setup.",
        )),
        StackProfile("graphql", "GraphQL", (
            "Follow the existing schema-first or code-first approach and enforce authorization at resolvers/services.",
            "Validate input, bound pagination, and avoid N+1 data access with the established batching strategy.",
            "Add schema and resolver tests for success, validation, and authorization behavior.",
        )),
        StackProfile("grpc", "gRPC / Protocol Buffers", (
            "Treat protobuf definitions as the API contract and regenerate code with the repository's pinned tooling.",
            "Use deadlines, cancellation, and structured status errors; preserve compatibility of published messages.",
            "Test service behavior and serialization using the existing local test setup.",
        )),
        StackProfile("react-native", "React Native", (
            "Follow the installed React Native version and existing navigation and state-management conventions.",
            "Handle platform-specific behavior explicitly and preserve accessible touch targets and loading/error states.",
            "Add tests with the existing React Native Testing Library setup.",
        )),
        StackProfile("electron", "Electron", (
            "Keep privileged operations in the main process and expose a narrow, validated preload bridge.",
            "Keep context isolation and sandbox protections enabled; never pass untrusted data into shell commands.",
            "Test main/preload/renderer behavior with the existing project tooling.",
        )),
        StackProfile("pandas", "Pandas / NumPy", (
            "Preserve existing dataframe and array shapes, dtypes, indexes, and missing-value semantics.",
            "Prefer vectorized operations and avoid mutating caller-owned data unless the API promises mutation.",
            "Add tests for empty inputs, null values, and representative boundary cases.",
        )),
        StackProfile("scikit-learn", "scikit-learn", (
            "Keep preprocessing inside the fitted pipeline to prevent train/test leakage.",
            "Use deterministic random_state values where supported and preserve the project's split strategy.",
            "Evaluate with the existing metrics and test data shape and feature-name assumptions.",
        )),
        StackProfile("tensorflow", "TensorFlow / Keras", (
            "Follow the TensorFlow/Keras API version already installed and separate model, data, and training concerns.",
            "Make seeds, device selection, input shapes, and checkpoint behavior explicit and configurable.",
            "Add lightweight tests for model shapes and data pipelines without requiring downloads.",
        )),
        StackProfile("transformers", "Hugging Face Transformers", (
            "Use model/tokenizer identifiers and cache conventions already configured; do not download assets implicitly in tests.",
            "Keep preprocessing, generation settings, and device placement explicit and reproducible.",
            "Test with injected or tiny local fixtures so unit tests do not require network access.",
        )),
        StackProfile("pytest", "pytest", (
            "Follow existing fixture and parametrization patterns; keep tests independent and deterministic.",
            "Test observable behavior, including failure paths, and avoid broad exception assertions.",
            "Do not add a second test framework or require external services for unit tests.",
        )),
        StackProfile("playwright", "Playwright", (
            "Use the browser and fixture conventions already configured; prefer role- and label-based locators.",
            "Keep tests deterministic with isolated state and explicit waits for user-visible conditions.",
            "Avoid unnecessary fixed sleeps and cover the relevant browser/device targets already in CI.",
        )),
        StackProfile("docker", "Docker", (
            "Follow the existing base image and multi-stage build conventions; pin versions where reproducibility requires it.",
            "Run as a non-root user where practical, minimize layers, and never bake credentials into images.",
            "Keep build context small and verify health checks and runtime configuration.",
        )),
        StackProfile("terraform", "Terraform", (
            "Follow the configured Terraform/provider versions and existing module/state conventions.",
            "Use variables and outputs for reusable interfaces; avoid committing secrets or unmanaged state.",
            "Validate formatting and plans with the repository's configured checks before describing infrastructure effects.",
        )),
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
    "js": "javascript",
    "ts": "typescript",
    "golang": "go",
    "c++": "cpp",
    "cxx": "cpp",
    "c#": "csharp",
    "dotnet": "csharp",
    "asp.net": "aspnet",
    ".net": "csharp",
    "spring-boot": "spring",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "fast-api": "fastapi",
    "reactjs": "react",
    "next": "nextjs",
    "next.js": "nextjs",
    "nodejs": "node",
    "torch": "pytorch",
    "sklearn": "scikit-learn",
    "hf": "transformers",
    "reactnative": "react-native",
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
