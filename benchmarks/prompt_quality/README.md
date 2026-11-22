# Prompt-quality pilot

This development-only benchmark compares three ways to brief the same local code-generation
model: a raw task request, that request plus the full fixture repository dump, and a prompt
compiled by PromptPump from the request and its scanned workspace context.

Each condition runs once for each of three small synthetic Python tasks. Generated code is
scored by the task's existing pytest suite. A task passes only when all of its tests pass;
PromptPump compile failures after two repair attempts and output-schema recovery count as
failures. The model is
`qwen2.5-coder:7b-instruct-q4_K_M` on Ollama 0.40.16. It generates code for every condition
and additionally compiles prompts for the PromptPump condition.

Run from the repository root with Ollama running and the model installed:

```powershell
python benchmarks/prompt_quality/run.py --model qwen2.5-coder:7b-instruct-q4_K_M
```

The runner writes a JSON report to `results/0.2.0-pilot.json`. Fixture requests, project
files, and tests live under `tasks/`. The pilot is exploratory: three tasks and one generation
per condition do not establish a general quality advantage.
