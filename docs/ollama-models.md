# Ollama model discovery and VRAM guide

PromptPump checks the models installed in the connected Ollama instance before compiling. It keeps the requested model when available; otherwise it selects an installed chat model, preferring code-tuned models for code prompts and general instruction/reasoning models for image and content prompts. It filters common embedding, reranking, speech, and audio-only model names. You do not need to install the default Qwen model if you already have another chat model such as `deepseek-r1:14b`.

`promptpump doctor` shows the model that will be used. Run `promptpump model --list` to list local chat models, `promptpump model` to choose one interactively, or `promptpump model deepseek-r1:14b` to save a particular model as the default. The choice is stored in `~/.promptpump/config.json`. The interactive startup menu also asks which model to use when multiple chat models are installed. The same discovery works when `OllamaClient()` is used without an explicit model. If no chat model is installed, PromptPump reports that and suggests pull commands instead of sending a request to a missing model.

## Popular models by GPU memory

The ranges below are rough starting points for Ollama's usual Q4 quantized tags. The listed Ollama download size is the model file size, **not** the complete VRAM requirement. Leave additional space for the context cache, Ollama runtime, desktop display, and other GPU workloads; lower `--num-ctx` when memory is tight. A model may still run partly on system RAM when it exceeds VRAM, at a lower speed.

| Approx. VRAM | Common candidates to try | Notes |
| --- | --- | --- |
| 4 GB | `qwen2.5-coder:3b`, `qwen3:4b`, `llama3.2:3b`, `deepseek-r1:1.5b`, `gemma3:4b`, `starcoder2:3b`, `granite-code:3b` | Prefer 1B–3B for headroom. Some 4B tags fit only with a short context or partial CPU offload. |
| 6–8 GB | `qwen2.5-coder:7b`, `qwen3:8b`, `deepseek-r1:7b` / `8b`, `llama3.1:8b`, `gemma3:4b`, `starcoder2:7b`, `mistral:7b`, `codellama:7b`, `granite-code:8b` | Strong practical range for many laptops. 7B/8B Q4 files are typically about 4–5 GB before runtime/context memory. |
| 10–16 GB | `qwen2.5-coder:14b`, `qwen3:14b`, `deepseek-r1:14b`, `gemma3:12b`, `phi4`, `codestral:22b`, `starcoder2:15b`, `codellama:13b`, `granite-code:20b` | 12B–15B Q4 files are usually around 8–10 GB; 20B–22B Q4 files are about 12–13 GB. Use a moderate context. |
| 16–24 GB | `gemma3:27b`, `qwen3:30b`, `qwen3:32b`, `deepseek-r1:32b`, `codellama:34b`, `granite-code:34b` | Q4 model files are roughly 17–20 GB. Longer contexts can push the run beyond this range. |
| 24–32 GB | Quantized 22B–34B models above, or larger models with smaller quantization/partial CPU offload | More headroom helps with context and concurrent workloads. Check the chosen tag's actual size and quantization before downloading. |

These are representative families, not a complete copy of Ollama's changing library. The library also contains community variants and custom tags. PromptPump does not require a model to be in this guide: any installed model that accepts Ollama's `/api/chat` endpoint can be selected as a fallback.

## Example pulls

```bash
# Code-focused
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:14b

# Reasoning/general instruction, including the DeepSeek fallback
ollama pull deepseek-r1:8b
ollama pull deepseek-r1:14b
ollama pull qwen3:8b
ollama pull gemma3:4b
```

Use `ollama list` to see local tags. The official library pages show current sizes and tags for [Qwen2.5 Coder](https://ollama.com/library/qwen2.5-coder), [Qwen3](https://ollama.com/library/qwen3), [DeepSeek-R1](https://ollama.com/library/deepseek-r1), [Gemma 3](https://ollama.com/library/gemma3), [Llama 3.2](https://ollama.com/library/llama3.2), [Phi-4](https://ollama.com/library/phi4), [Codestral](https://ollama.com/library/codestral), [StarCoder2](https://ollama.com/library/starcoder2), [Code Llama](https://ollama.com/library/codellama), and [Granite Code](https://ollama.com/library/granite-code).
