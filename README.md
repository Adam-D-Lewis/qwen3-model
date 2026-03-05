# Qwen3-4B Local Inference

Run Qwen3-4B locally with llama.cpp — no Docker, no separate scripts, just `pixi.toml`.

## Quick Start

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Interactive chat in the terminal
pixi run chat

# Or start an OpenAI-compatible API server
pixi run serve

# Test the server (in another terminal)
pixi run test-chat
```

## Tasks

| Task | Description |
|------|-------------|
| `pixi run download` | Download Qwen3-4B Q4_K_M GGUF (~2.7 GB) |
| `pixi run serve` | Start OpenAI-compatible API at `http://localhost:8000` |
| `pixi run chat` | Interactive terminal chat |
| `pixi run test-chat` | Send a test request to the running server |

`serve` and `chat` automatically run `download` first if the model isn't present.

## Model

| Attribute | Value |
|-----------|-------|
| Model | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |
| Quantization | Q4_K_M (~2.7 GB) |
| Source | [unsloth/Qwen3-4B-GGUF](https://huggingface.co/unsloth/Qwen3-4B-GGUF) |
| Context | 8192 tokens (configurable) |
| Platforms | Linux, macOS (Apple Silicon), Windows |

## Configuration

Override defaults with environment variables:

```bash
# CPU-only (no GPU)
N_GPU_LAYERS=0 pixi run chat

# The server API is compatible with any OpenAI client
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello!"}]}'
```
