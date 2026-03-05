# Qwen3-4B Local Inference

Run Qwen3-4B locally with llama.cpp — no Docker, no separate scripts, just `pixi.toml`.

## Quick Start

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Interactive chat in the terminal (CPU by default)
pixi run chat

# Or start an OpenAI-compatible API server
pixi run serve

# Test the server (in another terminal)
pixi run test-message
```

## Environments

The default environment runs on CPU and works on all platforms. A CUDA environment is available for Linux systems with an NVIDIA GPU.

| Environment | Command | Backend |
|-------------|---------|---------|
| `default` | `pixi run chat` | CPU + BLAS (all platforms) |
| `cuda` | `pixi run -e cuda chat` | CUDA (GPU, linux-64) |

## Tasks

| Task | Description |
|------|-------------|
| `pixi run download` | Download Qwen3-4B Q4_K_M GGUF (~2.7 GB) |
| `pixi run serve` | Start OpenAI-compatible API at `http://localhost:8000` |
| `pixi run chat` | Interactive terminal chat |
| `pixi run test-message` | Send a test request to the running server |

`serve` and `chat` automatically run `download` first if the model isn't present.

## Model

| Attribute | Value |
|-----------|-------|
| Model | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |
| Quantization | Q4_K_M (~2.7 GB) |
| Source | [unsloth/Qwen3-4B-GGUF](https://huggingface.co/unsloth/Qwen3-4B-GGUF) |
| Context | 8192 tokens |
| GPU env | Linux (CUDA 12+) |
| CPU env | Linux, macOS (Apple Silicon), Windows |

## Usage

```bash
# CPU chat (default, works everywhere)
pixi run chat

# GPU chat (Linux with NVIDIA GPU)
pixi run -e cuda chat

# CPU API server (default)
pixi run serve

# GPU API server
pixi run -e cuda serve

# The server API is compatible with any OpenAI client
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello!"}]}'
```
