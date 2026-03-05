# Qwen3-4B Local Inference

Run Qwen3-4B locally with llama.cpp or vLLM — no Docker, no separate scripts, just `pixi.toml`.

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

The default environment runs on CPU and works on all platforms. GPU environments are available for Linux systems with an NVIDIA GPU.

| Environment | Command | Backend |
|-------------|---------|---------|
| `default` | `pixi run chat` | llama.cpp CPU + BLAS (all platforms) |
| `cuda` | `pixi run -e cuda chat` | llama.cpp CUDA (GPU, linux-64) |
| `vllm` | `pixi run -e vllm chat` | vLLM (GPU, linux-64) |

## Tasks

| Task | Description |
|------|-------------|
| `download` | Download Qwen3-4B Q4_K_M GGUF (~2.7 GB) — auto-runs before chat/serve in default and cuda envs |
| `serve` | Start OpenAI-compatible API at `http://localhost:8000` |
| `chat` | Interactive terminal chat |
| `test-message` | Send a test request to the running server |

## Model

| Attribute | Value |
|-----------|-------|
| Model | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |
| Quantization | Q4_K_M (~2.7 GB) — used by default and cuda envs |
| Source (GGUF) | [unsloth/Qwen3-4B-GGUF](https://huggingface.co/unsloth/Qwen3-4B-GGUF) |
| Source (vLLM) | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) — downloaded automatically |
| Context | 8192 tokens |

## Usage

```bash
# CPU chat (default, works everywhere)
pixi run chat

# GPU chat with llama.cpp (Linux with NVIDIA GPU)
pixi run -e cuda chat

# GPU chat with vLLM (Linux with NVIDIA GPU)
pixi run -e vllm chat

# API servers
pixi run serve                # CPU
pixi run -e cuda serve        # GPU (llama.cpp)
pixi run -e vllm serve        # GPU (vLLM)

# The server API is compatible with any OpenAI client
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello!"}]}'
```
