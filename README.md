# Qwen3-4B Local Inference

A self-contained [pixi](https://pixi.sh) project for running Qwen3-4B locally across different platforms and hardware configurations. Everything lives in `pixi.toml` — no Docker, no extra scripts.

## Goal

The objective was to explore whether a single pixi workspace could provide a true "pull and run" experience for local LLM inference, automatically adapting to the user's platform (Linux, macOS, Windows) and available hardware (CPU-only vs. NVIDIA GPU).

## What we learned

**Supporting multiple backends adds significant complexity.** Each backend (llama.cpp, vLLM) has its own dependency chain, platform constraints, model formats, and hardware tuning requirements. Supporting both meant isolated pixi environments with no shared dependencies, duplicated task definitions, and backend-specific workarounds. There's a real tradeoff between coverage and maintainability — it's worth scoping early how many backends and platforms you actually need to support rather than trying to cover everything.

**Cross-platform aspirations hit real limits.** The default CPU environment works everywhere (Linux, macOS, Windows). llama.cpp with CUDA works on both Linux and Windows, but vLLM has no macOS or Windows support at all. macOS remains CPU-only (though Apple Silicon's Metal support via llama.cpp is a future possibility). Each backend requires platform-specific wheels, so true "one environment fits all" GPU support isn't feasible.

**vLLM memory tuning is hardware-specific.** Getting vLLM to run Qwen3-4B on a 12 GB GPU required trial-and-error: `enforce_eager=True` (skip CUDA graphs), `max_model_len=1024`, `max_num_seqs=1`, and `gpu_memory_utilization=0.78`. These values are specific to this GPU — a 24 GB card would use completely different settings. The llama.cpp GGUF approach (Q4_K_M at ~2.7 GB) is far more forgiving on constrained hardware.

**Dependency isolation was essential.** vLLM, llama-cpp-python (CUDA), and llama-cpp-python (CPU) each have conflicting transitive dependencies (numpy versions, transformers versions, glibc requirements for wheel compatibility). Pixi's feature/environment system handled this well, but it meant dependencies couldn't be shared at the workspace level — each environment needed its own isolated dependency tree.

## Future directions

- **Auto-detection tool**: A small Python package (installable in the default env) that detects GPU presence, VRAM size, and platform, then recommends which environment to use and tunes parameters (context length, memory utilization, batch size) accordingly.
- **Smart default tasks**: The default environment's `chat` and `serve` tasks could invoke this detection tool, then delegate to the appropriate pixi environment (`pixi run -e cuda ...` or `pixi run -e vllm ...`) with hardware-appropriate flags.
- **Quantized vLLM models**: Using AWQ or GPTQ quantized models with vLLM would reduce VRAM requirements and make the vLLM environment viable on more GPUs without manual tuning.

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

| Environment | Command | Backend | Platforms |
|-------------|---------|---------|-----------|
| `default` | `pixi run chat` | llama.cpp CPU + BLAS | Linux, macOS, Windows |
| `cuda` | `pixi run -e cuda chat` | llama.cpp CUDA (GPU) | Linux |
| `vllm` | `pixi run -e vllm chat` | vLLM (GPU) | Linux |

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
| Context | 8192 tokens (default/cuda), 1024 tokens (vllm, constrained by 12 GB VRAM) |

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
