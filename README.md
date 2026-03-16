# Qwen3-4B Local Inference

A self-contained [pixi](https://pixi.sh) project for running Qwen3-4B locally across different platforms and hardware configurations. Everything lives in `pixi.toml` — no Docker, no extra scripts.

## Goal

The objective was to explore whether a single pixi workspace could provide a true "pull and run" experience for local LLM inference, automatically adapting to the user's platform (Linux, macOS, Windows) and available hardware (CPU-only vs. NVIDIA GPU).

## What we learned

**Supporting multiple backends adds significant complexity.** Each backend (llama.cpp, vLLM) has its own dependency chain, platform constraints, model formats, and hardware tuning requirements. Supporting both meant isolated pixi environments with no shared dependencies, duplicated task definitions, and backend-specific workarounds. There's a real tradeoff between coverage and maintainability — it's worth scoping early how many backends and platforms you actually need to support rather than trying to cover everything.

**Cross-platform aspirations hit real limits.** The default CPU environment works everywhere (Linux, macOS, Windows). llama.cpp with CUDA works on both Linux and Windows, but vLLM has no macOS or Windows support at all. macOS remains CPU-only (though Apple Silicon's Metal support via llama.cpp is a future possibility). Each backend requires platform-specific wheels, so true "one environment fits all" GPU support isn't feasible.

**vLLM memory tuning is hardware-specific.** Getting vLLM to run Qwen3-4B on a 12 GB GPU required trial-and-error: `enforce_eager=True` (skip CUDA graphs), `max_model_len=1024`, `max_num_seqs=1`, and `gpu_memory_utilization=0.78`. These values are specific to this GPU — a 24 GB card would use completely different settings. The llama.cpp GGUF approach (Q4_K_M at ~2.7 GB) is far more forgiving on constrained hardware.

**Dependency isolation was essential.** vLLM, llama-cpp-python (CUDA), and llama-cpp-python (CPU) each have conflicting transitive dependencies (numpy versions, transformers versions, glibc requirements for wheel compatibility). Pixi's feature/environment system handled this well, but it meant dependencies couldn't be shared at the workspace level — each environment needed its own isolated dependency tree.

## Hardware compatibility

Matching environments to hardware is a fundamental problem in package management. This project currently handles it with manually-selected pixi environments (`default`, `cuda`, `vllm`), but the conda ecosystem has established patterns for automating this.

### Virtual packages (pseudo-packages for hardware)

Conda and pixi use **virtual packages** — solver-visible packages that represent hardware capabilities rather than installable software. The solver detects these automatically:

| Virtual package | Represents |
|----------------|------------|
| `__cuda` | CUDA driver version |
| `__glibc` | System glibc version |
| `__osx` | macOS version |
| `__linux` | Linux kernel version |
| `__archspec` | CPU microarchitecture (e.g., x86_64_v3) |

This project already uses pixi's version of this mechanism via `system-requirements`:

```toml
[feature.cuda]
system-requirements = { cuda = "12" }

[feature.vllm]
system-requirements = { cuda = "12", libc = "2.31" }
```

These constraints tell the solver "this environment requires specific hardware" — the same concept as depending on a pseudo-package like `__cuda >=12`.

### What's missing: GPU architecture, VRAM, and RAM

Conda and pixi currently only express CUDA **driver version** — not GPU compute capability, VRAM, or system RAM. For LLM inference, these are the real constraints:

| Constraint | Why it matters | Current support |
|-----------|---------------|-----------------|
| GPU compute capability (sm_XX) | vLLM requires ≥ 7.0 (Volta); bfloat16/FP8 requires ≥ 8.0 (Ampere) | **Not in conda/pixi**. Spack has `cuda_arch=80`; Nix has `cudaCapabilities = ["8.9"]` |
| GPU VRAM | vLLM needs ~12 GB for Qwen3-4B; llama.cpp GGUF needs ~3 GB | **No package manager supports this** |
| System RAM | CPU inference of a 4B Q4 model needs ~4-8 GB RAM | **No package manager supports this** |

#### GPU architectures and LLM requirements

| Architecture | Compute Capability | Example GPUs | LLM inference status |
|---|---|---|---|
| Pascal | sm_60-61 | GTX 1080, P100 | llama.cpp only; vLLM unsupported |
| Volta | sm_70 | V100 | vLLM minimum; FP16 only |
| Turing | sm_75 | RTX 2080, T4 | Supported; some quantization methods blocked |
| Ampere | sm_80-86 | RTX 3090, A100 | Full support including bfloat16, FP8, all quantization |
| Ada Lovelace | sm_89 | RTX 4090, L40 | Full support |
| Hopper | sm_90 | H100 | Best performance, FP8 native |

#### Precedent from other package managers

**Spack** has the strongest model for GPU architecture constraints. Its `CudaPackage` mixin provides a `cuda_arch` variant:

```yaml
# spack packages.yaml — set default GPU architecture for all packages
packages:
  all:
    variants:
      - cuda_arch=80
```

Packages propagate `cuda_arch` through their dependency tree, and each architecture value constrains which CUDA toolkit versions are compatible (e.g., `cuda_arch=90` requires `cuda@11.8:`).

**Nix** takes a similar approach with `cudaCapabilities` in the nixpkgs config, and even supports per-architecture package sets: `pkgsForCudaArch.sm_89.python3Packages.torch`.

Neither Spack nor Nix handle VRAM or RAM — those remain an open gap across all package managers.

### Package variants

**Package variants** are how conda-forge publishes multiple builds of the same package for different hardware targets (e.g., `pytorch` for CPU, CUDA 11.8, CUDA 12.1, ROCm). The solver selects the right variant based on which virtual packages are present. This is the mechanism the conda ecosystem uses to avoid requiring users to manually choose between hardware-specific packages.

### The universal binary problem

You can't have entirely universal binary artifacts. The options are:

- **Pre-built variants**: Multiple builds per hardware target (conda-forge's approach, and what this project does with separate environments)
- **Compile at install time**: Build from source against local hardware (e.g., `pip install llama-cpp-python` with `CMAKE_ARGS=-DGGML_CUDA=on`)
- **Runtime dispatch**: Ship multiple code paths, detect hardware at runtime (e.g., how some libraries load CPU vs GPU shared libraries dynamically)

### How this project maps to these patterns

This project uses **pre-built variants via separate pixi environments** — the manual version of what package variants automate. The `cuda` environment pulls CUDA-specific wheels from a dedicated index (`abetlen.github.io/llama-cpp-python/whl/cu124`), while `default` uses the CPU-only build from PyPI. A more automated approach would encode hardware requirements as package-level constraints so the solver itself picks the right backend, removing the need for users to manually select `-e cuda` vs `-e vllm`.

Ideally, pixi could gain support for additional virtual packages like `__cuda_arch` (compute capability), `__gpu_vram`, and `__ram` — allowing environments to declare constraints like "needs Ampere+ GPU with 12GB+ VRAM" directly in `pixi.toml`. Until then, the auto-detection tool (see Future Directions) is the practical workaround.

## Future directions

- **Solver-driven hardware selection**: Rather than manual environment selection, encode hardware requirements as virtual package constraints so `pixi run chat` automatically resolves to the best backend for the user's hardware. This is the direction that conda's package variant system is designed to support.
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
