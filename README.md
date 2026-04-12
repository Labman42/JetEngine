<p align="center">
  <h1 align="center">JetEngine</h1>
  <p align="center">
    A high-performance inference engine for block diffusion language models
    <br />
    <a href="https://jetastra.github.io/SDAR/">SDAR</a> · <a href="https://github.com/HKUNLP/LLaDA">LLaDa</a> · <a href="https://github.com/dllm-team/dLLM-Var">dLLM-Var</a>
  </p>
</p>

---

JetEngine is a lightweight, production-ready inference engine for **block diffusion language models** (SDAR, LLaDa, dLLM-Var). It supports dense and MoE architectures, hybrid Data Parallel + Tensor Parallel distributed inference, CUDA graph acceleration, and advanced remasking strategies for optimal generation quality.

### Benchmarks (SDAR-4B, block_length=4, batch_size=128)

| GPU | Throughput | Key Optimizations |
|-----|-----------|-------------------|
| **NVIDIA H200** | **5,600+ tok/s** | FA3 + CUDA Graphs + Chain Mechanism |
| **NVIDIA A800** | **3,000+ tok/s** | FA2 + Triton Kernels + Paged KV Cache |

## Features

- **Block Diffusion Decoding** — generates tokens in fixed-size blocks via iterative denoising, fundamentally different from autoregressive decoding
- **11 Remasking Strategies** — from simple sequential to novel joint-distribution-aware multi-token commit (`dynamic_pmax`)
- **Flash Attention 3** — Hopper SM90 paged attention for denoise, with FA2/flashinfer fallback
- **CUDA Graph Capture** — graphs for batch sizes 1–128, near-zero kernel launch overhead
- **Chain Mechanism** — runs up to 5 denoising steps within a single `step()` call, eliminating scheduler overhead
- **Hybrid DP+TP** — `tensor_parallel_size` × data parallel across any GPU count (e.g., TP=2 × DP=4 on 8 GPUs)
- **Streaming Generation** — `generate_streaming()` streams prompts through a fixed active window with automatic prompt interleaving for batch diversity
- **Model Offloading** — completely offload model weights and KV cache to free GPU memory for RL training loops
- **Selective Logits** — computes LM head only for DENOISING sequences, skipping SAVING sequences

## Installation

**Requirements:**
```
Python >= 3.10
PyTorch >= 2.1
transformers >= 4.52.4
flash-attn >= 2.5
accelerate
```

**Install:**
```bash
pip install flash-attn --no-build-isolation
git clone https://github.com/Labman42/JetEngine.git
cd JetEngine
pip install .
```

**Optional (for Hopper GPUs):**
```bash
# Flash Attention 3 — enables FA3 paged attention (significant speedup on H100/H200)
pip install flash-attn-3
```

## Quick Start

### Single GPU

```bash
CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu example.py
```

### Multi-GPU (Data Parallel)

```bash
# Uses all visible GPUs for data parallel inference
accelerate launch --multi_gpu example.py
```

### Hybrid TP + DP

```bash
# 8 GPUs: TP=2 (model split across 2 GPUs) × DP=4 (4 data shards)
accelerate launch --multi_gpu --num_processes=8 your_script.py \
    --tensor_parallel_size 2
```

## Usage

### Basic Inference

```python
from jetengine import LLM, SamplingParams

# Initialize engine
llm = LLM(
    "path/to/SDAR-4B-Chat",
    mask_token_id=151669,      # Required: model's mask token
    block_length=4,            # Required: block diffusion block size
    tensor_parallel_size=1,    # TP degree (1 = single GPU)
    max_model_len=4096,
    gpu_memory_utilization=0.9,
)

# Configure sampling
sampling_params = SamplingParams(
    temperature=1.0,
    max_tokens=4096,
    block_length=4,
    denoising_steps=4,
    remasking_strategy="dynamic_pmax",  # Best strategy for accuracy
    dynamic_threshold=0.75,             # Commit threshold for multi-token
    topk=0,
    topp=1.0,
)

# Generate
outputs = llm.generate_streaming(
    prompts,                   # List[str] or List[List[int]]
    sampling_params,
    max_active=128,            # Max concurrent sequences
)

for output in outputs:
    print(output["text"])
```

### Batch Generation (Non-Streaming)

```python
outputs = llm.generate(
    ["Solve x^2 = 4", "What is pi?"],
    sampling_params,
)
```

### Model Offloading (for RL Training)

```python
# Free GPU memory after inference for training
llm.offload_parameters()          # Free model weights (keep buffers)
llm.free_all_resources()          # Free everything (graphs + KV cache)

# Reload from a HuggingFace model for the next eval round
llm.reload_from_hf_model(hf_model)
```

## Remasking Strategies

The remasking strategy controls **which positions** in a block to commit at each denoising step and **how many** to commit simultaneously. This is the core design knob for block diffusion quality and speed.

### Recommended

| Strategy | Multi-Token | Description | Best For |
|----------|:-----------:|-------------|----------|
| **`dynamic_pmax`** | Yes | Threshold on P(argmax) per position; commits argmax token. EOS-safe. | **Best pass@1 & pass@k** |
| `low_confidence_dynamic` | Yes | Threshold on P(sampled); fallback to leftmost | General use |

### Position-Based (Single Token per Step)

| Strategy | Description |
|----------|-------------|
| `sequential` | Commit leftmost masked position (left-to-right) |
| `anti_sequential` | Commit rightmost masked position |
| `low_confidence_static` | Commit position with highest P(sampled) |
| `least_entropy` | Commit position with lowest entropy |
| `top2_margin` | Commit position with largest top1-top2 probability gap |

### Multi-Token Commit

| Strategy | Description |
|----------|-------------|
| `dynamic_pmax` | P(argmax) > threshold; commits argmax tokens with EOS safety |
| `low_confidence_dynamic` | P(sampled) > threshold; fallback to leftmost |
| `entropy_bounded` | Commit positions sorted by entropy up to a budget |
| `causal_waterfall` | Contiguous leftmost prefix where each P(argmax) > floor |
| `logit_salience` | Z-score outlier detection in logit space |
| `relative_top` | Commit prefix where P(argmax) ≥ α × max across block |
| `consensus_commit` | K=4 multinomial majority vote |

### Strategy Comparison (MATH-500, T=1.0, gen4, 8×H200)

| Strategy | pass@1 | pass@4 |
|----------|--------|--------|
| **dynamic_pmax (t=0.75)** | **0.698** | **0.840** |
| low_confidence_dynamic (t=0.90) | 0.682 | 0.834 |
| sequential | 0.669 | 0.730 |
| low_confidence_static | 0.677 | 0.750 |

> At gen16, `dynamic_pmax` achieves **pass@16 = 0.96** vs dynamic's 0.93 (+3%).

## Configuration Reference

### `LLM()` Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | — | Path to model checkpoint |
| `mask_token_id` | — | **Required.** Model's mask token ID |
| `block_length` | 4 | Block size for diffusion decoding |
| `tensor_parallel_size` | 1 | Number of GPUs for tensor parallelism |
| `max_num_seqs` | 512 | Maximum sequences in KV cache |
| `max_model_len` | 4096 | Maximum sequence length |
| `gpu_memory_utilization` | 0.8 | Fraction of GPU memory for KV cache |
| `enforce_eager` | False | Disable CUDA graphs (for debugging) |
| `kvcache_block_size` | 256 | KV cache page size (must be multiple of 256) |
| `dtype` | "auto" | Model dtype (`"auto"`, `"bfloat16"`, `"float16"`) |

### `SamplingParams` Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 1.0 | Sampling temperature |
| `max_tokens` | 64 | Maximum completion length |
| `block_length` | 4 | Block size (must match LLM) |
| `denoising_steps` | 4 | Denoising iterations per block |
| `remasking_strategy` | `"low_confidence_static"` | Strategy for token commitment |
| `dynamic_threshold` | 0.75 | Confidence threshold for dynamic/pmax strategies |
| `topk` | 0 | Top-k filtering (0 = disabled) |
| `topp` | 1.0 | Top-p (nucleus) filtering |
| `repetition_penalty` | 1.0 | Repetition penalty |
| `eb_threshold` | 0.35 | Entropy budget for `entropy_bounded` strategy |
| `pos_temp_slope` | 0.0 | Position-temperature: T_i = T × (1 + slope × i/(L-1)) |
| `stop_words` | None | Token IDs that trigger sequence termination |

## Supported Models

| Model | Type | `mask_token_id` | `block_length` |
|-------|------|:--------------:|:--------------:|
| [SDAR-1.7B/4B-Chat](https://huggingface.co/JetLM) | Dense | 151669 | 4 |
| SDAR-MoE | MoE | 151669 | 4 |
| [LLaDa-8B-Instruct](https://github.com/HKUNLP/LLaDA) | Dense | 126336 | 1024 |
| [dLLM-Var](https://github.com/dllm-team/dLLM-Var) | Dense | 126336 | 64 |

### SDAR Example

```python
llm = LLM("SDAR-4B-Chat", mask_token_id=151669, block_length=4)
sp = SamplingParams(temperature=1.0, max_tokens=4096, block_length=4,
                    denoising_steps=4, remasking_strategy="dynamic_pmax",
                    dynamic_threshold=0.75)
```

### LLaDa Example

```python
llm = LLM("LLaDA-8B-Instruct", mask_token_id=126336, block_length=1024,
           gpu_memory_utilization=0.9)
sp = SamplingParams(temperature=1.0, max_tokens=2048, block_length=1024,
                    denoising_steps=1024, remasking_strategy="low_confidence_dynamic",
                    dynamic_threshold=0.90)
```

> **Tip:** Set `block_length` > prompt length for pure diffusion mode. With `block_length` < prompt length, JetEngine uses prefill + block diffusion (hybrid mode with interesting behaviors).

### dLLM-Var Example

```python
llm = LLM("dLLM-Var", mask_token_id=126336, block_length=64,
           gpu_memory_utilization=0.9)
sp = SamplingParams(temperature=1.0, max_tokens=2048, block_length=64,
                    denoising_steps=64, remasking_strategy="low_confidence_dynamic",
                    dynamic_threshold=0.90)
```

## Architecture

```
LLM (llm.py)
 └─ LLMEngine (engine/llm_engine.py)
     ├─ ModelRunner (engine/model_runner.py)
     │   ├─ SDAR / SDAR-MoE / LLaDa model
     │   ├─ Flash Attention 3 / flashinfer / FA2 (layers/attention.py)
     │   ├─ CUDA Graph capture & replay (bs=1..128)
     │   └─ Paged KV Cache (kvcache_block_size=256)
     ├─ Scheduler (engine/scheduler.py)
     │   ├─ Block Manager — allocate/deallocate KV cache pages
     │   ├─ postprocess_unify() — batched sampling + strategy dispatch
     │   │   ├─ Dense path (all masked, step 0)
     │   │   ├─ Sparse path (partial masks, steps 1+)
     │   │   └─ Position fast-path (sequential/anti_sequential)
     │   └─ Chain mechanism — up to 5 steps per step() call
     └─ DistributedManager — hybrid DP + TP via accelerate
```

### Two Operating Modes

**Mode 1: Ideal Decode** (`total_seqs ≤ max_active`) — all sequences prefill together, denoise together, drain together. Chain=5 (full block per step). Maximum GPU utilization.

**Mode 2: Streaming Decode** (`total_seqs > max_active`) — sequences stream through via `generate_streaming()`. Prompts are automatically interleaved for batch diversity. Adaptive chain depth (3–5) based on pending queue.

## Optimization History

| Opt | Description | pass@1 | tok/s (H200) |
|-----|-------------|--------|:------------:|
| baseline | Original | 0.683 | ~1,500 |
| opt17 | CUDA graph 1-128, selective logits | — | 2,677 |
| opt18 | flashinfer paged attention | — | 4,109 |
| opt22 | Lazy entropy in chain intermediate | 0.709 | 4,242 |
| opt23 | Sparse logits (LM head only masked) | 0.711 | 4,268 |
| opt24 | **Flash Attention 3** (Hopper SM90) | 0.722 | **5,677** |
| opt25 | Chain=3 for pending prefills + FA3 prefill | 0.705 | 5,677 |
| fix | **Gen64 quality fix** (multinomial + interleave) | 0.685 | — |
| opt26 | **TP support fixes** + sequential fast-path | 0.666 | +27% seq |
| opt27 | **`dynamic_pmax`** strategy (P(argmax) threshold) | **0.696** | — |
| opt28 | Threshold tuning (0.9 → 0.75) | **0.698** | — |

## Notice

For pure diffusion models (LLaDa, dLLM-Var), the logits tensor scales with `batch × context_length × vocab_size` and can exhaust GPU memory. Mitigations:
- Decrease `max_num_seqs` in `LLM()` initialization
- Decrease `max_active` in `generate_streaming()`
- Use `gpu_memory_utilization=0.9` or lower

## Contact

For issues or inquiries:
- **Yihan Bian**, University of Maryland, College Park — [ybian@umd.edu](mailto:ybian@umd.edu)
- GitHub Issues: [Labman42/JetEngine](https://github.com/Labman42/JetEngine/issues)
