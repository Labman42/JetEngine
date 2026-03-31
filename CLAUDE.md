# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is JetEngine

JetEngine is a lightweight inference engine for block diffusion language models (SDAR, LLaDa, dLLM-Var). It supports dense and MoE architectures, tensor parallelism, CUDA graph capture, paged KV caching, and flashinfer acceleration. Built from nanovllm.

## Build & Run

```bash
# Install (requires flash-attn, CUDA GPU)
pip install flash-attn --no-build-isolation
pip install .

# Single GPU (must use accelerate or torchrun for distributed init)
CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu example.py
# or
CUDA_VISIBLE_DEVICES='0' torchrun --nproc_per_node=1 example.py

# Multi-GPU (data parallel by default, uses all visible GPUs)
accelerate launch --multi_gpu example.py
```

No test suite exists in this repo. Correctness is verified through end-to-end MATH-500 evaluation (see Evaluation section below).

## Architecture

### Entry Point & Engine

`LLM` (in `jetengine/llm.py`) is a thin subclass of `LLMEngine` (`jetengine/engine/llm_engine.py`). The engine orchestrates the full inference loop:

1. **`add_request()`** — tokenizes prompt, creates a `Sequence`, hands it to the `Scheduler`
2. **`step()`** — calls `scheduler.schedule()` to get prefill/denoise batches, runs them through `ModelRunner`, then postprocesses logits back through the scheduler
3. **`generate()`** — batched generation where all prompts are added upfront, loop `step()` until done
4. **`generate_streaming()`** — streaming generation with backpressure: `max_active` controls how many sequences run concurrently, new prompts are added as old ones finish

### Two Operating Modes

JetEngine operates in two distinct regimes depending on the relationship between total requests and `max_active`:

#### Mode 1: Ideal Decode (total_seqs ≤ max_active)

All sequences fit in memory simultaneously. The flow is:
1. **Prefill phase**: All sequences get prefilled in one batch
2. **Pure denoise phase**: All sequences denoise together with no interruptions
3. **Drain phase**: Sequences finish at different times, batch size shrinks

In this mode, there is no interleaving of prefill and denoise. The chain mechanism (chain=5 when no pending) completes a full block (4 denoising steps + SAVING) in a single scheduler step. This maximizes GPU utilization since the batch stays large.

Profile with: `tests/bench_throughput.py` or `tests/profile_realistic.py`

#### Mode 2: Streaming Decode (total_seqs > max_active)

More sequences than can fit in memory. The engine uses `generate_streaming()`:
1. Initial `max_active` sequences get prefilled
2. Denoise steps run on active sequences
3. As sequences finish and free KV cache blocks, new sequences from the waiting queue get prefilled
4. Prefill and denoise alternate in each `step()` call

In this mode, prefill of new sequences interleaves with denoise of active sequences. The chain depth is limited (chain=2 when prefills are pending) to let the scheduler refill slots. This introduces more scheduling overhead but keeps GPU utilization high.

Profile with: `tests/profile_streaming.py`

The MATH-500 evaluation is an instance of Mode 2: 500 problems × num_generations (e.g. 4) = 2000 total sequences, but max_active is typically 64-128.

### Scheduler (`jetengine/engine/scheduler.py`)

Manages sequence lifecycle through states: `WAITING → PREFILLING → DENOISING → SAVING → FINISHED`. Key responsibilities:
- **Block management** — allocates/deallocates paged KV cache blocks via `BlockManager`
- **Batching** — prepares separate prefill and denoise batches each step
- **Postprocessing** — implements all remasking strategies (sampling + token selection). Has two paths:
  - `postprocess()` — general path, supports per-sequence sampling params and mixed strategies
  - `postprocess_unify()` — optimized batched path when all sequences share the same sampling params

### Chain Mechanism

The chain mechanism runs multiple denoising steps within a single `step()` call, avoiding scheduler overhead between steps. Adaptive chain depth:
- `chain=5` when no pending prefills (full block in one step: 4 denoise + 1 SAVING)
- `chain=4` when batch_size ≤ 8 (streaming tail)
- `chain=2` when prefills are pending (let scheduler refill slots)

Key optimizations in the chain path:
- **Context reuse**: positions/block_tables are cached and reused for pure DENOISING chain steps
- **Zero-sync fast path**: for intermediate chain steps (step < denoising_steps - 1), GPU→CPU synchronization is eliminated by predicting SAVING transitions from CPU-side state
- **Chain batch tokens**: batch tensor from postprocess is passed directly to prepare_denoise, avoiding per-sequence Python loop + torch.cat

### Sequence (`jetengine/engine/sequence.py`)

Represents a single generation request. Tracks block diffusion state: `intermediate_block_tokens` (current block being denoised as a tensor), `block_trajectory`, `block_logprobs`, `block_entropies`. The `commit_block()` method finalizes a denoised block into the sequence's token stream.

### ModelRunner (`jetengine/engine/model_runner.py`)

Handles model execution: weight loading, KV cache allocation, input preparation (prefill vs denoise have different attention patterns), and CUDA graph capture/replay.

Init order: `load_model()` → `warmup_model()` → `allocate_kv_cache()` → `_init_flashinfer()` → `capture_cudagraph()`

Key features:
- **flashinfer**: `BatchPrefillWithPagedKVCacheWrapper` with `use_cuda_graph=True`, one wrapper per batch size (1..128). Falls back to `flash_attn_with_kvcache` for larger batches.
- **CUDA graphs**: Captured for batch sizes 1 to min(max_num_seqs, 128). Graph outputs hidden states; `compute_logits` (LM head) runs outside the graph to allow selective computation.
- **Selective logits**: Only computes LM head for DENOISING sequences (not SAVING ones), saving ~25% of the largest GEMM.

### Model Registry (`jetengine/models/registry.py`)

Models register via `@register_model("name")` decorator. Currently registered:
- `sdar` — dense SDAR models (Qwen-based architecture)
- `sdar_moe` — SDAR MoE variant with fused MoE kernels
- `llada` — LLaDa diffusion models

### Remasking Strategies

Defined in `SamplingParams.remasking_strategy` and implemented in the scheduler's postprocess methods:
- `sequential` — unmask left-to-right
- `low_confidence_static` — unmask top-k by confidence (default for evaluation)
- `low_confidence_dynamic` — unmask above threshold, fallback to sequential
- `entropy_bounded` — unmask by cumulative entropy budget
- `random` — random selection

### Key Config Parameters

- `Config` (`jetengine/config.py`): `mask_token_id` (required), `block_length`, `kvcache_block_size` (must be multiple of 256), `enforce_eager` (disable CUDA graphs), `torch_compile`, `quantize_fp8`
- `SamplingParams` (`jetengine/sampling_params.py`): `block_length`, `denoising_steps`, `remasking_strategy`, `dynamic_threshold`, `eb_threshold`, `repetition_penalty`

### Layers (`jetengine/layers/`)

Custom tensor-parallel layers: `QKVParallelLinear`, `MergedColumnParallelLinear`, `RowParallelLinear`, `VocabParallelEmbedding`, `ParallelLMHead`.

Optimized operations:
- `RMSNorm` — `@torch.compile` with fused residual add
- `SiluAndMul` — `@torch.compile` with liger-kernel
- Attention — flashinfer for denoise (paged KV, CUDA graph compatible), Triton sparse attention for prefill

### Kernels (`jetengine/kernels/triton/`)

Custom Triton kernels for block prefill attention (`sparse_attn_varlen`) and `store_kvcache_kernel`. Also includes fused MoE kernel.

## Evaluation

### MATH-500 End-to-End Evaluation

The primary correctness and performance benchmark. Uses `open-r1/scripts/evaluate.py` with the SDAR-4B-Chat model.

**Config**: `open-r1/recipes/evaluation/eval_SDAR-4B_math.yaml`

**Launch command** (single GPU):
```bash
cd /mnt/shared-storage-user/liudawei/home/open-r1
conda activate /mnt/shared-storage-user/liudawei/envs/open-r1
CUDA_VISIBLE_DEVICES=0 accelerate launch --multi_gpu scripts/evaluate.py \
    --config recipes/evaluation/eval_SDAR-4B_math.yaml \
    --dataset_name /mnt/shared-storage-user/liudawei/MyDatasets/MATH-500/ \
    --num_generations 4 \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_active 128 \
    --pass_k 1 4 \
    --output_dataset_name <test_name>
```

**Important notes**:
- Must use `accelerate launch --multi_gpu` (not `--num_processes=1`) — it initializes the distributed process group required by JetEngine
- `--pass_k` values must be ≤ `--num_generations` (e.g. `--pass_k 1 4` requires `--num_generations` ≥ 4)
- This is a Mode 2 (streaming) workload: 500 × 4 = 2000 sequences through max_active=128 slots
- The eval config in the yaml has `max_num_seqs: 512` (max capacity), but `max_active` controls actual concurrency

**Results location**: `/mnt/shared-storage-user/liudawei/work_dirs_ckpt/grpo_checkpoints/open-r1-eval/<test_name>/results.json`

**Quality threshold**: pass@1 ≥ 0.678 (baseline). Current best: pass@1 ≈ 0.705 at temp=0.6/top_p=0.95.

### Throughput Profiling

#### Mode 1: Ideal Decode (pure denoise, no prefill interleaving)

```bash
cd /mnt/shared-storage-user/liudawei/home/JetEngine
conda activate /mnt/shared-storage-user/liudawei/envs/open-r1

# Quick throughput measurement
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tests/bench_throughput.py

# Detailed profiling (monkey-patched step components)
CUDA_VISIBLE_DEVICES=0 python /tmp/profile_opt20.py  # or write inline script
```

Measures peak throughput with homogeneous prompts. All sequences start together, denoise together. Chain=5 (full block per step). Typical results on H200: 4114 tok/s at bs=128.

#### Mode 2: Streaming Decode (prefill + denoise interleaving)

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --multi_gpu tests/profile_streaming.py \
    --model /mnt/shared-storage-user/liudawei/Models/SDAR/SDAR-4B-Chat \
    --max_active 128 --num_prompts 256 --mask_id 151669
```

Measures throughput with diverse prompts streaming through the engine. Shows prefill/denoise interleaving, batch fragmentation, and chain hit rates. Typical chain hit rate: ~59% with diverse prompts (vs 99%+ with homogeneous).

### Performance Profile (H200, SDAR-4B-Chat, bs=128)

Component breakdown (from profiling with torch.cuda.synchronize barriers):

| Component | Time | % of Wall | Calls | Per-call |
|-----------|------|-----------|-------|----------|
| denoise_fwd | 2.655s | 58.8% | 167 | 15.9ms |
| postprocess | 0.649s | 14.4% | 168 | 3.9ms |
| prefill_fwd | 1.170s | 25.9% | 1 | 1170ms |
| schedule | 0.003s | 0.1% | 35 | 0.09ms |

Note: Profiling sync overhead inflates total time (4.5s vs 3.3s unprofiled). Proportions are informative, not absolute.

## Important Patterns

- Block diffusion generates tokens in fixed-size blocks (`block_length=4`), each block goes through `denoising_steps=4` steps before being committed.
- Each denoising step transfers a fixed number of tokens from masked to unmasked (1 per step for block_length=4, denoising_steps=4).
- The SAVING state runs the forward pass to write final token representations into KV cache — **skipping SAVING forward corrupts KV cache** (discovered: pass@1 dropped from 0.68 → 0.14).
- CUDA graphs are captured for batch sizes 1 to 128. Hidden states output from graph, LM head runs outside for selective logits.
- `consistent_sampling_params` flag enables batched postprocessing. Set automatically when all sequences share the same `SamplingParams`.
- The `build/` directory contains stale copies — always work in the top-level `jetengine/` package.

## Optimization History

| Opt | Description | pass@1 | tok/s (bs=128) |
|-----|-------------|--------|----------------|
| baseline | Original | 0.683 | ~1500 |
| opt13 | chain across SAVING (100% chain hit) | 0.694 | - |
| opt17 | CUDA graph 1-128, selective logits | - | 2677 (bs=64) |
| opt18 | flashinfer paged attention | - | 4109 |
| opt19 | bs=128 verified | 0.708 | 4109 |
| opt20 | GPU sync elimination, chain batch tokens | 0.705 | 4114 |
| opt21 | Sparse sampling (only masked positions) | 0.722 | 4180 |
| opt22 | Lazy entropy (skip in chain intermediate) | 0.709 | 4242 |

### Ruled-Out Optimizations

- **FP8 weight quantization**: 35% per-layer relative error compounds over 36 layers → NaN. FP8 e4m3fn has only 3 mantissa bits.
- **torch.compile on decoder layers**: Incompatible with CUDA graph capture — attention layer's `get_context()` dynamic dispatch causes graph breaks.
- **denoising_steps < 4**: Quality drops catastrophically (0.683 → 0.441 with steps=3).
- **Skip SAVING forward pass**: Corrupts KV cache (pass@1 → 0.14).
- **FP8 dynamic input quantization**: Overhead > GEMM savings for M ≤ 256.

### Remaining Optimization Avenues

- **Fused Triton postprocess kernel**: Fuse softmax+sampling+entropy into a single kernel for masked positions
- **SmoothQuant FP8**: Per-channel weight + per-tensor activation (needs calibration)
- **INT4 GPTQ/AWQ**: Major quality risk, needs calibration
- **Tensor parallelism**: Near-linear scaling for multi-GPU
- **Larger block_length (8)**: 2x tokens/step but needs model retraining

## Development Workflow

### Optimization Cycle

1. **Implement** the optimization (new kernel, algorithm change, etc.)
2. **Unit test** — every new operator or algorithm must have a minimal standalone test verifying correctness against a reference implementation
3. **Throughput benchmark** — `tests/bench_throughput.py` for Mode 1
4. **End-to-end eval** — MATH-500 pass@1 ≥ 0.678 (Mode 2, streaming)
5. **Git commit** on success — each successful speedup gets its own commit
6. **Git revert** on failure — revert to last good commit and try a different approach

### Git Discipline

- Each optimization that passes both unit tests and end-to-end eval gets committed
- Commit messages: `opt<N>: <description>` (e.g. `opt20: GPU sync elimination in chain fast path`)
- If an optimization breaks quality or doesn't improve throughput, `git revert` to the last good state before continuing

### Testing Requirements

- **New Triton kernels**: must have a test comparing output against a pure PyTorch reference implementation, checking both correctness (max relative error < threshold) and edge cases
- **New sampling algorithms** (e.g. sparse sampling): must have a test verifying the output distribution matches the dense reference, and that token selection is identical
- **Postprocess changes**: must verify against `postprocess()` (the general path) as ground truth
- Unit tests go in `tests/test_*.py`, benchmarks in `tests/bench_*.py`

## Test & Profiling Infrastructure

| File | Purpose |
|------|---------|
| `tests/bench_throughput.py` | Mode 1 throughput benchmark (ideal decode) |
| `tests/profile_streaming.py` | Mode 2 profiler (streaming, diverse prompts) |
| `tests/profile_realistic.py` | Mode 1 profiler (homogeneous prompts, detailed) |
| `tests/test_correctness.py` | Unit tests for entropy, commit_block, postprocess |
| `tests/bench_fp8_static.py` | FP8 vs BF16 GEMM benchmark |
| `tests/test_fp8_static.py` | FP8 linear layer correctness test |
| `tests/bench_compile.py` | torch.compile A/B benchmark |
| `tests/test_sparse_sampling.py` | Sparse sampling correctness + benchmark |

## Files Modified (from nanovllm baseline)

- `jetengine/engine/scheduler.py` — postprocess_unify, chain_state, zero-sync path, try_allocate_chain_blocks
- `jetengine/engine/model_runner.py` — flashinfer init, CUDA graph capture, selective logits, chain batch tokens
- `jetengine/engine/sequence.py` — _commit_block_from_cpu, tensor-based block state
- `jetengine/engine/llm_engine.py` — chain mechanism, adaptive chain depth, generate_streaming
- `jetengine/layers/linear.py` — FP8 quantize_to_fp8 method, _fp8_linear function
- `jetengine/layers/attention.py` — flashinfer path in BlockAttention.forward
- `jetengine/config.py` — quantize_fp8, torch_compile flags
