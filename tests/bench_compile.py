"""
A/B benchmark: torch.compile decoder layers.
Tests if compiled kernels captured in CUDA graphs improve throughput.

Run: CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tests/bench_compile.py
"""
import torch
import sys
sys.path.insert(0, '.')

from jetengine.llm import LLM
from jetengine.sampling_params import SamplingParams
from time import perf_counter


def main():
    path = "/mnt/shared-storage-user/liudawei/Models/SDAR/SDAR-4B-Chat"

    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=4096,
        block_length=4,
        denoising_steps=4,
        remasking_strategy="low_confidence_static",
        topk=0,
        topp=0.95,
    )

    prompt = "Solve: what is the integral of x^2 from 0 to 1? Think step by step."
    n_seqs = 64
    n_runs = 3
    prompts = [prompt] * n_seqs

    # Run both tests sequentially in separate processes to avoid memory leaks
    # Test A: without torch.compile (baseline)
    print("=== A: Baseline (no torch.compile) ===")
    llm = LLM(
        path,
        enforce_eager=False,
        tensor_parallel_size=1,
        mask_token_id=151669,
        block_length=4,
        max_num_seqs=64,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        torch_compile=False,
    )

    # Warmup
    llm.generate([prompt] * 8, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()

    for run in range(n_runs):
        llm._chain_attempts = 0
        llm._chain_successes = 0
        torch.cuda.synchronize()
        t = perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        dt = perf_counter() - t
        gen_tokens = sum(len(o.get('token_ids', [])) - 50 for o in outputs)
        print(f"  Run {run+1}: {dt:.3f}s, {gen_tokens/dt:.0f} tok/s")

    # Aggressively free resources
    llm.free_all_resources()
    del llm
    import gc; gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"  After cleanup: {torch.cuda.memory_allocated()/1e9:.1f} GiB allocated")

    # Test B: with torch.compile
    print("\n=== B: torch.compile (decoder layers) ===")
    llm2 = LLM(
        path,
        enforce_eager=False,
        tensor_parallel_size=1,
        mask_token_id=151669,
        block_length=4,
        max_num_seqs=64,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        torch_compile=True,
    )

    # Warmup (includes compile warmup)
    print("  Warmup (includes compilation)...")
    llm2.generate([prompt] * 8, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()

    for run in range(n_runs):
        llm2._chain_attempts = 0
        llm2._chain_successes = 0
        torch.cuda.synchronize()
        t = perf_counter()
        outputs = llm2.generate(prompts, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        dt = perf_counter() - t
        gen_tokens = sum(len(o.get('token_ids', [])) - 50 for o in outputs)
        print(f"  Run {run+1}: {dt:.3f}s, {gen_tokens/dt:.0f} tok/s")


if __name__ == "__main__":
    main()
