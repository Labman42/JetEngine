"""
Simple throughput benchmark. No instrumentation overhead.
Run: CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tests/bench_throughput.py
"""
import torch
import sys
sys.path.insert(0, '.')

from jetengine.llm import LLM
from jetengine.sampling_params import SamplingParams
from time import perf_counter


def main():
    path = "/mnt/shared-storage-user/liudawei/Models/SDAR/SDAR-4B-Chat"
    llm = LLM(
        path,
        enforce_eager=False,
        tensor_parallel_size=1,
        mask_token_id=151669,
        block_length=4,
        max_num_seqs=128,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
    )

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
    n_runs = 3

    # Warmup
    print("Warmup...")
    llm.generate([prompt] * 16, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()

    for n_seqs in [64, 128]:
        prompts = [prompt] * n_seqs
        print(f"\n=== batch_size={n_seqs} ===")
        for run in range(n_runs):
            llm._chain_attempts = 0
            llm._chain_successes = 0
            llm._chain_depth_counts = {}
            torch.cuda.synchronize()
            t = perf_counter()
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            torch.cuda.synchronize()
            dt = perf_counter() - t
            gen_tokens = sum(len(o.get('token_ids', [])) - 50 for o in outputs)
            chain_rate = llm._chain_successes / max(llm._chain_attempts, 1) * 100
            depth_str = " ".join(f"d{k}={v}" for k, v in sorted(llm._chain_depth_counts.items()))
            print(f"  Run {run+1}: {dt:.3f}s, {gen_tokens/dt:.0f} tok/s, "
                  f"chain={llm._chain_successes}/{llm._chain_attempts} ({chain_rate:.0f}%), "
                  f"depths=[{depth_str}]")


if __name__ == "__main__":
    main()
