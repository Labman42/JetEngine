"""
Realistic profiling of JetEngine: runs generate_streaming with max_tokens=4096
and instruments the model_runner.run method to collect timing data.

Run: CUDA_VISIBLE_DEVICES=0 accelerate launch --multi_gpu tests/profile_realistic.py \
       --model /path/to/SDAR-4B-Chat --max_active 64 --num_prompts 32
"""
import argparse
import sys
import os
from time import perf_counter
from collections import defaultdict, Counter
import functools

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jetengine.llm import LLM
from jetengine.sampling_params import SamplingParams
from jetengine.engine.sequence import RunType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max_active", type=int, default=64)
    parser.add_argument("--num_prompts", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--mask_id", type=int, default=151669)
    parser.add_argument("--gpu_mem", type=float, default=0.95)
    args = parser.parse_args()

    engine = LLM(
        args.model,
        mask_token_id=args.mask_id,
        gpu_memory_utilization=args.gpu_mem,
        max_num_seqs=args.max_active,
        block_length=4,
    )

    sp = SamplingParams(
        max_tokens=args.max_tokens,
        block_length=4,
        denoising_steps=4,
        temperature=1.0,
        topk=50,
        topp=1.0,
        remasking_strategy="low_confidence_dynamic",
        dynamic_threshold=0.9,
    )

    # Monkey-patch model_runner.run to collect timings
    timings = defaultdict(list)
    bs_counter = Counter()
    graph_hit_counter = Counter()  # {True: n, False: n}
    original_run = engine.model_runner.run

    @functools.wraps(original_run)
    def instrumented_run(seqs, run_type, **kwargs):
        bs = len(seqs)
        torch.cuda.synchronize()
        t0 = perf_counter()
        result = original_run(seqs, run_type, **kwargs)
        torch.cuda.synchronize()
        t1 = perf_counter()
        ms = (t1 - t0) * 1000

        if run_type == RunType.PREFILL:
            timings["prefill_forward"].append(ms)
        else:
            timings["denoise_forward"].append(ms)
            bs_counter[bs] += 1
            # Check if graph was used (infer from timing - graph < eager for same bs)
            # Better: check directly
            used_graph = hasattr(engine.model_runner, 'graphs') and bs in engine.model_runner.graphs
            graph_hit_counter[used_graph] += 1
        return result

    engine.model_runner.run = instrumented_run

    # Also patch postprocess
    original_pp = engine.scheduler.postprocess_unify

    @functools.wraps(original_pp)
    def instrumented_pp(seqs, logits, run_type):
        torch.cuda.synchronize()
        t0 = perf_counter()
        result = original_pp(seqs, logits, run_type)
        t1 = perf_counter()
        ms = (t1 - t0) * 1000
        if run_type == RunType.DENOISE:
            timings["denoise_postprocess"].append(ms)
        elif run_type == RunType.PREFILL:
            timings["prefill_postprocess"].append(ms)
        return result

    engine.scheduler.postprocess_unify = instrumented_pp

    # Also patch schedule
    original_schedule = engine.scheduler.schedule

    @functools.wraps(original_schedule)
    def instrumented_schedule():
        t0 = perf_counter()
        result = original_schedule()
        t1 = perf_counter()
        timings["schedule"].append((t1 - t0) * 1000)
        return result

    engine.scheduler.schedule = instrumented_schedule

    # Run
    prompts = ["What is the solution to x^2 - 5x + 6 = 0? Show your work step by step."] * args.num_prompts
    print(f"[profile] Running generate_streaming: {args.num_prompts} prompts, max_tokens={args.max_tokens}, max_active={args.max_active}")
    t_start = perf_counter()
    outputs = engine.generate_streaming(prompts, sp, max_active=args.max_active, use_tqdm=True)
    t_total = perf_counter() - t_start

    # Print results
    import numpy as np
    print("\n" + "=" * 80)
    print(f"Total wall time: {t_total:.1f}s")
    print(f"{'Phase':<30} {'Count':>6} {'Mean(ms)':>10} {'Median(ms)':>10} {'P95(ms)':>10} {'Total(ms)':>10} {'% of total':>10}")
    print("=" * 80)

    total_ms = t_total * 1000
    for name in ["schedule", "prefill_forward", "prefill_postprocess", "denoise_forward", "denoise_postprocess"]:
        vals = timings.get(name, [])
        if not vals:
            continue
        arr = np.array(vals)
        pct = arr.sum() / total_ms * 100
        print(
            f"{name:<30} {len(arr):>6} {arr.mean():>10.3f} {np.median(arr):>10.3f} "
            f"{np.percentile(arr, 95):>10.3f} {arr.sum():>10.1f} {pct:>9.1f}%"
        )

    # Batch size distribution
    print(f"\nDenoise batch size distribution:")
    for bs in sorted(bs_counter.keys()):
        count = bs_counter[bs]
        print(f"  bs={bs:>3}: {count:>6} steps ({count/sum(bs_counter.values())*100:.1f}%)")

    total_graph = graph_hit_counter.get(True, 0)
    total_no_graph = graph_hit_counter.get(False, 0)
    total_steps = total_graph + total_no_graph
    print(f"\nCUDA graph hit rate: {total_graph}/{total_steps} = {total_graph/max(total_steps,1)*100:.1f}%")

    # Chain statistics
    attempts = getattr(engine, '_chain_attempts', 0)
    successes = getattr(engine, '_chain_successes', 0)
    rate = successes / max(attempts, 1) * 100
    print(f"Chain hit rate: {successes}/{attempts} = {rate:.1f}%")


if __name__ == "__main__":
    main()
