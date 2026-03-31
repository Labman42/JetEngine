"""
Streaming fill profiler: simulates generate_streaming with diverse prompts
to capture prefill interleaved with denoise — the real-world bottleneck pattern.

Run: CUDA_VISIBLE_DEVICES=0 accelerate launch --multi_gpu tests/profile_streaming.py \
       --model /path/to/SDAR-4B-Chat --max_active 64 --num_prompts 128
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
from jetengine.engine.sequence import RunType, SequenceStatus


# Diverse prompts to simulate real workloads (different lengths → staggered finishing)
DIVERSE_PROMPTS = [
    "What is 2+2?",
    "Explain the theory of relativity in detail, covering both special and general relativity.",
    "Write a Python function to compute the Fibonacci sequence using memoization.",
    "What is the capital of France?",
    "Prove that the square root of 2 is irrational.",
    "Summarize the plot of Romeo and Juliet.",
    "What is the solution to x^2 - 5x + 6 = 0? Show your work step by step.",
    "Describe the process of photosynthesis in detail.",
    "Write a haiku about the ocean.",
    "Explain how neural networks learn through backpropagation.",
    "What are the prime factors of 360?",
    "Compare and contrast TCP and UDP protocols.",
    "Write a recursive function to compute factorial in C++.",
    "What is the derivative of sin(x)*cos(x)?",
    "Explain the difference between supervised and unsupervised learning.",
    "Solve the differential equation dy/dx = 2xy with y(0)=1.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max_active", type=int, default=64)
    parser.add_argument("--num_prompts", type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--mask_id", type=int, default=151669)
    parser.add_argument("--gpu_mem", type=float, default=0.95)
    parser.add_argument("--denoising_steps", type=int, default=4)
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
        denoising_steps=args.denoising_steps,
        temperature=1.0,
        topk=50,
        topp=1.0,
        remasking_strategy="low_confidence_dynamic",
        dynamic_threshold=0.9,
    )

    # Monkey-patch model_runner.run to collect timings
    timings = defaultdict(list)
    bs_counter = Counter()
    graph_hit_counter = Counter()
    step_timeline = []  # [(wall_time, event_type, duration_ms, batch_size)]
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
            total_tokens = sum(len(s) for s in seqs)
            step_timeline.append((t0, "prefill", ms, bs, total_tokens))
        else:
            timings["denoise_forward"].append(ms)
            bs_counter[bs] += 1
            used_graph = hasattr(engine.model_runner, 'graphs') and bs in engine.model_runner.graphs
            graph_hit_counter[used_graph] += 1
            step_timeline.append((t0, "denoise", ms, bs, bs * 4))
        return result

    engine.model_runner.run = instrumented_run

    # Patch postprocess
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

    # Patch schedule
    original_schedule = engine.scheduler.schedule

    @functools.wraps(original_schedule)
    def instrumented_schedule():
        t0 = perf_counter()
        result = original_schedule()
        t1 = perf_counter()
        timings["schedule"].append((t1 - t0) * 1000)
        return result

    engine.scheduler.schedule = instrumented_schedule

    # Use diverse prompts
    prompts = [DIVERSE_PROMPTS[i % len(DIVERSE_PROMPTS)] for i in range(args.num_prompts)]

    print(f"[profile_streaming] {args.num_prompts} diverse prompts, "
          f"max_tokens={args.max_tokens}, max_active={args.max_active}, "
          f"denoising_steps={args.denoising_steps}")
    t_start = perf_counter()
    outputs = engine.generate_streaming(prompts, sp, max_active=args.max_active, use_tqdm=True)
    t_total = perf_counter() - t_start

    # Print results
    import numpy as np
    print("\n" + "=" * 90)
    print(f"Total wall time: {t_total:.1f}s")
    print(f"{'Phase':<30} {'Count':>6} {'Mean(ms)':>10} {'Median(ms)':>10} "
          f"{'P95(ms)':>10} {'Total(ms)':>10} {'% of total':>10}")
    print("=" * 90)

    total_ms = t_total * 1000
    for name in ["schedule", "prefill_forward", "prefill_postprocess",
                 "denoise_forward", "denoise_postprocess"]:
        vals = timings.get(name, [])
        if not vals:
            continue
        arr = np.array(vals)
        pct = arr.sum() / total_ms * 100
        print(
            f"{name:<30} {len(arr):>6} {arr.mean():>10.3f} {np.median(arr):>10.3f} "
            f"{np.percentile(arr, 95):>10.3f} {arr.sum():>10.1f} {pct:>9.1f}%"
        )

    # Prefill analysis
    prefill_count = len(timings.get("prefill_forward", []))
    denoise_count = len(timings.get("denoise_forward", []))
    print(f"\nPrefill calls: {prefill_count}, Denoise calls: {denoise_count}")
    if prefill_count > 0:
        prefill_total = sum(timings["prefill_forward"])
        denoise_total = sum(timings["denoise_forward"])
        print(f"Prefill total: {prefill_total:.1f}ms ({prefill_total/total_ms*100:.1f}%)")
        print(f"Denoise total: {denoise_total:.1f}ms ({denoise_total/total_ms*100:.1f}%)")

    # Batch size distribution
    print(f"\nDenoise batch size distribution:")
    for bs in sorted(bs_counter.keys()):
        count = bs_counter[bs]
        print(f"  bs={bs:>3}: {count:>6} steps ({count/sum(bs_counter.values())*100:.1f}%)")

    total_graph = graph_hit_counter.get(True, 0)
    total_no_graph = graph_hit_counter.get(False, 0)
    total_steps = total_graph + total_no_graph
    print(f"\nCUDA graph hit rate: {total_graph}/{total_steps} = "
          f"{total_graph/max(total_steps,1)*100:.1f}%")

    # Chain statistics
    attempts = getattr(engine, '_chain_attempts', 0)
    successes = getattr(engine, '_chain_successes', 0)
    rate = successes / max(attempts, 1) * 100
    print(f"Chain hit rate: {successes}/{attempts} = {rate:.1f}%")

    # Timeline analysis: how often do prefill and denoise interleave?
    if step_timeline:
        interleave_count = 0
        for i in range(1, len(step_timeline)):
            if step_timeline[i][1] != step_timeline[i-1][1]:
                interleave_count += 1
        print(f"\nTimeline interleave count: {interleave_count} transitions "
              f"(out of {len(step_timeline)} events)")

        # Prefill token count distribution
        prefill_tokens = [e[4] for e in step_timeline if e[1] == "prefill"]
        if prefill_tokens:
            arr = np.array(prefill_tokens)
            print(f"Prefill tokens/call: mean={arr.mean():.0f}, "
                  f"median={np.median(arr):.0f}, max={arr.max():.0f}")

    # Throughput
    total_output_tokens = sum(len(o["token_ids"]) for o in outputs)
    total_prompt_tokens = sum(len(engine.tokenizer.encode(p)) for p in prompts)
    total_completion_tokens = total_output_tokens - total_prompt_tokens
    print(f"\nTotal completion tokens: {total_completion_tokens:,}")
    print(f"Throughput: {total_completion_tokens / t_total:.0f} tok/s")


if __name__ == "__main__":
    main()
