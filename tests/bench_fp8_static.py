"""
Benchmark FP8 static GEMM vs BF16 for all SDAR-4B layer shapes.
Tests: pure GEMM (no dynamic quantization), with static per-tensor scales.

Run: CUDA_VISIBLE_DEVICES=0 python tests/bench_fp8_static.py
"""
import torch
import time


def bench_gemm(M, K, N, n_iter=2000):
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    a_fp8 = a_bf16.to(torch.float8_e4m3fn)
    b_fp8 = b_bf16.to(torch.float8_e4m3fn)
    sa = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    sb = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    for _ in range(100):
        torch.mm(a_bf16, b_bf16.t())
        torch._scaled_mm(a_fp8, b_fp8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        torch.mm(a_bf16, b_bf16.t())
    torch.cuda.synchronize()
    bf16_ms = (time.perf_counter() - t0) / n_iter * 1000

    t0 = time.perf_counter()
    for _ in range(n_iter):
        torch._scaled_mm(a_fp8, b_fp8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    fp8_ms = (time.perf_counter() - t0) / n_iter * 1000

    return bf16_ms, fp8_ms


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    for M in [256, 512]:
        bs = M // 4
        print(f"\n{'='*60}")
        print(f"M={M} (bs={bs}, block_len=4)")
        print(f"{'Layer':<18} {'BF16':>10} {'FP8':>10} {'Speedup':>10}")
        print("-" * 50)

        shapes = [
            (M, 2560, 3584, "QKV proj"),
            (M, 2560, 2560, "O proj"),
            (M, 2560, 13824, "Gate+Up proj"),
            (M, 6912, 2560, "Down proj"),
            (M, 2560, 151936, "LM Head"),
        ]

        total_bf16 = total_fp8 = 0
        for m, k, n, name in shapes:
            bf16, fp8 = bench_gemm(m, k, n)
            mul = 36 if name != "LM Head" else 1
            total_bf16 += bf16 * mul
            total_fp8 += fp8 * mul
            print(f"{name:<18} {bf16:.4f}ms {fp8:.4f}ms {bf16/fp8:.2f}x")

        print(f"\nTotal (36 layers + LM Head):")
        print(f"  BF16: {total_bf16:.3f}ms")
        print(f"  FP8:  {total_fp8:.3f}ms")
        print(f"  Speedup: {total_bf16/total_fp8:.2f}x")


def bench_cast_overhead():
    """Benchmark the overhead of bf16->fp8 cast for activations."""
    print(f"\n{'='*60}")
    print("Activation cast overhead (bf16 -> fp8 e4m3fn)")
    print(f"{'Shape':<20} {'Cast':>10} {'GEMM bf16':>12} {'Cast+GEMM fp8':>15} {'Net speedup':>12}")
    print("-" * 72)

    n_iter = 2000
    shapes = [
        (512, 2560, "bs128 hidden"),
        (512, 6912, "bs128 intermediate"),
        (256, 2560, "bs64 hidden"),
    ]
    for M, K, name in shapes:
        a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        N = 13824  # Gate+Up proj as representative
        b_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        b_fp8 = b_bf16.to(torch.float8_e4m3fn)
        sa = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        sb = torch.tensor(1.0, dtype=torch.float32, device="cuda")

        for _ in range(100):
            a_bf16.to(torch.float8_e4m3fn)
            torch.mm(a_bf16, b_bf16.t())
            torch._scaled_mm(a_bf16.to(torch.float8_e4m3fn), b_fp8.t(),
                             scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()

        # Cast only
        t0 = time.perf_counter()
        for _ in range(n_iter):
            a_bf16.to(torch.float8_e4m3fn)
        torch.cuda.synchronize()
        cast_ms = (time.perf_counter() - t0) / n_iter * 1000

        # BF16 GEMM
        t0 = time.perf_counter()
        for _ in range(n_iter):
            torch.mm(a_bf16, b_bf16.t())
        torch.cuda.synchronize()
        bf16_ms = (time.perf_counter() - t0) / n_iter * 1000

        # Cast + FP8 GEMM
        t0 = time.perf_counter()
        for _ in range(n_iter):
            a_q = a_bf16.to(torch.float8_e4m3fn)
            torch._scaled_mm(a_q, b_fp8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        fp8_ms = (time.perf_counter() - t0) / n_iter * 1000

        speedup = bf16_ms / fp8_ms
        print(f"{name:<20} {cast_ms:.4f}ms {bf16_ms:.4f}ms   {fp8_ms:.4f}ms      {speedup:.2f}x")


if __name__ == "__main__":
    main()
    bench_cast_overhead()
