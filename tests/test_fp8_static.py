"""Quick test: FP8 static quantization linear layer correctness.
Run: CUDA_VISIBLE_DEVICES=0 python tests/test_fp8_static.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
if not dist.is_initialized():
    dist.init_process_group("nccl")

from jetengine.layers.linear import ColumnParallelLinear, RowParallelLinear, MergedColumnParallelLinear


def test_fp8_linear():
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    pg = dist.group.WORLD

    # Test ColumnParallelLinear (QKV proj shape)
    layer = ColumnParallelLinear(2560, 3584, pg)
    # Fill with reasonable weights
    layer.weight.data.normal_(0, 0.02)

    x = torch.randn(512, 2560, dtype=torch.bfloat16, device="cuda")
    x = x * 2  # Simulate post-RMSNorm activation (std~2)

    with torch.inference_mode():
        y_bf16 = layer(x)

    # Quantize to FP8
    layer.quantize_to_fp8()
    print(f"Weight dtype: {layer.weight.dtype}, scale: {layer.weight_scale.item():.6f}")
    print(f"Input scale: {layer.input_scale.item():.1f}")

    with torch.inference_mode():
        y_fp8 = layer(x)

    diff = (y_bf16 - y_fp8).abs()
    rel_diff = diff / (y_bf16.abs() + 1e-6)
    print(f"\nColumnParallel (2560 -> 3584):")
    print(f"  BF16 output: mean={y_bf16.mean():.4f}, std={y_bf16.std():.4f}")
    print(f"  FP8 output:  mean={y_fp8.mean():.4f}, std={y_fp8.std():.4f}")
    print(f"  Abs diff:    mean={diff.mean():.6f}, max={diff.max():.6f}")
    print(f"  Rel diff:    mean={rel_diff.mean():.6f}, max={rel_diff.max():.6f}")
    print(f"  Has NaN: {y_fp8.isnan().any()}, Inf: {y_fp8.isinf().any()}")

    # Test Gate+Up proj (larger)
    layer2 = MergedColumnParallelLinear(2560, [6912, 6912], pg)
    layer2.weight.data.normal_(0, 0.02)
    with torch.inference_mode():
        y2_bf16 = layer2(x)
    layer2.quantize_to_fp8()
    with torch.inference_mode():
        y2_fp8 = layer2(x)
    diff2 = (y2_bf16 - y2_fp8).abs()
    rel2 = diff2 / (y2_bf16.abs() + 1e-6)
    print(f"\nMergedColumn (2560 -> 13824):")
    print(f"  Abs diff: mean={diff2.mean():.6f}, max={diff2.max():.6f}")
    print(f"  Rel diff: mean={rel2.mean():.6f}")
    print(f"  Has NaN: {y2_fp8.isnan().any()}, Inf: {y2_fp8.isinf().any()}")

    # Test RowParallel (down_proj)
    layer3 = RowParallelLinear(6912, 2560, pg)
    layer3.weight.data.normal_(0, 0.02)
    # MLP intermediate has larger values after SiLU
    x3 = torch.randn(512, 6912, dtype=torch.bfloat16, device="cuda") * 5
    with torch.inference_mode():
        y3_bf16 = layer3(x3)
    layer3.quantize_to_fp8()
    with torch.inference_mode():
        y3_fp8 = layer3(x3)
    diff3 = (y3_bf16 - y3_fp8).abs()
    rel3 = diff3 / (y3_bf16.abs() + 1e-6)
    print(f"\nRowParallel (6912 -> 2560, input std=5):")
    print(f"  Abs diff: mean={diff3.mean():.6f}, max={diff3.max():.6f}")
    print(f"  Rel diff: mean={rel3.mean():.6f}")
    print(f"  Has NaN: {y3_fp8.isnan().any()}, Inf: {y3_fp8.isinf().any()}")

    # Test with extreme inputs (check clipping)
    x_extreme = torch.randn(512, 2560, dtype=torch.bfloat16, device="cuda") * 50
    layer4 = ColumnParallelLinear(2560, 3584, pg)
    layer4.weight.data.normal_(0, 0.02)
    with torch.inference_mode():
        y4_bf16 = layer4(x_extreme)
    layer4.quantize_to_fp8()
    with torch.inference_mode():
        y4_fp8 = layer4(x_extreme)
    diff4 = (y4_bf16 - y4_fp8).abs()
    rel4 = diff4 / (y4_bf16.abs() + 1e-6)
    print(f"\nExtreme input (std=50, scale clips at 15*50=750 > 448):")
    print(f"  BF16: mean={y4_bf16.mean():.4f}, std={y4_bf16.std():.4f}")
    print(f"  FP8:  mean={y4_fp8.mean():.4f}, std={y4_fp8.std():.4f}")
    print(f"  Rel diff: mean={rel4.mean():.6f}")
    print(f"  Has NaN: {y4_fp8.isnan().any()}, Inf: {y4_fp8.isinf().any()}")

    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)


if __name__ == "__main__":
    test_fp8_linear()
