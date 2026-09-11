"""Compare full-page load specialization against the masked implementation.

Requires only PyTorch, Triton, Git, and a CUDA GPU; no model checkpoint.
Run from the repository root:
    python tests/test_paged_attention_full_pages.py --benchmark --out /tmp/paged-check
The default baseline is pinned before the optimization, even after this test is committed.
"""
import argparse
import hashlib
import importlib.util
import json
import itertools
from pathlib import Path
import statistics
import subprocess
import tempfile

import torch
from triton.runtime.errors import OutOfResources

BASELINE_REF = "bf8cb31832205c9a8afbf4a1489128b7cb92b866"
KERNEL_PATH = "jetengine/kernels/triton/attention/fused_page_attention_v3.py"
# prefixes, active block, M, N, warps, stages, page size, head dimension
CASES = [
    ((0,), 4, 16, 64, 4, 2, 256, 128),
    ((1,), 32, 32, 64, 4, 2, 256, 128),
    ((255,), 32, 16, 128, 4, 3, 256, 128),
    ((256,), 32, 32, 64, 4, 2, 256, 128),
    ((257,), 64, 32, 256, 8, 3, 256, 128),
    ((8224,), 32, 32, 64, 4, 2, 256, 128),
    ((0, 256), 4, 16, 64, 4, 2, 256, 128),
    ((1, 257), 32, 32, 128, 8, 2, 256, 128),
    ((255, 256), 64, 16, 256, 4, 2, 256, 128),
    ((256, 8224), 32, 32, 128, 4, 3, 256, 128),
    ((257, 0), 64, 32, 64, 8, 2, 256, 128),
    ((256,), 32, 32, 256, 4, 2, 256, 128),
    ((256, 257), 32, 16, 256, 4, 2, 128, 128),
    ((256, 257), 32, 32, 64, 4, 2, 256, 96),
    ((255, 8224), 4, 16, 128, 8, 3, 256, 128),
    ((1, 256), 64, 32, 256, 8, 3, 256, 128),
]


def load_kernel(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fused_kv_cache_attention_kernel.fn


def make_inputs(case, seed, q_heads=8, kv_heads=2):
    prefixes, block, m, n, warps, stages, page, dim = case
    torch.manual_seed(seed)
    batch = len(prefixes)
    tokens = batch * block
    # Native-style fused QKV views exercise non-contiguous sequence strides.
    packed = torch.randn(tokens, (q_heads + 2 * kv_heads) * dim,
                         device="cuda", dtype=torch.bfloat16)
    q = packed[:, :q_heads * dim].view(tokens, q_heads, dim)
    k = packed[:, q_heads * dim:(q_heads + kv_heads) * dim].view(tokens, kv_heads, dim)
    v = packed[:, (q_heads + kv_heads) * dim:].view(tokens, kv_heads, dim)
    counts = [(p + page - 1) // page for p in prefixes]
    pages = sum(counts) + 3
    pk = torch.randn(pages, page, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
    pv = torch.randn_like(pk)
    # Each sequence owns distinct, deliberately shuffled physical pages.
    physical = torch.randperm(pages).tolist()
    table = torch.full((batch, max(max(counts), 1)), -1, dtype=torch.int32)
    cursor = 0
    for seq, count in enumerate(counts):
        table[seq, :count] = torch.tensor(physical[cursor:cursor + count], dtype=torch.int32)
        cursor += count
    cuq = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * block
    cuk = torch.tensor([0] + list(itertools.accumulate(prefixes)),
                       dtype=torch.int32, device="cuda")
    return dict(q=q, k=k, v=v, pk=pk, pv=pv, table=table.cuda(), cuq=cuq, cuk=cuk,
                prefixes=prefixes, block=block, page=page, dim=dim, q_heads=q_heads,
                kv_heads=kv_heads, batch=batch, m=m, n=n, warps=warps, stages=stages)


def launch(kernel, x, out):
    args = []
    for name in ("q", "k", "v", "pk", "pv", "table"):
        tensor = x[name]
        args.extend([tensor, *tensor.stride()])
    args.extend([out, *out.stride(), x['cuq'], x['cuk'], x['batch'], x['block'],
                 x['page'], x['q_heads'], x['kv_heads'], x['dim'], x['dim'] ** -0.5])
    return kernel[(x['batch'], x['q_heads'])](
        *args, BLOCK_M=x['m'], BLOCK_N=x['n'], BLOCK_DMODEL=128,
        num_warps=x['warps'], num_stages=x['stages'])


def reference(x):
    outputs = []
    group = x['q_heads'] // x['kv_heads']
    for seq, prefix in enumerate(x['prefixes']):
        start, end = seq * x['block'], (seq + 1) * x['block']
        count = (prefix + x['page'] - 1) // x['page']
        pages = x['table'][seq, :count].long()
        prefix_k = x['pk'].index_select(0, pages).flatten(0, 1)[:prefix]
        prefix_v = x['pv'].index_select(0, pages).flatten(0, 1)[:prefix]
        k = torch.cat((prefix_k, x['k'][start:end])).float().repeat_interleave(group, 1)
        v = torch.cat((prefix_v, x['v'][start:end])).float().repeat_interleave(group, 1)
        q = x['q'][start:end].float()
        p = torch.einsum('qhd,khd->hqk', q, k).mul(x['dim'] ** -0.5).softmax(-1)
        outputs.append(torch.einsum('hqk,khd->qhd', p, v))
    return torch.cat(outputs)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False  # FP32 reference only; Triton stays unchanged.
    repo = args.repo_dir.resolve()
    baseline_source = subprocess.check_output(
        ['git', 'show', args.baseline_ref + ':' + KERNEL_PATH], cwd=repo)
    current_path = repo / KERNEL_PATH
    current_source = current_path.read_bytes()
    result = dict(baseline_ref=args.baseline_ref,
                  baseline_sha256=hashlib.sha256(baseline_source).hexdigest(),
                  current_sha256=hashlib.sha256(current_source).hexdigest(),
                  torch=torch.__version__, device=torch.cuda.get_device_name(), cases=[])
    with tempfile.TemporaryDirectory(prefix='paged-attention-reference-') as tmp:
        old_path = Path(tmp) / 'masked_reference.py'
        old_path.write_bytes(baseline_source)
        old = load_kernel(old_path, 'masked_reference')
        new = load_kernel(current_path, 'current_paged_attention')
        for index, case in enumerate(CASES):
            x = make_inputs(case, 1234 + index)
            original, actual = torch.empty_like(x['q']), torch.empty_like(x['q'])
            before_k, before_v = x['pk'].clone(), x['pv'].clone()
            try:
                launch(old, x, original)
            except OutOfResources as error:
                result['cases'].append(dict(case=case, skipped=str(error),
                                            reason='Baseline configuration exceeds device resources'))
                print('SKIP', index, case, str(error), flush=True)
                continue
            # A resource failure introduced only by the modified kernel is a test failure.
            launch(new, x, actual)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
            assert bool(torch.isfinite(actual).all())
            assert torch.equal(before_k, x['pk']) and torch.equal(before_v, x['pv'])
            ref = reference(x)
            delta = actual.float() - ref
            relative_l2 = float(delta.norm() / ref.norm().clamp_min(1e-12))
            max_abs = float(delta.abs().max())
            assert relative_l2 < 0.01 and max_abs < 0.03, (case, relative_l2, max_abs)
            row = dict(case=case, exact_to_masked=True, cache_unchanged=True,
                       reference_relative_l2=relative_l2, reference_max_abs=max_abs)
            result['cases'].append(row)
            print('PASS', index, case, flush=True)
        if args.benchmark:
            result['benchmark'] = {}
            for prefix in (8192, 257):
                x = make_inputs(((prefix,), 32, 32, 64, 4, 2, 256, 128), 1234, 32, 8)
                outputs = {name: torch.empty_like(x['q']) for name in ('masked', 'full_pages')}
                kernels = dict(masked=old, full_pages=new)
                resources = {}
                for name, kernel in kernels.items():
                    for _ in range(2):
                        compiled = launch(kernel, x, outputs[name])
                    resources[name] = dict(registers=compiled.n_regs,
                                           local_bytes=compiled.n_spills * 4,
                                           shared_bytes=compiled.metadata.shared)
                torch.cuda.synchronize()
                torch.testing.assert_close(outputs['masked'], outputs['full_pages'], rtol=0, atol=0)
                samples = []
                for repeat in range(20):
                    order = ['masked', 'full_pages'] if repeat % 2 == 0 else ['full_pages', 'masked']
                    for name in order:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        launch(kernels[name], x, outputs[name])
                        end.record()
                        end.synchronize()
                        samples.append(dict(repeat=repeat, arm=name, ms=start.elapsed_time(end)))
                medians = {name: statistics.median(s['ms'] for s in samples if s['arm'] == name)
                           for name in kernels}
                result['benchmark'][str(prefix)] = dict(median_ms=medians, samples=samples, resources=resources,
                                           scope='Full kernel; preallocated outputs; eager CUDA events; clocks not locked')
                print('BENCHMARK', prefix, medians, flush=True)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    passed = sum('skipped' not in case for case in result['cases'])
    assert passed, 'No supported baseline configurations were tested'
    print(passed, 'cases passed;', len(result['cases']) - passed, 'baseline-resource skips.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', default=BASELINE_REF)
    parser.add_argument('--repo-dir', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--out', type=Path)
    run(parser.parse_args())
