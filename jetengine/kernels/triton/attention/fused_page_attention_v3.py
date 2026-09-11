import torch
import triton
import triton.language as tl
import math
from typing import Optional
import triton.testing

# Migrated from layers.fused_page_attention_v3 (v3, autotuned streaming)

LOG2E = 1.4426950408889634

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'num_warps': 2, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'num_warps': 2, 'num_stages': 3}),
    ],
    key=['num_q_heads', 'num_kv_heads', 'head_dim', 'block_len', 'block_size']
)
@triton.jit
def fused_kv_cache_attention_kernel(
    q_ptr, q_stride_seq, q_stride_head, q_stride_dim,
    k_ptr, k_stride_seq, k_stride_head, k_stride_dim,
    v_ptr, v_stride_seq, v_stride_head, v_stride_dim,
    k_cache_ptr, k_cache_stride_block, k_cache_stride_pos, k_cache_stride_head, k_cache_stride_dim,
    v_cache_ptr, v_cache_stride_block, v_cache_stride_pos, v_cache_stride_head, v_cache_stride_dim,
    block_tables_ptr, block_tables_stride_seq, block_tables_stride_block,
    o_ptr, o_stride_seq, o_stride_head, o_stride_dim,
    cu_seqlens_q_ptr, cu_seqlens_k_ptr,
    num_seqs, block_len, block_size, num_q_heads, num_kv_heads, head_dim,
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    LOG2E: tl.constexpr = LOG2E
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_start_q = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_start_k_cache = tl.load(cu_seqlens_k_ptr + seq_idx)
    seq_end_k_cache = tl.load(cu_seqlens_k_ptr + seq_idx + 1)
    seq_len_k_cache = seq_end_k_cache - seq_start_k_cache
    kv_group_size = num_q_heads // num_kv_heads
    kv_head_idx = head_idx // kv_group_size
    d_idx = tl.arange(0, BLOCK_DMODEL)
    n_idx = tl.arange(0, BLOCK_N)
    for start_m in range(0, block_len, BLOCK_M):
        m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        q_offs = start_m + tl.arange(0, BLOCK_M)
        q_mask = q_offs < block_len
        dim_mask = d_idx[None, :] < head_dim
        q_ptrs = (
            q_ptr + (seq_start_q + q_offs[:, None]) * q_stride_seq + head_idx * q_stride_head + d_idx[None, :] * q_stride_dim
        )
        q = tl.load(q_ptrs, mask=q_mask[:, None] & dim_mask, other=0.0, cache_modifier=".ca")
        num_logical_blocks = (seq_len_k_cache + block_size - 1) // block_size
        kv_pos = 0
        for logical_block_num in range(num_logical_blocks):
            physical_block_id = tl.load(
                block_tables_ptr + seq_idx * block_tables_stride_seq + logical_block_num * block_tables_stride_block
            )
            tokens_in_block = tl.minimum(seq_len_k_cache - kv_pos, block_size)
            num_tiles = (tokens_in_block + BLOCK_N - 1) // BLOCK_N
            # Full pages: all prefix lanes are valid; partial page retains original checks.
            if (tokens_in_block == block_size) & ((block_size % BLOCK_N == 0) & (head_dim == BLOCK_DMODEL)):
                if num_tiles > 0:
                    tile_n = 0
                    actual_n = tl.minimum(tokens_in_block - tile_n, BLOCK_N)
                    k_cache_base = k_cache_ptr + physical_block_id * k_cache_stride_block + kv_head_idx * k_cache_stride_head
                    v_cache_base = v_cache_ptr + physical_block_id * v_cache_stride_block + kv_head_idx * v_cache_stride_head
                    offs_n = (tile_n + n_idx)[:, None] * k_cache_stride_pos
                    k_ptrs = k_cache_base + offs_n + d_idx[None, :] * k_cache_stride_dim
                    v_ptrs = v_cache_base + offs_n + d_idx[None, :] * v_cache_stride_dim
                    kv_mask = (n_idx[:, None] < actual_n) & dim_mask
                    k = tl.load(k_ptrs, cache_modifier=".cg")
                    v = tl.load(v_ptrs, cache_modifier=".cg")
                    for t in range(1, num_tiles):
                        tile_n_next = t * BLOCK_N
                        actual_n_next = tl.minimum(tokens_in_block - tile_n_next, BLOCK_N)
                        offs_n_next = (tile_n_next + n_idx)[:, None] * k_cache_stride_pos
                        k_ptrs_next = k_cache_base + offs_n_next + d_idx[None, :] * k_cache_stride_dim
                        v_ptrs_next = v_cache_base + offs_n_next + d_idx[None, :] * v_cache_stride_dim
                        kv_mask_next = (n_idx[:, None] < actual_n_next) & dim_mask
                        k_next = tl.load(k_ptrs_next, cache_modifier=".cg")
                        v_next = tl.load(v_ptrs_next, cache_modifier=".cg")
                        qk = tl.dot(q, tl.trans(k), allow_tf32=True)
                        qk = qk.to(tl.float32) * sm_scale
                        nmask = n_idx[None, :] < actual_n
                        qk = tl.where(nmask, qk, -float('inf'))
                        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
                        alpha = tl.exp2((m_i - m_new) * LOG2E)
                        p = tl.exp2((qk - m_new[:, None]) * LOG2E)
                        acc = acc * alpha[:, None] + tl.dot(p, v.to(p.dtype), allow_tf32=True)
                        l_i = l_i * alpha + tl.sum(p, axis=1)
                        m_i = m_new
                        k, v = k_next, v_next
                        actual_n = actual_n_next
                    qk = tl.dot(q, tl.trans(k), allow_tf32=True)
                    qk = qk.to(tl.float32) * sm_scale
                    nmask = n_idx[None, :] < actual_n
                    qk = tl.where(nmask, qk, -float('inf'))
                    m_new = tl.maximum(m_i, tl.max(qk, axis=1))
                    alpha = tl.exp2((m_i - m_new) * LOG2E)
                    p = tl.exp2((qk - m_new[:, None]) * LOG2E)
                    acc = acc * alpha[:, None] + tl.dot(p, v.to(p.dtype), allow_tf32=True)
                    l_i = l_i * alpha + tl.sum(p, axis=1)
                    m_i = m_new
            else:
                if num_tiles > 0:
                    tile_n = 0
                    actual_n = tl.minimum(tokens_in_block - tile_n, BLOCK_N)
                    k_cache_base = k_cache_ptr + physical_block_id * k_cache_stride_block + kv_head_idx * k_cache_stride_head
                    v_cache_base = v_cache_ptr + physical_block_id * v_cache_stride_block + kv_head_idx * v_cache_stride_head
                    offs_n = (tile_n + n_idx)[:, None] * k_cache_stride_pos
                    k_ptrs = k_cache_base + offs_n + d_idx[None, :] * k_cache_stride_dim
                    v_ptrs = v_cache_base + offs_n + d_idx[None, :] * v_cache_stride_dim
                    kv_mask = (n_idx[:, None] < actual_n) & dim_mask
                    k = tl.load(k_ptrs, mask=kv_mask, other=0.0, cache_modifier=".cg")
                    v = tl.load(v_ptrs, mask=kv_mask, other=0.0, cache_modifier=".cg")
                    for t in range(1, num_tiles):
                        tile_n_next = t * BLOCK_N
                        actual_n_next = tl.minimum(tokens_in_block - tile_n_next, BLOCK_N)
                        offs_n_next = (tile_n_next + n_idx)[:, None] * k_cache_stride_pos
                        k_ptrs_next = k_cache_base + offs_n_next + d_idx[None, :] * k_cache_stride_dim
                        v_ptrs_next = v_cache_base + offs_n_next + d_idx[None, :] * v_cache_stride_dim
                        kv_mask_next = (n_idx[:, None] < actual_n_next) & dim_mask
                        k_next = tl.load(k_ptrs_next, mask=kv_mask_next, other=0.0, cache_modifier=".cg")
                        v_next = tl.load(v_ptrs_next, mask=kv_mask_next, other=0.0, cache_modifier=".cg")
                        qk = tl.dot(q, tl.trans(k), allow_tf32=True)
                        qk = qk.to(tl.float32) * sm_scale
                        nmask = n_idx[None, :] < actual_n
                        qk = tl.where(nmask, qk, -float('inf'))
                        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
                        alpha = tl.exp2((m_i - m_new) * LOG2E)
                        p = tl.exp2((qk - m_new[:, None]) * LOG2E)
                        acc = acc * alpha[:, None] + tl.dot(p, v.to(p.dtype), allow_tf32=True)
                        l_i = l_i * alpha + tl.sum(p, axis=1)
                        m_i = m_new
                        k, v = k_next, v_next
                        actual_n = actual_n_next
                    qk = tl.dot(q, tl.trans(k), allow_tf32=True)
                    qk = qk.to(tl.float32) * sm_scale
                    nmask = n_idx[None, :] < actual_n
                    qk = tl.where(nmask, qk, -float('inf'))
                    m_new = tl.maximum(m_i, tl.max(qk, axis=1))
                    alpha = tl.exp2((m_i - m_new) * LOG2E)
                    p = tl.exp2((qk - m_new[:, None]) * LOG2E)
                    acc = acc * alpha[:, None] + tl.dot(p, v.to(p.dtype), allow_tf32=True)
                    l_i = l_i * alpha + tl.sum(p, axis=1)
                    m_i = m_new
            kv_pos += tokens_in_block
        num_tiles_cur = (block_len + BLOCK_N - 1) // BLOCK_N
        if num_tiles_cur > 0:
            tile_n = 0
            actual_n = tl.minimum(block_len - tile_n, BLOCK_N)
            k_abs = (seq_start_q + tile_n + n_idx[:, None]) * k_stride_seq
            v_abs = (seq_start_q + tile_n + n_idx[:, None]) * v_stride_seq
            k_ptrs = k_ptr + k_abs + kv_head_idx * k_stride_head + d_idx[None, :] * k_stride_dim
            v_ptrs = v_ptr + v_abs + kv_head_idx * v_stride_head + d_idx[None, :] * v_stride_dim
            kv_mask = (n_idx[:, None] < actual_n) & dim_mask
            k = tl.load(k_ptrs, mask=kv_mask, other=0.0, cache_modifier=".ca")
            v = tl.load(v_ptrs, mask=kv_mask, other=0.0, cache_modifier=".ca")
            for t in range(1, num_tiles_cur):
                tile_n_next = t * BLOCK_N
                actual_n_next = tl.minimum(block_len - tile_n_next, BLOCK_N)
                k_abs_next = (seq_start_q + tile_n_next + n_idx[:, None]) * k_stride_seq
                v_abs_next = (seq_start_q + tile_n_next + n_idx[:, None]) * v_stride_seq
                k_ptrs_next = k_ptr + k_abs_next + kv_head_idx * k_stride_head + d_idx[None, :] * k_stride_dim
                v_ptrs_next = v_ptr + v_abs_next + kv_head_idx * v_stride_head + d_idx[None, :] * v_stride_dim
                kv_mask_next = (n_idx[:, None] < actual_n_next) & dim_mask
                k_next = tl.load(k_ptrs_next, mask=kv_mask_next, other=0.0, cache_modifier=".ca")
                v_next = tl.load(v_ptrs_next, mask=kv_mask_next, other=0.0, cache_modifier=".ca")
                qk = tl.dot(q, tl.trans(k), allow_tf32=True)
                qk = qk.to(tl.float32) * sm_scale
                nmask = n_idx[None, :] < actual_n
                qk = tl.where(nmask, qk, -float('inf'))
                m_new = tl.maximum(m_i, tl.max(qk, axis=1))
                alpha = tl.exp2((m_i - m_new) * LOG2E)
                p = tl.exp2((qk - m_new[:, None]) * LOG2E)
                acc = acc * alpha[:, None] + tl.dot(p, v.to(p.dtype), allow_tf32=True)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                m_i = m_new
                k, v = k_next, v_next
                actual_n = actual_n_next
            qk = tl.dot(q, tl.trans(k), allow_tf32=True)
            qk = qk.to(tl.float32) * sm_scale
            nmask = n_idx[None, :] < actual_n
            qk = tl.where(nmask, qk, -float('inf'))
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp2((m_i - m_new) * LOG2E)
            p = tl.exp2((qk - m_new[:, None]) * LOG2E)
            acc = acc * alpha[:, None] + tl.dot(p, v.to(p.dtype), allow_tf32=True)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new
        l_safe = tl.where(l_i == 0, 1.0, l_i)
        acc = acc / l_safe[:, None]
        o_ptrs = (
            o_ptr + (seq_start_q + q_offs[:, None]) * o_stride_seq + head_idx * o_stride_head + d_idx[None, :] * o_stride_dim
        )
        tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=q_mask[:, None] & dim_mask)


def fused_kv_cache_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_len: int,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    q_total_tokens, num_q_heads, head_dim = q.shape
    _, num_kv_heads, _ = k.shape
    _, block_size, _, _ = k_cache.shape
    num_seqs = block_tables.shape[0]
    assert cu_seqlens_q.shape[0] == num_seqs + 1
    assert cu_seqlens_k.shape[0] == num_seqs + 1
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    o = torch.empty_like(q)
    grid = (num_seqs, num_q_heads)
    fused_kv_cache_attention_kernel[grid](
        q, q.stride(0), q.stride(1), q.stride(2),
        k, k.stride(0), k.stride(1), k.stride(2),
        v, v.stride(0), v.stride(1), v.stride(2),
        k_cache, k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache, v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_tables, block_tables.stride(0), block_tables.stride(1),
        o, o.stride(0), o.stride(1), o.stride(2),
        cu_seqlens_q, cu_seqlens_k,
        num_seqs, block_len, block_size, num_q_heads, num_kv_heads, head_dim,
        float(sm_scale),
        # BLOCK_M=head_dim if head_dim in (16, 32, 64, 128) else 16,
        # BLOCK_N=64,
        BLOCK_DMODEL=head_dim,
    )
    return o

if __name__ == "__main__":
    pass
