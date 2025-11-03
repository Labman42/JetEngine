import torch
from torch import nn
import triton
import triton.language as tl

from jetengine.utils.context import get_context
from jetengine.engine.sequence import RunType
from jetengine.kernels.triton.attention import sparse_attn_varlen
# from jetengine.kernels.triton.attention import fused_kv_cache_attention
# from jetengine.kernels.triton.attention import fused_kv_cache_attention_v5
from flash_attn import flash_attn_with_kvcache, flash_attn_varlen_func


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        pass

class BlockAttention(Attention):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__(num_heads, head_dim, scale, num_kv_heads)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        should_store_whole = (context.run_type == RunType.PREFILL)
        if should_store_whole and k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            
        if context.run_type == RunType.PREFILL:
            o = sparse_attn_varlen(q, k, v,
                                cu_seqlens_q=context.cu_seqlens_q,
                                cu_seqlens_k=context.cu_seqlens_k,
                                staircase_size=context.block_length)
        else:
            q = q.view(-1, context.block_length, self.num_heads, self.head_dim)
            k = k.view(-1, context.block_length, self.num_kv_heads, self.head_dim)
            v = v.view(-1, context.block_length, self.num_kv_heads, self.head_dim)
            o = flash_attn_with_kvcache(q, k_cache=k_cache, v_cache=v_cache, k=k, v=v,
                                        cache_seqlens=context.context_lens,
                                        block_table=context.block_tables,
                                        causal=False)  # Assuming non-causal for benchmark consistency     
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
    
class LladaBlockAttention(Attention):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__(num_heads, head_dim, scale, num_kv_heads)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        should_store_whole = (context.run_type == RunType.PREFILL)
        if should_store_whole and k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            
        if context.run_type == RunType.PREFILL:
            max_seqlen_q = (context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]).max().item()
            max_seqlen_k = (context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]).max().item()
            o = flash_attn_varlen_func(q, k, v,
                            max_seqlen_q=max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                            max_seqlen_k=max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,)
        else:
            q = q.view(-1, context.block_length, self.num_heads, self.head_dim)
            k = k.view(-1, context.block_length, self.num_kv_heads, self.head_dim)
            v = v.view(-1, context.block_length, self.num_kv_heads, self.head_dim)
            o = flash_attn_with_kvcache(q, k_cache=k_cache, v_cache=v_cache, k=k, v=v,
                                        cache_seqlens=context.context_lens,
                                        block_table=context.block_tables,
                                        causal=False)  # Assuming non-causal for benchmark consistency     
        o = o.view(-1, self.num_heads * self.head_dim)
        return o


        
