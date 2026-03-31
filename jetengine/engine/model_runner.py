import math
import pickle
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist

from jetengine.config import Config
from jetengine.engine.sequence import Sequence, RunType, SequenceStatus
from jetengine.models.sdar import SDARForCausalLM
from jetengine.models.sdar_moe import SDARMoeForCausalLM
from jetengine.models.llada import LladaForCausalLM
from jetengine.utils.context import set_context, get_context, reset_context
from jetengine.utils.loader import load_model
from jetengine.engine.distributed_manager import DistributedManager
from jetengine.layers.linear import ColumnParallelLinear, RowParallelLinear

try:
    import flashinfer
    _FLASHINFER_AVAILABLE = True
except ImportError:
    _FLASHINFER_AVAILABLE = False


def quantize_model_fp8(model: torch.nn.Module):
    """Quantize all linear layers to static FP8 (weights pre-quantized, activations cast inline).
    Uses per-tensor weight scaling. Activation cast is a simple bf16->fp8 truncation (no dynamic scaling).
    Skips LM head when weights are tied with embeddings."""
    from jetengine.layers.linear import LinearBase
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, LinearBase):
            continue
        if not hasattr(module, 'weight') or module.weight is None:
            continue
        # Skip if weight is shared with embedding (tie_word_embeddings)
        if hasattr(module, 'weight') and module.weight.dtype == torch.float8_e4m3fn:
            continue
        module.quantize_to_fp8()
        count += 1
    print(f"[FP8-static] Quantized {count} linear layers to FP8 (direct cast, no dynamic scaling)")


class ModelRunner:

    def __init__(self, config: Config, dist_manager: DistributedManager):
        self.config = config
        self.dist_manager = dist_manager

        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = dist_manager.tp_size
        self.rank = dist_manager.tp_rank

        torch.cuda.set_device(dist_manager.device)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.torch_dtype)
        torch.set_default_device("cuda")

        model_kwargs = {"config": hf_config,
                        "process_group": self.dist_manager.tp_group}
        if "sdar" in hf_config.model_type and "moe" in hf_config.model_type:
            raise ValueError(f"MoE not supported for dp tp hybrid yet")
            self.ModelClass = SDARMoeForCausalLM
        elif "sdar" in hf_config.model_type:
            self.ModelClass = SDARForCausalLM
        elif "llada" in hf_config.model_type.lower():  # <-- ADD THIS BLOCK
            self.ModelClass = LladaForCausalLM
        else:
            raise ValueError(f"Unsupported model type: {hf_config.model_type}")
        self.model = self.ModelClass(**model_kwargs)
        load_model(self.model, config.model)
        if config.quantize_fp8:
            quantize_model_fp8(self.model)
            torch.cuda.empty_cache()
        if config.torch_compile:
            self._apply_torch_compile()
        # Sampler is removed from here
        self.warmup_model()
        self.allocate_kv_cache()
        self._init_flashinfer()
        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        
    def _init_flashinfer(self):
        """Initialize flashinfer wrappers for denoise attention acceleration.
        Creates one wrapper per batch size (1..fi_max_bs) with use_cuda_graph=True.
        Larger batches fall back to flash_attn."""
        self._use_flashinfer = _FLASHINFER_AVAILABLE
        self._fi_wrappers = {}
        self._fi_bufs = {}
        if not self._use_flashinfer:
            return

        # Cap flashinfer wrappers: each wrapper uses ~74 MiB of GPU memory
        # 128 wrappers ≈ 9.5 GiB, feasible on H200 (140 GiB)
        fi_max_bs = min(self.config.max_num_seqs, 128)
        max_bs = min(self.config.max_num_seqs, 256)
        block_len = self.config.block_length
        max_blocks_per_seq = math.ceil(
            (self.config.max_model_len + block_len) / self.block_size
        ) + 1

        # Slot mapping for block tokens (shared, max size)
        self._fi_slot_mapping = torch.zeros(
            max_bs * block_len, dtype=torch.int64, device="cuda"
        )

        # Create wrappers outside inference_mode so buffers are writable later
        for bs in range(1, fi_max_bs + 1):
            workspace = torch.empty(2 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            qo_indptr = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
            kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
            kv_indices = torch.zeros(bs * max_blocks_per_seq, dtype=torch.int32, device="cuda")
            kv_last_page_len = torch.zeros(bs, dtype=torch.int32, device="cuda")
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace, "NHD",
                use_cuda_graph=True,
                qo_indptr_buf=qo_indptr,
                paged_kv_indptr_buf=kv_indptr,
                paged_kv_indices_buf=kv_indices,
                paged_kv_last_page_len_buf=kv_last_page_len,
            )
            self._fi_wrappers[bs] = wrapper
            self._fi_bufs[bs] = {
                'workspace': workspace,
                'qo_indptr': qo_indptr,
                'kv_indptr': kv_indptr,
                'kv_indices': kv_indices,
                'kv_last_page_len': kv_last_page_len,
            }

        print(f"[flashinfer] Created {fi_max_bs} wrappers (CUDA graph mode), page_size={self.block_size}")

    def _plan_flashinfer(self, batch_size, block_tables, context_lens, block_len):
        """Plan flashinfer attention for current denoise batch.
        Uses the pre-allocated per-batch-size wrapper and buffers."""
        wrapper = self._fi_wrappers.get(batch_size)
        if wrapper is None:
            return None, None
        bufs = self._fi_bufs[batch_size]

        # qo_indptr: uniform block_len per sequence
        qo_indptr = bufs['qo_indptr']
        torch.arange(0, (batch_size + 1) * block_len, block_len,
                      dtype=torch.int32, device="cuda", out=qo_indptr)

        # Total KV length per sequence = context_lens + block_len (includes new block tokens)
        total_kv_lens = context_lens.int() + block_len  # (bs,)
        pages_per_seq = (total_kv_lens + self.block_size - 1) // self.block_size  # (bs,)

        # Build paged_kv_indptr
        kv_indptr = bufs['kv_indptr']
        kv_indptr[0] = 0
        torch.cumsum(pages_per_seq, dim=0, out=kv_indptr[1:])
        total_pages = kv_indptr[batch_size].item()

        # Flatten block_tables into paged_kv_indices
        kv_indices = bufs['kv_indices'][:total_pages]
        max_pages = pages_per_seq.max().item()
        if block_tables is not None and block_tables.shape[1] >= max_pages:
            if pages_per_seq.min().item() == max_pages:
                # Uniform pages: single reshape+copy
                kv_indices.copy_(block_tables[:, :max_pages].reshape(-1)[:total_pages])
            else:
                # Variable pages: vectorized gather using mask
                bt = block_tables[:, :max_pages]  # (bs, max_pages)
                col_idx = torch.arange(max_pages, device="cuda").unsqueeze(0)
                mask = col_idx < pages_per_seq.unsqueeze(1)  # (bs, max_pages)
                kv_indices.copy_(bt[mask])

        # last_page_len
        kv_last_page_len = bufs['kv_last_page_len']
        remainder = total_kv_lens % self.block_size
        kv_last_page_len.copy_(torch.where(remainder == 0, self.block_size, remainder))

        # Compute slot mapping for block tokens
        slot_mapping = self._fi_slot_mapping[:batch_size * block_len]
        offsets = torch.arange(block_len, dtype=torch.int64, device="cuda")
        positions = context_lens.unsqueeze(1).to(torch.int64) + offsets.unsqueeze(0)
        block_idx = positions // self.block_size
        block_offset = positions % self.block_size
        physical_blocks = torch.gather(block_tables.to(torch.int64), 1, block_idx)
        slot_mapping.copy_((physical_blocks * self.block_size + block_offset).reshape(-1))

        # Plan the wrapper (updates pre-allocated buffers read by CUDA graph)
        wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=kv_last_page_len,
            num_qo_heads=self.config.num_attention_heads // self.world_size,
            num_kv_heads=self.config.num_key_value_heads // self.world_size,
            head_dim_qk=self.config.head_dim,
            page_size=self.block_size,
            causal=False,
            q_data_type=self.config.torch_dtype,
        )

        return wrapper, slot_mapping

    def _apply_torch_compile(self):
        """Apply torch.compile to decoder layers for kernel fusion."""
        inner_model = getattr(self.model, 'model', None)
        if inner_model is None or not hasattr(inner_model, 'layers'):
            print("[torch.compile] No decoder layers found, skipping")
            return
        count = 0
        for i, layer in enumerate(inner_model.layers):
            inner_model.layers[i] = torch.compile(layer, fullgraph=False)
            count += 1
        print(f"[torch.compile] Compiled {count} decoder layers")

    def reinit_model(self):
        self.model = self.ModelClass(
            self.config.hf_config, self.dist_manager.tp_group)

    def exit(self):
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens //
                       max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len, self.config.mask_token_id)
                for _ in range(num_seqs)]
        self.run(seqs, RunType.PREFILL)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = config.num_key_value_heads // self.world_size
        block_bytes = 2 * config.num_hidden_layers * self.block_size * \
            num_kv_heads * config.head_dim * config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(
            total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        print(f"[rank {dist.get_rank()}][KVCache] Allocated {config.num_kvcache_blocks:,} blocks "
        f"({config.num_kvcache_blocks * block_bytes / (1024**3):.2f} GiB) "
        f"based on peak memory usage.")
        self.kv_cache = torch.zeros(2, config.num_hidden_layers, config.num_kvcache_blocks,
                                    self.block_size, num_kv_heads, config.head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        if max_len == 0:
            return None
        block_tables = [seq.block_table + [-1] *
                        (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables
    def prepare_prefill_loop(self, seqs: list[Sequence]):
        input_ids, positions, cu_seqlens_q, slot_mapping, is_last_step = [], [], [0], [], []
        max_seqlen_q = 0
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq.token_ids)
            positions.extend(range(seqlen))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen)
            max_seqlen_q = max(max_seqlen_q, seqlen)
            is_last_step.append(False)
            # Slot mapping for prefill
            if not seq.block_table:
                continue
            # Slot mapping for prefill
            if not seq.block_table:
                continue
            for i in range(seqlen):
                block_idx = i // self.block_size
                block_offset = i % self.block_size
                physical_block_id = seq.block_table[block_idx]
                slot = physical_block_id * self.block_size + block_offset
                slot_mapping.append(slot)

        device = torch.device("cuda")
        input_ids_cpu = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True)
        positions_cpu = torch.tensor(positions, dtype=torch.int64, pin_memory=True)
        cu_seqlens_q_cpu = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True)
        slot_mapping_cpu = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True)

        input_ids = input_ids_cpu.to(device=device, non_blocking=True)
        positions = positions_cpu.to(device=device, non_blocking=True)
        cu_seqlens_q = cu_seqlens_q_cpu.to(device=device, non_blocking=True)
        slot_mapping = slot_mapping_cpu.to(device=device, non_blocking=True)
        set_context(
            run_type=RunType.PREFILL,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            slot_mapping=slot_mapping,
            is_last_denoise_step=is_last_step,  # <-- Pass the new flag
            block_length=self.config.block_length
        )
        return input_ids, positions

    def prepare_prefill(self, seqs: list[Sequence]):
        device = torch.device("cuda") 
        all_token_ids_flat = []
        all_block_tables_flat = []
        seqlens_list = []
        block_table_lens_list = []
        is_last_step_list = []

        for seq in seqs:
            seq_len = len(seq.token_ids)
            seqlens_list.append(seq_len)
            all_token_ids_flat.extend(seq.token_ids)
            is_last_step_list.append(False)

            if seq.block_table:
                block_table_lens_list.append(len(seq.block_table))
                all_block_tables_flat.extend(seq.block_table)
            else:
                block_table_lens_list.append(0)
        
        # (total_tokens,)
        input_ids_cpu = torch.tensor(all_token_ids_flat, dtype=torch.int64, pin_memory=True)
        input_ids = input_ids_cpu.to(device=device, non_blocking=True)
        
        # (total_physical_blocks,)
        flat_block_tables_cpu = torch.tensor(
            all_block_tables_flat, dtype=torch.int32, pin_memory=True
        )
        flat_block_tables = flat_block_tables_cpu.to(device=device, non_blocking=True)
        
        # (B,)
        seqlens_q_cpu = torch.tensor(seqlens_list, dtype=torch.int32, pin_memory=True)
        block_table_lens_cpu = torch.tensor(
            block_table_lens_list, dtype=torch.int32, pin_memory=True
        )
        is_last_step_cpu = torch.tensor(is_last_step_list, dtype=torch.bool, pin_memory=True)

        seqlens_q = seqlens_q_cpu.to(device=device, non_blocking=True)
        block_table_lens = block_table_lens_cpu.to(device=device, non_blocking=True)
        is_last_step = is_last_step_cpu.to(device=device, non_blocking=True)
        
        batch_size = len(seqs)
        if batch_size == 0:
            # Handle empty batch case
            cu_seqlens_q_cpu = torch.tensor([0], dtype=torch.int32, pin_memory=True)
            cu_seqlens_q = cu_seqlens_q_cpu.to(device=device, non_blocking=True)
            max_seqlen_q = 0
            positions_cpu = torch.empty(0, dtype=torch.int64, pin_memory=True)
            slot_mapping_cpu = torch.empty(0, dtype=torch.int32, pin_memory=True)
            positions = positions_cpu.to(device=device, non_blocking=True)
            slot_mapping = slot_mapping_cpu.to(device=device, non_blocking=True)
        else:
            # (B+1,)
            cu_seqlens_q = torch.nn.functional.pad(
                seqlens_q.cumsum(dim=0, dtype=torch.int32), (1, 0)
            )
            cu_block_table_lens = torch.nn.functional.pad(
                block_table_lens.cumsum(dim=0, dtype=torch.int32), (1, 0)
            )
            max_seqlen_q = seqlens_q.max().item()
            total_tokens = input_ids.shape[0]
            token_indices_global = torch.arange(total_tokens, dtype=torch.int64, device=device)
            seq_start_offsets = torch.repeat_interleave(
                cu_seqlens_q[:-1], repeats=seqlens_q
            )
            positions = token_indices_global - seq_start_offsets # This is `i`
            
            # [True, False, True, ...] (B,)
            has_block_table_mask_per_seq = (block_table_lens > 0)
            # [T, T, T, F, F, T, T, T, T, ...] (total_tokens,)
            has_block_table_mask_per_token = torch.repeat_interleave(
                has_block_table_mask_per_seq, repeats=seqlens_q
            )
            # Filter `positions` to only those that need a slot
            # (total_slots,)
            i = positions[has_block_table_mask_per_token]
            # (total_tokens,)
            seq_idx_for_each_token = torch.repeat_interleave(
                torch.arange(batch_size, device=device), repeats=seqlens_q
            )
            seq_idx_with_slot = seq_idx_for_each_token[has_block_table_mask_per_token]
            block_idx = i // self.block_size
            block_offset = i % self.block_size
            block_table_start_offsets = cu_block_table_lens[seq_idx_with_slot]
            block_table_global_idx = block_table_start_offsets + block_idx
            physical_block_id = flat_block_tables[block_table_global_idx]
            slot_mapping = physical_block_id * self.block_size + block_offset
        set_context(
            run_type=RunType.PREFILL,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            slot_mapping=slot_mapping.to(torch.int32), # Ensure int32
            is_last_denoise_step=is_last_step,
            block_length=self.config.block_length
        )
        return input_ids, positions

    def prepare_denoise_loop(self, seqs: list[Sequence]):
        input_ids, positions = [], []
        cached_lens = []

        for seq in seqs:
            # The query is the current intermediate block
            q_tokens = seq.intermediate_block_tokens
            q_len = len(q_tokens)

            # The context (key/value) is the confirmed part of the sequence
            k_len = len(seq)

            input_ids.extend(q_tokens)
            # Positions are global
            positions.extend(range(k_len, k_len + q_len))
            cached_lens.append(k_len)

        device = torch.device("cuda")
        input_ids_cpu = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True)
        positions_cpu = torch.tensor(positions, dtype=torch.int64, pin_memory=True)
        cached_lens_cpu = torch.tensor(cached_lens, dtype=torch.int32, pin_memory=True)

        input_ids = input_ids_cpu.to(device=device, non_blocking=True)
        positions = positions_cpu.to(device=device, non_blocking=True)
        cached_lens = cached_lens_cpu.to(device=device, non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        set_context(
            run_type=RunType.DENOISE,
            context_lens=cached_lens,
            block_tables=block_tables,
            block_length=self.config.block_length
        )

        return input_ids, positions
    
    def prepare_denoise(self, seqs: list[Sequence], reuse_context: bool = False,
                        chain_batch_tokens: torch.Tensor | None = None):
        device = torch.device("cuda")
        batch_size = len(seqs)
        block_len = seqs[0].block_length

        if chain_batch_tokens is not None:
            # Fast path: use pre-existing batch tensor from chain_state (avoids loop + cat)
            input_ids = chain_batch_tokens.reshape(-1)
        else:
            block_tokens_list = []
            for seq in seqs:
                t = seq.intermediate_block_tokens
                if t.device != device:
                    t = t.to(device, non_blocking=True)
                    seq.intermediate_block_tokens = t
                block_tokens_list.append(t)
            input_ids = torch.cat(block_tokens_list)

        if reuse_context:
            # Fast path for chain steps: only update input_ids, reuse positions/context
            # Re-set the cached context (reset_context clears it between run() calls)
            if hasattr(self, '_chain_context_kwargs'):
                set_context(**self._chain_context_kwargs)
            return input_ids, self._chain_positions

        cached_lens = torch.tensor(
            [len(seq) for seq in seqs],
            dtype=torch.int32,
            device=device
        )

        # Cache the offsets tensor (same every step)
        if not hasattr(self, '_denoise_offsets') or self._denoise_offsets.shape[0] != block_len:
            self._denoise_offsets = torch.arange(block_len, dtype=torch.int64, device=device).unsqueeze(0)

        positions = (cached_lens.unsqueeze(1).to(torch.int64) + self._denoise_offsets).view(-1)
        block_tables = self.prepare_block_tables(seqs)

        fi_wrapper = None
        fi_slot_mapping = None
        if self._use_flashinfer and block_tables is not None:
            fi_wrapper, fi_slot_mapping = self._plan_flashinfer(
                batch_size, block_tables, cached_lens, block_len
            )

        ctx_kwargs = dict(
            run_type=RunType.DENOISE,
            context_lens=cached_lens,
            block_tables=block_tables,
            block_length=self.config.block_length,
            fi_wrapper=fi_wrapper,
            fi_slot_mapping=fi_slot_mapping,
        )
        set_context(**ctx_kwargs)

        # Cache for chain reuse
        self._chain_positions = positions
        self._chain_context_kwargs = ctx_kwargs

        return input_ids, positions


    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor):
        return self.model.compute_logits(self.model(input_ids, positions))

    @torch.inference_mode()
    def run_model_selective(self, input_ids: torch.Tensor, positions: torch.Tensor, logits_len: int):
        """Run model and compute logits only for the first logits_len tokens."""
        hidden = self.model(input_ids, positions)
        return self.model.compute_logits(hidden[:logits_len])

    @torch.inference_mode()
    def _run_denoise_with_cudagraph(
        self,
        seqs: list[Sequence],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        num_need_logits: int = None,
        reuse_context: bool = False,
        masked_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, bool]:
        if (
            self.enforce_eager
            or not hasattr(self, "graphs")
            or not getattr(self, "graphs", None)
        ):
            return None, False

        context = get_context()
        if not reuse_context and context.run_type != RunType.DENOISE:
            return None, False

        batch_size = len(seqs)
        if batch_size == 0:
            return None, False

        graph = self.graphs.get(batch_size)
        if graph is None:
            return None, False

        graph_vars = self.graph_vars
        block_len = self.config.block_length
        global_bs = batch_size * block_len

        if global_bs > graph_vars["input_ids"].shape[0]:
            return None, False

        # Always update input_ids (tokens change every step)
        graph_vars["input_ids"][:global_bs].copy_(input_ids)

        if not reuse_context:
            # Full context setup (first chain step or after batch change)
            if context.context_lens is None:
                return None, False
            graph_vars["positions"][:global_bs].copy_(positions)
            graph_context_lens = graph_vars["context_lens"][:batch_size]
            graph_context_lens.copy_(context.context_lens)

            graph_block_tables = graph_vars["block_tables"][:batch_size]
            graph_block_tables.fill_(-1)
            if context.block_tables is not None:
                required_blocks = context.block_tables.shape[1]
                if required_blocks > graph_block_tables.shape[1]:
                    return None, False
                graph_block_tables[:, :required_blocks].copy_(context.block_tables)

            # Plan flashinfer with updated metadata before graph replay
            fi_wrapper = None
            fi_slot_mapping = None
            if self._use_flashinfer and context.block_tables is not None:
                fi_wrapper, fi_slot_mapping = self._plan_flashinfer(
                    batch_size, graph_block_tables, graph_context_lens, block_len
                )

            set_context(
                run_type=RunType.DENOISE,
                context_lens=graph_context_lens,
                block_tables=graph_block_tables,
                block_length=self.config.block_length,
                is_last_denoise_step=context.is_last_denoise_step,
                fi_wrapper=fi_wrapper,
                fi_slot_mapping=fi_slot_mapping,
            )
        else:
            # reuse_context: skip plan() for chain steps (same batch/pages)
            # The pre-allocated buffers and workspace are unchanged from the previous step
            pass

        graph.replay()
        # Graph outputs hidden states; compute_logits runs selectively
        hidden = graph_vars["hidden_out"][:global_bs]
        # Only compute logits for sequences that need them (DENOISING first in batch)
        if num_need_logits is not None and num_need_logits < batch_size:
            logits_len = num_need_logits * block_len
            hidden = hidden[:logits_len]
        # Sparse logits: only compute LM head for masked positions
        if masked_indices is not None:
            hidden = hidden[masked_indices]
        logits = self.model.compute_logits(hidden)
        return logits, True

    def run(self, seqs: list[Sequence], run_type: RunType, num_need_logits: int = None,
            reuse_context: bool = False, chain_batch_tokens: torch.Tensor | None = None,
            masked_indices: torch.Tensor | None = None) -> torch.Tensor:
        if run_type == RunType.PREFILL:
            input_ids, positions = self.prepare_prefill(seqs)
        elif run_type == RunType.DENOISE:
            input_ids, positions = self.prepare_denoise(
                seqs, reuse_context=reuse_context, chain_batch_tokens=chain_batch_tokens)
        else:
            return None

        need_selective = (run_type == RunType.DENOISE and
                          num_need_logits is not None and
                          num_need_logits < len(seqs))

        if run_type == RunType.DENOISE and not self.enforce_eager:
            logits, used_graph = self._run_denoise_with_cudagraph(
                seqs, input_ids, positions, num_need_logits=num_need_logits,
                reuse_context=reuse_context, masked_indices=masked_indices,
            )
            if not used_graph:
                if need_selective:
                    logits = self.run_model_selective(
                        input_ids, positions, num_need_logits * seqs[0].block_length)
                else:
                    logits = self.run_model(input_ids, positions)
        elif need_selective:
            logits = self.run_model_selective(
                input_ids, positions, num_need_logits * seqs[0].block_length)
        else:
            logits = self.run_model(input_ids, positions)
        reset_context()
        return logits if self.rank == 0 else None

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        # Cap graph captures at flashinfer wrapper limit (saves memory for unused batch sizes)
        fi_cap = 128 if self._use_flashinfer else 256
        max_bs = min(self.config.max_num_seqs, fi_cap)
        max_global_bs = max_bs * self.config.block_length
        max_num_blocks = math.ceil(
            (config.max_model_len + self.config.block_length) / self.block_size
        ) + 1
        input_ids = torch.zeros(max_global_bs, dtype=torch.int64)
        positions = torch.zeros(max_global_bs, dtype=torch.int64)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)

        self._graph_includes_logits = False
        # Graph outputs hidden states; compute_logits runs outside graph (selectively)
        hidden_out = torch.zeros(
            max_global_bs, config.hf_config.hidden_size, dtype=config.torch_dtype
        )

        # Capture every batch size from 1 to max_bs for near-100% hit rate
        self.graph_bs = list(range(1, max_bs + 1))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            # Plan flashinfer for this batch size (wrappers created in _init_flashinfer)
            fi_wrapper = None
            fi_slot_mapping = None
            if self._use_flashinfer and bs in self._fi_wrappers:
                fi_wrapper, fi_slot_mapping = self._plan_flashinfer(
                    bs, block_tables[:bs], context_lens[:bs], self.config.block_length
                )
            set_context(run_type=RunType.DENOISE,
                        context_lens=context_lens[:bs], block_tables=block_tables[:bs],
                        block_length=self.config.block_length,
                        fi_wrapper=fi_wrapper, fi_slot_mapping=fi_slot_mapping)
            global_bs = bs * self.config.block_length
            # warmup
            hidden = self.model(input_ids[:global_bs], positions[:global_bs])
            hidden_out[:global_bs] = hidden
            with torch.cuda.graph(graph, self.graph_pool):
                hidden = self.model(input_ids[:global_bs], positions[:global_bs])
                hidden_out[:global_bs] = hidden
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            context_lens=context_lens,
            block_tables=block_tables,
            hidden_out=hidden_out,
        )
