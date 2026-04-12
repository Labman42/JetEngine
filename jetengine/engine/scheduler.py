from collections import deque
from dataclasses import dataclass
import torch
from torch.nn import functional as F
import numpy as np
import random

from jetengine.config import Config
from jetengine.engine.sequence import Sequence, SequenceStatus, RunType
from jetengine.engine.block_manager import BlockManager
from jetengine.layers.sampler import sample_with_temperature_topk_topp
from flashinfer.logits_processor import LogitsPipe, Temperature, Softmax, TopP, TopK, Sample
from flashinfer.sampling import top_p_sampling_from_probs, top_k_top_p_sampling_from_probs
from torch.distributions import Categorical

EPS = 1e-12


@dataclass
class ScheduleResult:
    prefill: list[Sequence]
    denoise: list[Sequence]
    @property
    def has_work(self) -> bool:
        return bool(self.prefill or self.denoise)

class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.mask_token_id = config.mask_token_id
        self.diversity_enforce = config.diversity_enforce
        self.epsilon_greedy = config.epsilon_greedy
        self.epsilon = config.epsilon
        self.barrier = config.diversity_enforce_barrier
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.running: list[Sequence] = []
        self.waiting_prefill: deque[Sequence] = deque()
        self.prefill_ready: deque[Sequence] = deque()
        self.sample_pipe = LogitsPipe([
                                Temperature(),      # Scale logits by temperature
                                Softmax(),          # Convert logits to probabilities
                            ])
        self.log_step_signals = getattr(config, 'log_step_signals', False)
        
    def _sample_tokens(self, probs: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
        """Sample tokens from probability distribution.
        Hybrid path: torch.multinomial for top_p=1.0/top_k=0 (2x faster than
        flashinfer on this path), flashinfer for top_p<1.0 or top_k>0 (3x
        faster than torch on nucleus/topk filtering). The flashinfer
        correlation bug (identical samples for identical rows) is mitigated
        by prompt interleaving in generate_streaming."""
        if probs.shape[0] < 8:
            return torch.multinomial(probs, num_samples=1).squeeze(-1).to(torch.int64)
        if top_p >= 1.0 and top_k <= 0:
            # Pure multinomial: torch is 2x faster than flashinfer here
            return torch.multinomial(probs, num_samples=1).squeeze(-1).to(torch.int64)
        # Nucleus / top-k filtering: flashinfer is 3x faster than torch here
        return top_k_top_p_sampling_from_probs(
            probs, top_k=top_k, top_p=top_p
        ).to(torch.int64)

    def apply_repetition_penalty(self, probs: torch.Tensor, seqs: list[Sequence]):
        # probs: (batch_size, block_len, vocab_size)
        for i, seq in enumerate(seqs):
            if seq.repetition_penalty == 1.0 or seq.num_tokens <= seq.max_tokens // 2:
                continue
            
            penalty = seq.repetition_penalty
            seen_tokens = list(seq.token_counts.keys())
            
            # Apply penalty to all positions in the block for the seen tokens
            # We use advanced indexing. 
            # probs[i, :, seen_tokens] selects (block_len, num_seen)
            if seen_tokens:
                probs[i, :, seen_tokens] /= penalty
                
            # Renormalize
            probs[i] /= probs[i].sum(dim=-1, keepdim=True)

    def add(self, seq: Sequence):
        if not self._try_add_to_running(seq):
            self.waiting_prefill.append(seq)

    def _try_add_to_running(self, seq: Sequence) -> bool:
        if len(self.running) >= self.max_num_seqs:
            return False
        if seq.is_finished:
            return False
        if not self.block_manager.can_allocate(seq):
            return False

        self.block_manager.allocate(seq)
        if seq.status == SequenceStatus.WAITING:
            seq.status = SequenceStatus.PREFILLING
            self.prefill_ready.append(seq)
        self.running.append(seq)
        return True

    def _release_finished_sequences(self):
        if not self.running:
            return
        finished = [seq for seq in self.running if seq.is_finished]
        if not finished:
            return
        for seq in finished:
            self.block_manager.deallocate(seq)
        self.running = [seq for seq in self.running if not seq.is_finished]
        self.prefill_ready = deque(
            seq for seq in self.prefill_ready if not seq.is_finished
        )

    def _fill_slots_from_waiting(self):
        while self.waiting_prefill and len(self.running) < self.max_num_seqs:
            seq = self.waiting_prefill[0]
            if self._try_add_to_running(seq):
                self.waiting_prefill.popleft()
            else:
                break

    def is_finished(self):
        return not self.running and not self.waiting_prefill and not self.prefill_ready

    def _prepare_prefill_batch(self) -> list[Sequence]:
        batch: list[Sequence] = []
        while self.prefill_ready:
            seq = self.prefill_ready.popleft()
            if seq.is_finished:
                # Fix: Ensure we deallocate if a finished sequence was in prefill_ready
                if seq.block_table:
                    self.block_manager.deallocate(seq)
                continue
            if seq.status not in (SequenceStatus.PREFILLING, SequenceStatus.WAITING):
                continue
            seq.status = SequenceStatus.PREFILLING
            batch.append(seq)
        return batch

    def _commit_saving_sequences(self) -> list[Sequence]:
        """Commit SAVING sequences immediately without running them through the model.
        SAVING sequences have fully-denoised blocks ready to commit — no forward pass needed.
        Returns list of sequences that finished during commit."""
        saving_seqs = [seq for seq in self.running if seq.status == SequenceStatus.SAVING]
        if not saving_seqs:
            return []
        device = saving_seqs[0].intermediate_block_tokens.device
        s_tokens = torch.stack([s.intermediate_block_tokens for s in saving_seqs])
        s_traj = torch.stack([s.block_trajectory for s in saving_seqs])
        s_logp = torch.stack([s.block_logprobs for s in saving_seqs])
        s_ent = torch.stack([s.block_entropies for s in saving_seqs])
        if device.type != 'cpu':
            s_tokens_cpu = s_tokens.cpu().tolist()
            s_traj_cpu = s_traj.cpu().tolist()
            s_logp_cpu = s_logp.cpu().tolist()
            s_ent_cpu = s_ent.cpu().tolist()
        else:
            s_tokens_cpu = s_tokens.tolist()
            s_traj_cpu = s_traj.tolist()
            s_logp_cpu = s_logp.tolist()
            s_ent_cpu = s_ent.tolist()
        finished = []
        for j, seq in enumerate(saving_seqs):
            seq._commit_block_from_cpu(s_tokens_cpu[j], s_traj_cpu[j], s_logp_cpu[j], s_ent_cpu[j])
            seq.num_to_transfer = 0
            if seq.is_finished:
                finished.append(seq)
            else:
                seq.start_new_block()
        return finished

    def _prepare_denoise_batch(self, prefill_batch: list[Sequence]) -> list[Sequence]:
        batch: list[Sequence] = []
        prefill_ids = {seq.seq_id for seq in prefill_batch}

        requests: list[tuple[Sequence, int]] = []
        total_needed = 0
        available_blocks = len(self.block_manager.free_block_ids)

        denoising_batch = []
        saving_batch = []
        for seq in self.running:
            if seq.seq_id in prefill_ids:
                continue
            # Include both DENOISING and SAVING sequences — SAVING needs the forward pass
            # to update KV cache with final committed tokens' representations
            if seq.status not in (SequenceStatus.DENOISING, SequenceStatus.SAVING):
                continue

            num_new_blocks = seq.num_new_blocks_needed(self.block_manager.block_size)
            if num_new_blocks > 0:
                if total_needed + num_new_blocks > available_blocks:
                    print(f"[Warning] Cannot append {num_new_blocks} blocks for seq {seq.seq_id}. Not enough memory.")
                    continue
                requests.append((seq, num_new_blocks))
                total_needed += num_new_blocks

            if seq.status == SequenceStatus.DENOISING:
                denoising_batch.append(seq)
            else:
                saving_batch.append(seq)

        # DENOISING first, SAVING last — allows selective compute_logits
        batch = denoising_batch + saving_batch
        self._num_denoising_in_batch = len(denoising_batch)

        if requests:
            self.block_manager.append_blocks_batch(requests)

        return batch

    def try_allocate_chain_blocks(self, seqs: list[Sequence]) -> bool:
        """Allocate any new KV blocks needed for mid-chain sequences.
        Called between chain iterations when just-committed sequences
        may need an extra block (crosses block_size boundary).
        Returns True if allocation succeeded."""
        requests = []
        total_needed = 0
        available = len(self.block_manager.free_block_ids)
        for seq in seqs:
            n = seq.num_new_blocks_needed(self.block_manager.block_size)
            if n > 0:
                total_needed += n
                if total_needed > available:
                    return False
                requests.append((seq, n))
        if requests:
            self.block_manager.append_blocks_batch(requests)
        return True

    def schedule(self) -> ScheduleResult:
        self._release_finished_sequences()
        self._fill_slots_from_waiting()

        prefill_batch = self._prepare_prefill_batch()
        denoise_batch = self._prepare_denoise_batch(prefill_batch)

        if not prefill_batch and not denoise_batch:
            if self.running or self.waiting_prefill:
                pass  # Normal during drain when all seqs are saving/finishing
            return ScheduleResult(prefill=[], denoise=[])

        return ScheduleResult(prefill=prefill_batch, denoise=denoise_batch)

    def postprocess_loop(self, seqs: list[Sequence], logits: torch.Tensor, run_type: RunType):
        if run_type == RunType.PREFILL:
            for seq in seqs:
                seq.num_cached_tokens = seq.num_prefill_tokens
                seq.status = SequenceStatus.DENOISING
        
        elif run_type == RunType.DENOISE:
            start_idx = 0
            if self.consistent_sampling_params: # Assume in training environment
                probs = self.sample_pipe(logits, temperature=seqs[0].temperature)
                entropies = -(probs.clamp_min(EPS) * (probs.clamp_min(EPS)).log()).sum(dim=-1)
                batch_seq_x0 = top_k_top_p_sampling_from_probs(probs, top_k=seqs[0].top_k, top_p=seqs[0].top_p).to(torch.int64)
                batch_seq_x0_p = torch.gather(probs, -1, batch_seq_x0.unsqueeze(-1)).squeeze(-1)    
            for seq in seqs:
                # Extract the part of the tensors relevant to this sequence
                if seq.status == SequenceStatus.DENOISING:
                    block_len = seq.block_length
                    if not self.consistent_sampling_params:
                        probs = self.sample_pipe(logits[start_idx : start_idx + block_len], temperature=seq.temperature)
                        seq_x0 = top_k_top_p_sampling_from_probs(probs, top_k=seq.top_k, top_p=seq.top_p).to(torch.int64)
                        seq_x0_p = torch.gather(probs, -1, seq_x0.unsqueeze(-1)).squeeze(-1)    
                        seq_entropies = -(probs.clamp_min(EPS) * (probs.clamp_min(EPS)).log()).sum(dim=-1)

                    else:
                        seq_x0 = batch_seq_x0[start_idx : start_idx + block_len]
                        seq_x0_p = batch_seq_x0_p[start_idx : start_idx + block_len]
                        seq_entropies = entropies[start_idx : start_idx + block_len]
                
                    seq_x0_logp = torch.log(seq_x0_p.clamp_min(EPS))
                    
                    # Use tensor directly
                    current_block_tensor = seq.intermediate_block_tokens.to(logits.device)
                    mask_index = (current_block_tensor == self.mask_token_id)
                    num_to_transfer = seq.num_transfer_tokens_per_step[seq.current_denoising_step]
                    
                    transfer_index = torch.zeros_like(seq_x0, dtype=torch.bool)
                    
                    if seq.remasking_strategy == 'sequential':
                        if mask_index.any():
                            first_mask_pos = mask_index.nonzero(as_tuple=True)[0].min().item()
                            end_pos = min(first_mask_pos + num_to_transfer, block_len)
                            transfer_index[first_mask_pos:end_pos] = True
                    
                    elif 'low_confidence_static' in seq.remasking_strategy:
                        confidence = torch.where(mask_index, seq_x0_p, -np.inf)
                        # For dynamic, add threshold logic here if desired
                        _, top_indices = torch.topk(confidence, num_to_transfer)
                        transfer_index[top_indices] = True
                    
                    elif 'low_confidence_dynamic' in seq.remasking_strategy:
                        confidence = torch.where(mask_index, seq_x0_p, -np.inf)
                        transfer_index = torch.where(confidence > seq.dynamic_threshold, True, False)
                        if sum(transfer_index) < num_to_transfer:
                            _, top_indices = torch.topk(confidence, num_to_transfer)
                            transfer_index[top_indices] = True
                        num_to_transfer = transfer_index.sum().item() if transfer_index.sum().item() > 0 else num_to_transfer
                    elif 'entropy_bounded' in seq.remasking_strategy:
                        block_probs = probs[start_idx : start_idx + block_len] if not self.consistent_sampling_params else probs[start_idx : start_idx + block_len]
                        P = block_probs[mask_index]
                        entropies = -(P.clamp_min(EPS) * (P.clamp_min(EPS)).log()).sum(dim=-1)
                        ent_sorted, order = torch.sort(entropies, dim=0, descending=False)
                        cumsum = torch.cumsum(ent_sorted, dim=0)
                        k = torch.searchsorted(cumsum, torch.tensor(seq.eb_threshold, device=P.device), right=False).item()
                        if k == 0:
                            k = 1
                        selected_token_indices = mask_index.nonzero(as_tuple=True)[0][order[:k]]
                        transfer_index[selected_token_indices] = True
                        num_to_transfer = k

                    elif seq.remasking_strategy == 'highest_confidence':
                        confidence = torch.where(mask_index, seq_x0_p, -np.inf)
                        _, top_indices = torch.topk(confidence, num_to_transfer)
                        transfer_index[top_indices] = True

                    elif seq.remasking_strategy == 'anti_sequential':
                        if mask_index.any():
                            last_mask_pos = mask_index.nonzero(as_tuple=True)[0].max().item()
                            start_pos = max(last_mask_pos - num_to_transfer + 1, 0)
                            transfer_index[start_pos:last_mask_pos + 1] = True

                    elif seq.remasking_strategy == 'top2_margin':
                        block_probs = probs[start_idx : start_idx + block_len]
                        top2_vals, _ = block_probs.topk(2, dim=-1)  # (block_len, 2)
                        margin = top2_vals[:, 0] - top2_vals[:, 1]
                        margin = torch.where(mask_index, margin, torch.tensor(-torch.inf, device=margin.device))
                        _, top_indices = torch.topk(margin, num_to_transfer)
                        transfer_index[top_indices] = True

                    elif seq.remasking_strategy == 'least_entropy':
                        block_probs = probs[start_idx : start_idx + block_len]
                        P = block_probs[mask_index]
                        entropies = -(P.clamp_min(EPS) * (P.clamp_min(EPS)).log()).sum(dim=-1)
                        _, order = torch.sort(entropies, dim=0, descending=False)
                        k = min(num_to_transfer, len(order))
                        selected = mask_index.nonzero(as_tuple=True)[0][order[:k]]
                        transfer_index[selected] = True

                    # update - Tensors
                    seq.intermediate_block_tokens = torch.where(transfer_index, seq_x0, current_block_tensor)
                    seq.intermediate_block_tokens_entropy = torch.where(transfer_index, seq_entropies, seq.intermediate_block_tokens_entropy.to(logits.device))
                    
                    # track trajectory
                    if seq.block_trajectory is None: # Should not happen with new init but safe check
                         seq.block_trajectory = torch.zeros(block_len, dtype=torch.long, device=logits.device)
                    else:
                        seq.block_trajectory = seq.block_trajectory.to(logits.device)

                    if seq.block_logprobs is None:
                        seq.block_logprobs = torch.zeros(block_len, dtype=torch.float, device=logits.device)
                    else:
                        seq.block_logprobs = seq.block_logprobs.to(logits.device)
                        
                    if seq.block_entropies is None:
                        seq.block_entropies = torch.zeros(block_len, dtype=torch.float, device=logits.device)
                    else:
                        seq.block_entropies = seq.block_entropies.to(logits.device)
                        
                    first_time_global = seq.global_denoising_step + 1
                    
                    # Vectorized updates
                    seq.block_trajectory = torch.where(transfer_index & (seq.block_trajectory == 0), torch.tensor(first_time_global, device=logits.device), seq.block_trajectory)
                    seq.block_logprobs = torch.where(transfer_index, seq_x0_logp, seq.block_logprobs)
                    seq.block_entropies = torch.where(transfer_index, seq_entropies, seq.block_entropies)
                    
                    seq.current_denoising_step += 1
                    seq.global_denoising_step += 1
                    
                    # Check if block is fully denoised
                    is_fully_denoised = (self.mask_token_id not in seq.intermediate_block_tokens) or \
                                        (seq.current_denoising_step >= seq.denoising_steps)

                    if is_fully_denoised:
                        # Block is done, commit it and check if generation is finished
                        seq.status = SequenceStatus.FINISHED if seq.is_finished else SequenceStatus.SAVING
                    seq.num_to_transfer = num_to_transfer
                    
                elif seq.status == SequenceStatus.SAVING:
                    # If saving, commit the block and start a new one
                    seq.commit_block(seq.intermediate_block_tokens)
                    seq.num_to_transfer = 0
                    if not seq.is_finished:
                        seq.start_new_block()

                start_idx += seq.block_length
                
        # Filter out finished sequences from the running list
        finished_seqs = [seq for seq in self.running if seq.is_finished]
        self.running = [seq for seq in self.running if not seq.is_finished]
        for seq in finished_seqs:
            self.block_manager.deallocate(seq)
            
    def postprocess_unify(self, seqs: list[Sequence], logits: torch.Tensor, run_type: RunType,
                          chain_state: dict | None = None) -> list[Sequence]:
        if run_type == RunType.PREFILL:
            for seq in seqs:
                seq.num_cached_tokens = seq.num_prefill_tokens
                seq.status = SequenceStatus.DENOISING
            return []

        if run_type != RunType.DENOISE or not seqs:
            return []

        device = logits.device
        block_len = seqs[0].block_length

        # Separate SAVING sequences (need commit, skip sampling) from DENOISING (need sampling)
        saving_seq_indices = []
        denoising_seq_indices = []
        for i, seq in enumerate(seqs):
            if seq.status == SequenceStatus.SAVING:
                saving_seq_indices.append(i)
            else:
                denoising_seq_indices.append(i)

        # Handle SAVING sequences: commit blocks (forward already ran for KV cache update)
        if saving_seq_indices:
            saving_seqs = [seqs[i] for i in saving_seq_indices]
            s_tokens = torch.stack([s.intermediate_block_tokens for s in saving_seqs])
            s_traj = torch.stack([s.block_trajectory for s in saving_seqs])
            s_logp = torch.stack([s.block_logprobs for s in saving_seqs])
            s_ent = torch.stack([s.block_entropies for s in saving_seqs])
            # Two GPU→CPU syncs: one for int64 tensors, one for float tensors
            int_batch = torch.cat([s_tokens, s_traj], dim=1).cpu().tolist()
            float_batch = torch.cat([s_logp, s_ent], dim=1).cpu().tolist()
            for j, seq in enumerate(saving_seqs):
                seq._commit_block_from_cpu(
                    int_batch[j][:block_len], int_batch[j][block_len:],
                    float_batch[j][:block_len], float_batch[j][block_len:]
                )
                seq.num_to_transfer = 0
                if not seq.is_finished:
                    seq.start_new_block()

        denoising_seqs = [seqs[i] for i in denoising_seq_indices]
        d_batch_size = len(denoising_seqs)

        if denoising_seqs:
            _s = denoising_seqs

            # --- Build batch_current_tokens: use chain_state if available ---
            if chain_state is not None:
                batch_current_tokens = chain_state['tokens']
                batch_logprobs = chain_state['logprobs']
                batch_entropies = chain_state['entropies']
                batch_trajectory = chain_state['trajectory']
                cs_denoising_steps = chain_state['denoising_steps']
                cs_global_steps = chain_state['global_steps']
            else:
                batch_current_tokens = torch.stack([s.intermediate_block_tokens.to(device, non_blocking=True) for s in _s])
                batch_logprobs = torch.stack([s.block_logprobs.to(device, non_blocking=True) for s in _s])
                batch_entropies = torch.stack([s.block_entropies.to(device, non_blocking=True) for s in _s])
                batch_trajectory = torch.stack([s.block_trajectory.to(device, non_blocking=True) for s in _s])
                cs_denoising_steps = torch.tensor(
                    [seq.current_denoising_step for seq in _s], device=device
                )
                cs_global_steps = torch.tensor(
                    [seq.global_denoising_step for seq in _s], device=device
                )

            mask_token_mask = (batch_current_tokens == self.mask_token_id)

            # --- Softmax + Sampling (sparse: only masked positions) ---
            flat_mask = mask_token_mask.view(-1)

            d_logits_3d = logits.view(d_batch_size, block_len, -1)
            d_logits = d_logits_3d.reshape(-1, d_logits_3d.shape[-1])
            vocab_size = d_logits.shape[-1]

            # Predict all_masked from CPU-side state to avoid GPU→CPU sync
            if chain_state is not None:
                cpu_step = chain_state.get('cpu_step', None)
                all_masked = (cpu_step is not None and cpu_step == 0)
            else:
                # First step: all positions start masked
                all_masked = all(seq.current_denoising_step == 0 for seq in _s)

            # Fast path for position-based strategies (sequential / anti_sequential):
            # they commit a deterministic position per step (leftmost/rightmost mask) and
            # do not use confidence at other positions. We can sample only the target row
            # per sequence — bs rows instead of bs*block_len.
            _strategy_early = _s[0].remasking_strategy
            _schedule_all_ones = all(n == 1 for n in _s[0].num_transfer_tokens_per_step)
            _use_pos_fast_path = (
                _strategy_early in ("sequential", "anti_sequential")
                and _schedule_all_ones
                and d_batch_size >= 8  # torch.multinomial path needs ≥8 rows for parity
                and not any(seq.repetition_penalty != 1.0 and seq.num_tokens > seq.max_tokens // 2 for seq in _s)
            )

            if _use_pos_fast_path:
                _row_arange = torch.arange(d_batch_size, device=device)
                mask_available_fp = mask_token_mask.any(dim=1)
                if _strategy_early == "sequential":
                    target_positions = torch.where(
                        mask_available_fp,
                        torch.argmax(mask_token_mask.int(), dim=1),
                        torch.zeros(d_batch_size, device=device, dtype=torch.long),
                    )
                else:  # anti_sequential
                    reversed_mask = mask_token_mask.flip(dims=[1])
                    last_mask_pos_rev = torch.where(
                        mask_available_fp,
                        torch.argmax(reversed_mask.int(), dim=1),
                        torch.zeros(d_batch_size, device=device, dtype=torch.long),
                    )
                    target_positions = (block_len - 1 - last_mask_pos_rev).clamp_min(0)

                target_linear_idx = _row_arange * block_len + target_positions  # (bs,)
                target_logits = d_logits[target_linear_idx]  # (bs, vocab)

                if target_logits.shape[0] < 8:
                    target_probs = torch.softmax(
                        target_logits.float() / _s[0].temperature, dim=-1
                    ).to(target_logits.dtype)
                else:
                    target_probs = self.sample_pipe(target_logits, temperature=_s[0].temperature)

                target_samples = self._sample_tokens(target_probs, _s[0].top_k, _s[0].top_p)
                target_probs_at_sample = torch.gather(
                    target_probs, 1, target_samples.unsqueeze(-1)
                ).squeeze(-1)

                # Build full-shape tensors; only target positions are populated.
                batch_seq_x0 = batch_current_tokens.clone()
                batch_seq_x0[_row_arange, target_positions] = target_samples
                batch_seq_x0_p = torch.zeros(d_batch_size, block_len, device=device,
                                             dtype=target_probs_at_sample.dtype)
                batch_seq_x0_p[_row_arange, target_positions] = target_probs_at_sample
                batch_seq_x0_logp = torch.zeros_like(batch_seq_x0_p)
                batch_seq_x0_logp[_row_arange, target_positions] = torch.log(
                    target_probs_at_sample.clamp_min(EPS)
                )
                # Entropy at target only (cheap, bs rows)
                _target_log_p = target_probs.clamp_min(EPS).log()
                _target_entropy = -(target_probs * _target_log_p).sum(dim=-1)
                _pos_fp_entropies = torch.zeros(d_batch_size, block_len, device=device,
                                                dtype=target_probs_at_sample.dtype)
                _pos_fp_entropies[_row_arange, target_positions] = _target_entropy.to(
                    target_probs_at_sample.dtype
                )
                _sparse_probs = None
                _masked_indices = None
                use_dense = False  # skip dense/sparse branches
                _pos_fp_done = True
            else:
                _pos_fp_done = False

            # Use dense path when all masked or too few masked positions for flashinfer kernel
            if _pos_fp_done:
                pass  # already sampled via fast path
            elif all_masked:
                use_dense = True
            elif d_batch_size * block_len < 8:
                use_dense = True  # total positions too few for flashinfer
            else:
                # Predict masked count from CPU-side state to avoid GPU sync
                if chain_state is not None and chain_state.get('cpu_step') is not None:
                    predicted_masked = d_batch_size * max(1, block_len - chain_state['cpu_step'])
                else:
                    # Fallback: estimate from sequence state (only for small batches)
                    min_remaining = min((block_len - seq.current_denoising_step) for seq in _s)
                    predicted_masked = d_batch_size * max(1, min_remaining)
                use_dense = predicted_masked < 8

            # Pre-compute per-position temperature vector when pos_temp_slope != 0.
            # T_i = T_base * (1 + slope * i / (L-1)). slope > 0 → warmer rightmost.
            _pts = getattr(_s[0], 'pos_temp_slope', 0.0)
            _use_pos_temp = (_pts != 0.0 and block_len > 1)
            if _use_pos_temp:
                _T_base = _s[0].temperature
                _pos_frac = torch.arange(block_len, device=device, dtype=torch.float32) / (block_len - 1)
                _T_vec = _T_base * (1.0 + _pts * _pos_frac)  # (block_len,)
                _T_vec = _T_vec.clamp_min(0.01)  # safety

            if _pos_fp_done:
                pass  # fast path already populated batch_seq_x0*, skip dense/sparse
            elif use_dense:
                # Dense path: all positions masked (first denoising step)
                if _use_pos_temp:
                    # Per-position temperature: reshape → scale → softmax
                    logits_3d = d_logits.view(d_batch_size, block_len, -1).float()
                    scaled = logits_3d / _T_vec.view(1, block_len, 1)
                    probs = torch.softmax(scaled, dim=-1).to(d_logits.dtype).view(d_batch_size, block_len, -1)
                elif d_logits.shape[0] < 8:
                    probs = torch.softmax(d_logits.float() / _s[0].temperature, dim=-1).to(d_logits.dtype).view(d_batch_size, block_len, -1)
                else:
                    probs = self.sample_pipe(d_logits, temperature=_s[0].temperature).view(d_batch_size, block_len, -1)
                self.apply_repetition_penalty(probs, _s)
                flat_probs = probs.view(-1, vocab_size)
                flat_samples = self._sample_tokens(flat_probs, _s[0].top_k, _s[0].top_p)
                batch_seq_x0 = flat_samples.view(d_batch_size, block_len)
                batch_seq_x0_p = torch.gather(flat_probs, 1, flat_samples.unsqueeze(-1)).view(d_batch_size, block_len)
                batch_seq_x0_logp = torch.log(batch_seq_x0_p.clamp_min(EPS))
            else:
                # Sparse path: only process masked positions
                masked_indices = flat_mask.nonzero(as_tuple=True)[0]
                sparse_logits = d_logits[masked_indices]
                if _use_pos_temp:
                    # Map each sparse row to its block position and apply position T
                    pos_idx = (masked_indices % block_len).long()
                    T_per_row = _T_vec[pos_idx].unsqueeze(1)  # (num_masked, 1)
                    sparse_probs = torch.softmax(sparse_logits.float() / T_per_row, dim=-1).to(sparse_logits.dtype)
                elif sparse_logits.shape[0] < 8:
                    sparse_probs = torch.softmax(sparse_logits.float() / _s[0].temperature, dim=-1).to(sparse_logits.dtype)
                else:
                    sparse_probs = self.sample_pipe(sparse_logits, temperature=_s[0].temperature)

                # Apply repetition penalty on sparse probs (need per-sequence mapping)
                if any(seq.repetition_penalty != 1.0 and seq.num_tokens > seq.max_tokens // 2 for seq in _s):
                    seq_indices = masked_indices // block_len  # which sequence each masked pos belongs to
                    for i, seq in enumerate(_s):
                        if seq.repetition_penalty == 1.0 or seq.num_tokens <= seq.max_tokens // 2:
                            continue
                        row_indices = (seq_indices == i).nonzero(as_tuple=True)[0]
                        if row_indices.shape[0] == 0:
                            continue
                        seen_tokens = list(seq.token_counts.keys())
                        if seen_tokens:
                            sparse_probs[row_indices[:, None], torch.tensor(seen_tokens, device=device)] /= seq.repetition_penalty
                            sparse_probs[row_indices] /= sparse_probs[row_indices].sum(dim=-1, keepdim=True)

                sparse_samples = self._sample_tokens(sparse_probs, _s[0].top_k, _s[0].top_p)
                sparse_x0_p = torch.gather(sparse_probs, 1, sparse_samples.unsqueeze(-1)).squeeze(-1)

                # Scatter back to full (batch_size, block_len) shape
                batch_seq_x0 = batch_current_tokens.clone()
                batch_seq_x0.view(-1)[masked_indices] = sparse_samples
                batch_seq_x0_p = torch.zeros(d_batch_size, block_len, device=device, dtype=sparse_x0_p.dtype)
                batch_seq_x0_p.view(-1)[masked_indices] = sparse_x0_p
                batch_seq_x0_logp = torch.zeros(d_batch_size, block_len, device=device, dtype=sparse_x0_p.dtype)
                batch_seq_x0_logp.view(-1)[masked_indices] = torch.log(sparse_x0_p.clamp_min(EPS))
                # Store sparse_probs for entropy computation later
                _sparse_probs = sparse_probs
                _masked_indices = masked_indices

            batch_global_step_plus_1 = (cs_global_steps + 1).unsqueeze(1)

            # Compute num_to_transfer from GPU-side denoising steps
            if chain_state is not None:
                # All sequences have same schedule (consistent_sampling_params)
                schedule = _s[0].num_transfer_tokens_per_step  # e.g. [1, 1, 1, 1]
                if not hasattr(self, '_transfer_schedule') or len(self._transfer_schedule) != len(schedule):
                    self._transfer_schedule = torch.tensor(schedule, device=device, dtype=torch.long)
                batch_num_to_transfer = self._transfer_schedule[cs_denoising_steps.clamp(max=len(schedule)-1)]
            else:
                num_to_transfer_list = [
                    seq.num_transfer_tokens_per_step[seq.current_denoising_step] for seq in _s
                ]
                batch_num_to_transfer = torch.tensor(num_to_transfer_list, device=device, dtype=torch.long)

            mask_available = mask_token_mask.any(dim=1)
            effective_num_to_transfer = torch.where(
                mask_available, batch_num_to_transfer, torch.zeros_like(batch_num_to_transfer)
            )

            strategy = _s[0].remasking_strategy
            transfer_index = torch.zeros((d_batch_size, block_len), dtype=torch.bool, device=device)
            needs_entropy = (strategy == "entropy_bounded")

            if strategy == "sequential":
                range_tensor = torch.arange(block_len, device=device).unsqueeze(0)
                first_mask_pos = torch.where(
                    mask_available,
                    torch.argmax(mask_token_mask.int(), dim=1),
                    torch.full_like(effective_num_to_transfer, block_len),
                )
                start = first_mask_pos.unsqueeze(1)
                end = (start + effective_num_to_transfer.unsqueeze(1)).clamp_max(block_len)
                seq_transfer_index = (range_tensor >= start) & (range_tensor < end)
                transfer_index = seq_transfer_index & mask_token_mask

            elif strategy == "low_confidence_static":
                confidence = torch.where(
                    mask_token_mask, batch_seq_x0_p, torch.full_like(batch_seq_x0_p, -torch.inf)
                )
                # Use block_len as k (small, avoids GPU→CPU sync for .max().item())
                _, top_indices = torch.topk(confidence, k=block_len, dim=1)
                k_mask = torch.arange(block_len, device=device).unsqueeze(0) < effective_num_to_transfer.unsqueeze(1)
                transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
                transfer_index.scatter_(1, top_indices, k_mask)
                transfer_index &= mask_token_mask

            elif strategy == "low_confidence_dynamic":
                confidence = torch.where(
                    mask_token_mask, batch_seq_x0_p, torch.full_like(batch_seq_x0_p, -torch.inf)
                )
                dyn_transfer_index = confidence > _s[0].dynamic_threshold
                dyn_transfer_index &= mask_token_mask
                num_transferred_dyn = dyn_transfer_index.sum(dim=1)
                needs_fallback = (num_transferred_dyn < effective_num_to_transfer) & mask_available
                # Always compute fallback (cheap for small block_len) to avoid GPU sync
                range_tensor = torch.arange(block_len, device=device).unsqueeze(0)
                first_mask_pos = torch.where(
                    mask_available,
                    torch.argmax(mask_token_mask.int(), dim=1),
                    torch.full_like(effective_num_to_transfer, block_len),
                ).unsqueeze(1)
                end = (first_mask_pos + effective_num_to_transfer.unsqueeze(1)).clamp_max(block_len)
                fallback_index = (range_tensor >= first_mask_pos) & (range_tensor < end) & mask_token_mask
                # Use fallback where dynamic threshold didn't transfer enough
                transfer_index = torch.where(
                    needs_fallback.unsqueeze(1), fallback_index, dyn_transfer_index
                )

            elif strategy == "entropy_bounded":
                # Entropy needed: compute it for masked positions only
                needs_entropy = True
                pass  # handled below after entropy computation

            elif strategy == "random":
                B, L = mask_token_mask.shape
                scores = torch.rand((B, L), device=device)
                scores = scores.masked_fill(~mask_token_mask, -1.0)
                _, top_indices = scores.topk(block_len, dim=-1)
                k_mask = torch.arange(block_len, device=device).unsqueeze(0) < effective_num_to_transfer.unsqueeze(1)
                transfer_index = torch.zeros_like(mask_token_mask, dtype=torch.bool)
                transfer_index.scatter_(1, top_indices, k_mask)
                transfer_index &= mask_token_mask

            elif strategy == "highest_confidence":
                # Opposite of low_confidence_static: commit MOST confident first
                confidence = torch.where(
                    mask_token_mask, batch_seq_x0_p, torch.full_like(batch_seq_x0_p, -torch.inf)
                )
                _, top_indices = torch.topk(confidence, k=block_len, dim=1)
                k_mask = torch.arange(block_len, device=device).unsqueeze(0) < effective_num_to_transfer.unsqueeze(1)
                transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
                transfer_index.scatter_(1, top_indices, k_mask)
                transfer_index &= mask_token_mask

            elif strategy == "anti_sequential":
                # Right-to-left: unmask last masked position first
                range_tensor = torch.arange(block_len, device=device).unsqueeze(0)
                # Find last masked position (rightmost)
                reversed_mask = mask_token_mask.flip(dims=[1])
                last_mask_pos_rev = torch.where(
                    mask_available,
                    torch.argmax(reversed_mask.int(), dim=1),
                    torch.full_like(effective_num_to_transfer, block_len),
                )
                last_mask_pos = block_len - 1 - last_mask_pos_rev
                end = (last_mask_pos + 1).unsqueeze(1)
                start = (end - effective_num_to_transfer.unsqueeze(1)).clamp_min(0)
                seq_transfer_index = (range_tensor >= start) & (range_tensor < end)
                transfer_index = seq_transfer_index & mask_token_mask

            elif strategy == "top2_margin":
                # Commit position with largest gap between top-1 and top-2 probabilities
                if use_dense:
                    top2_vals, _ = probs.topk(2, dim=-1)  # (batch, block_len, 2)
                    margin = top2_vals[:, :, 0] - top2_vals[:, :, 1]  # (batch, block_len)
                else:
                    sparse_top2, _ = _sparse_probs.topk(2, dim=-1)  # (num_masked, 2)
                    sparse_margin = sparse_top2[:, 0] - sparse_top2[:, 1]
                    margin = torch.zeros(d_batch_size, block_len, device=device, dtype=sparse_margin.dtype)
                    margin.view(-1)[_masked_indices] = sparse_margin
                margin = torch.where(
                    mask_token_mask, margin, torch.full_like(margin, -torch.inf)
                )
                _, top_indices = torch.topk(margin, k=block_len, dim=1)
                k_mask = torch.arange(block_len, device=device).unsqueeze(0) < effective_num_to_transfer.unsqueeze(1)
                transfer_index = torch.zeros_like(margin, dtype=torch.bool)
                transfer_index.scatter_(1, top_indices, k_mask)
                transfer_index &= mask_token_mask

            elif strategy == "least_entropy":
                # Commit position with lowest entropy first (most certain)
                needs_entropy = True
                pass  # handled below after entropy computation, similar to entropy_bounded

            elif strategy == "causal_waterfall":
                # Commit a contiguous leftmost prefix of MASKED positions whose
                # per-position P(argmax) each exceeds cw_tau (NOT cumulative).
                # Only ever commits tokens the model is *independently* confident
                # about, avoiding the "commit from broad marginals" pathology.
                # At T=1.0 this usually fires only on syntactically-forced tokens
                # (delimiters, digits after "="), matching the joint factorization
                # where those positions are near-independent of others in the block.
                if use_dense:
                    max_probs = probs.max(dim=-1).values
                    argmax_tokens = probs.argmax(dim=-1)
                else:
                    sparse_top = _sparse_probs.max(dim=-1)
                    max_probs = torch.zeros(d_batch_size, block_len, device=device,
                                            dtype=_sparse_probs.dtype)
                    argmax_tokens = torch.full((d_batch_size, block_len), -1,
                                               device=device, dtype=torch.long)
                    max_probs.view(-1)[_masked_indices] = sparse_top.values
                    argmax_tokens.view(-1)[_masked_indices] = sparse_top.indices

                # EOS safety: suppress commitments whose argmax is a stop token.
                # The eval uses stop_words = {151645, 151643, 151670} for SDAR.
                stop_ids = torch.tensor(list(_s[0].stop_words or []),
                                        device=device, dtype=argmax_tokens.dtype)
                if stop_ids.numel() > 0:
                    is_stop = (argmax_tokens.unsqueeze(-1) == stop_ids).any(dim=-1)
                else:
                    is_stop = torch.zeros_like(mask_token_mask)

                # Per-position confident gate (NOT cumulative — that was the bug).
                # cw_tau default −0.35 ⇒ require p_max > 0.70 per committed position.
                floor = float(torch.tensor(_s[0].cw_tau, device=device).exp().item())
                confident = (max_probs > floor) & mask_token_mask & (~is_stop)

                # Contiguous leftmost prefix only
                prefix = (torch.cumprod(confident.int(), dim=1) > 0)
                transfer_index = prefix & mask_token_mask

                # Always make progress: commit leftmost mask if nothing qualifies
                no_commit = (~transfer_index.any(dim=1)) & mask_available
                if no_commit.any():
                    first_mp = torch.argmax(mask_token_mask.int(), dim=1)
                    rows = torch.arange(d_batch_size, device=device)
                    fallback = torch.zeros_like(transfer_index)
                    fallback[rows, first_mp] = True
                    fallback &= mask_token_mask
                    transfer_index = torch.where(
                        no_commit.unsqueeze(1), fallback, transfer_index
                    )

            elif strategy == "logit_salience":
                # Commit only the leftmost masked position whose top logit is an
                # extreme outlier (z > lsz_threshold, default 10). Adds a second
                # position iff it ALSO passes the outlier bar AND is adjacent.
                # Scale-invariant; fires only for "crystallized" tokens at T=1.0.
                mu = d_logits.mean(dim=-1)
                sigma = d_logits.std(dim=-1).clamp_min(1e-6)
                max_logits = d_logits.max(dim=-1).values
                argmax_tokens_flat = d_logits.argmax(dim=-1)
                z_all = ((max_logits - mu) / sigma).view(d_batch_size, block_len)
                argmax_tokens = argmax_tokens_flat.view(d_batch_size, block_len)

                # EOS safety
                stop_ids = torch.tensor(list(_s[0].stop_words or []),
                                        device=device, dtype=argmax_tokens.dtype)
                if stop_ids.numel() > 0:
                    is_stop = (argmax_tokens.unsqueeze(-1) == stop_ids).any(dim=-1)
                else:
                    is_stop = torch.zeros_like(mask_token_mask)

                salient = (z_all > _s[0].lsz_threshold) & mask_token_mask & (~is_stop)
                prefix = (torch.cumprod(salient.int(), dim=1) > 0)
                transfer_index = prefix & mask_token_mask

                no_commit = (~transfer_index.any(dim=1)) & mask_available
                if no_commit.any():
                    first_mp = torch.argmax(mask_token_mask.int(), dim=1)
                    rows = torch.arange(d_batch_size, device=device)
                    fallback = torch.zeros_like(transfer_index)
                    fallback[rows, first_mp] = True
                    fallback &= mask_token_mask
                    transfer_index = torch.where(
                        no_commit.unsqueeze(1), fallback, transfer_index
                    )

            elif strategy == "consensus_commit":
                # K-sample majority-vote multi-commit. For each masked position,
                # draw K=4 independent multinomial samples from the softmax and
                # commit IF ≥3 samples agree on the same token (committed value).
                # Rationale: this is a Monte-Carlo proxy for joint marginalization —
                # positions where multiple samples converge are "self-determined"
                # (the distribution is sharply peaked around the modal token), so
                # committing is safe even when committing multiple positions in
                # parallel. Positions with diffuse distributions will have low
                # consensus and fall back to leftmost.
                K = 4
                if use_dense:
                    fp = flat_probs  # (bs*L, vocab)
                    rows_all = fp.shape[0]
                else:
                    fp = _sparse_probs  # (num_masked, vocab)
                    rows_all = fp.shape[0]

                if rows_all == 0:
                    # Nothing masked (shouldn't happen but be safe)
                    consensus_tokens_full = batch_current_tokens.clone()
                    has_majority_full = torch.zeros_like(mask_token_mask, dtype=torch.bool)
                else:
                    # K samples, shape (K, N)
                    samples_k = torch.stack(
                        [torch.multinomial(fp, 1).squeeze(-1) for _ in range(K)], dim=0
                    )
                    # Sort-based mode check for K=4:
                    # majority exists iff sorted[0..2] all equal OR sorted[1..3] all equal.
                    sorted_k = torch.sort(samples_k, dim=0).values  # (K, N)
                    case_A = (sorted_k[0] == sorted_k[1]) & (sorted_k[1] == sorted_k[2])
                    case_B = (sorted_k[1] == sorted_k[2]) & (sorted_k[2] == sorted_k[3])
                    has_majority = case_A | case_B
                    mode_val = torch.where(case_A, sorted_k[0], sorted_k[1])  # (N,)

                    if use_dense:
                        has_majority_full = has_majority.view(d_batch_size, block_len)
                        consensus_tokens_full = batch_current_tokens.clone()
                        consensus_tokens_full.view(-1)[has_majority.nonzero(as_tuple=True)[0]] = (
                            mode_val[has_majority]
                        )
                    else:
                        has_majority_full = torch.zeros(d_batch_size, block_len, device=device, dtype=torch.bool)
                        consensus_tokens_full = batch_current_tokens.clone()
                        masked_lin_idx = _masked_indices
                        majority_idx_within = has_majority.nonzero(as_tuple=True)[0]
                        majority_lin_idx = masked_lin_idx[majority_idx_within]
                        has_majority_full.view(-1)[majority_lin_idx] = True
                        consensus_tokens_full.view(-1)[majority_lin_idx] = mode_val[majority_idx_within]

                # EOS safety: never commit a stop token via consensus
                stop_ids = torch.tensor(list(_s[0].stop_words or []),
                                        device=device, dtype=consensus_tokens_full.dtype)
                if stop_ids.numel() > 0:
                    is_stop = (consensus_tokens_full.unsqueeze(-1) == stop_ids).any(dim=-1)
                    has_majority_full = has_majority_full & (~is_stop)

                _qualifies_mask = has_majority_full & mask_token_mask
                # Override committed tokens with the consensus (not the first sample)
                batch_seq_x0 = torch.where(_qualifies_mask, consensus_tokens_full, batch_seq_x0)
                # Fallback: leftmost if nothing consensus-qualified
                num_transferred = _qualifies_mask.sum(dim=1)
                needs_fallback = (num_transferred < 1) & mask_available
                if needs_fallback.any():
                    first_mp = torch.argmax(mask_token_mask.int(), dim=1)
                    rows = torch.arange(d_batch_size, device=device)
                    fallback = torch.zeros_like(_qualifies_mask)
                    fallback[rows, first_mp] = True
                    fallback &= mask_token_mask
                    transfer_index = torch.where(
                        needs_fallback.unsqueeze(1), fallback, _qualifies_mask
                    )
                else:
                    transfer_index = _qualifies_mask

            elif strategy == "dynamic_pmax":
                # Like low_confidence_dynamic but thresholds on P(argmax) per
                # position instead of P(sampled-token). At T=1.0 multinomial
                # sampling often picks non-argmax tokens — their probabilities
                # can be much lower than the position's true peakedness, so
                # vanilla dynamic under-fires. Using P(argmax) correctly
                # measures position confidence regardless of sample noise, so
                # multi-token commit triggers more often on genuinely peaked
                # positions — which is exactly the "independence-safe"
                # condition for joint factorization.
                if use_dense:
                    max_probs = probs.max(dim=-1).values
                    argmax_tokens = probs.argmax(dim=-1)
                else:
                    sparse_top = _sparse_probs.max(dim=-1)
                    max_probs = torch.zeros(d_batch_size, block_len, device=device,
                                            dtype=_sparse_probs.dtype)
                    argmax_tokens = torch.full((d_batch_size, block_len), -1,
                                               device=device, dtype=torch.long)
                    max_probs.view(-1)[_masked_indices] = sparse_top.values
                    argmax_tokens.view(-1)[_masked_indices] = sparse_top.indices

                # Replace the committed position's token with the argmax
                # (not the multinomial sample). This ensures we commit the
                # high-confidence token the threshold was measuring.
                _qualifies_mask = (max_probs > _s[0].dynamic_threshold) & mask_token_mask

                # EOS safety gate (same as other novel strategies): never
                # commit a stop token from this strategy.
                stop_ids = torch.tensor(list(_s[0].stop_words or []),
                                        device=device, dtype=argmax_tokens.dtype)
                if stop_ids.numel() > 0:
                    is_stop = (argmax_tokens.unsqueeze(-1) == stop_ids).any(dim=-1)
                    _qualifies_mask = _qualifies_mask & (~is_stop)

                # Override the sampled tokens with argmax at committed positions
                batch_seq_x0 = torch.where(_qualifies_mask, argmax_tokens, batch_seq_x0)
                # Also update the per-token probability to P(argmax) for the
                # committed positions (so logprob tracking reflects what we committed).
                batch_seq_x0_p = torch.where(_qualifies_mask, max_probs, batch_seq_x0_p)
                batch_seq_x0_logp = torch.where(
                    _qualifies_mask, max_probs.clamp_min(EPS).log(), batch_seq_x0_logp
                )

                # Dynamic-like: commit all qualifying (may be non-contiguous)
                num_transferred = _qualifies_mask.sum(dim=1)
                needs_fallback = (num_transferred < 1) & mask_available
                if needs_fallback.any():
                    first_mp = torch.argmax(mask_token_mask.int(), dim=1)
                    rows = torch.arange(d_batch_size, device=device)
                    fallback = torch.zeros_like(_qualifies_mask)
                    fallback[rows, first_mp] = True
                    fallback &= mask_token_mask
                    transfer_index = torch.where(
                        needs_fallback.unsqueeze(1), fallback, _qualifies_mask
                    )
                else:
                    transfer_index = _qualifies_mask

            elif strategy == "confidence_chain":
                # Chain-aware multi-commit: the leftmost position has the most
                # context (the committed prefix) and is easiest to determine, so
                # accept it at a LOWER confidence bar. Rightmost positions are
                # most exposed to uncommitted neighbors so require HIGHER
                # confidence. Implements a position-dependent threshold
                # schedule: thr(i) = base + slope * i.
                # Defaults: base=0.55 at i=0, thr=0.80 at i=3 (slope=0.083/pos)
                if use_dense:
                    max_probs = probs.max(dim=-1).values
                    argmax_tokens = probs.argmax(dim=-1)
                else:
                    sparse_top = _sparse_probs.max(dim=-1)
                    max_probs = torch.zeros(d_batch_size, block_len, device=device,
                                            dtype=_sparse_probs.dtype)
                    argmax_tokens = torch.full((d_batch_size, block_len), -1,
                                               device=device, dtype=torch.long)
                    max_probs.view(-1)[_masked_indices] = sparse_top.values
                    argmax_tokens.view(-1)[_masked_indices] = sparse_top.indices

                stop_ids = torch.tensor(list(_s[0].stop_words or []),
                                        device=device, dtype=argmax_tokens.dtype)
                if stop_ids.numel() > 0:
                    is_stop = (argmax_tokens.unsqueeze(-1) == stop_ids).any(dim=-1)
                else:
                    is_stop = torch.zeros_like(mask_token_mask)

                # Per-position threshold schedule (increasing left→right)
                base = 0.55
                slope = (0.85 - base) / max(1, block_len - 1)
                thr = torch.arange(block_len, device=device, dtype=max_probs.dtype) * slope + base
                qualifies = (
                    (max_probs > thr.unsqueeze(0))
                    & mask_token_mask
                    & (~is_stop)
                )
                # Contiguous leftmost prefix: if pos k fails, pos k+1 and beyond fail too
                prefix = (torch.cumprod(qualifies.int(), dim=1) > 0)
                transfer_index = prefix & mask_token_mask

                no_commit = (~transfer_index.any(dim=1)) & mask_available
                if no_commit.any():
                    first_mp = torch.argmax(mask_token_mask.int(), dim=1)
                    rows = torch.arange(d_batch_size, device=device)
                    fallback = torch.zeros_like(transfer_index)
                    fallback[rows, first_mp] = True
                    fallback &= mask_token_mask
                    transfer_index = torch.where(
                        no_commit.unsqueeze(1), fallback, transfer_index
                    )

            elif strategy == "dynamic_contiguous":
                # Same 0.9 threshold as low_confidence_dynamic but commits only a
                # contiguous LEFTMOST prefix of qualifying positions. Strict
                # left-bias preserves causality: position k commits only if 0..k-1
                # also pass or are already committed. Guards against "island of
                # confidence" where pos 3 passes but pos 1 is a guess.
                if use_dense:
                    max_probs = probs.max(dim=-1).values
                    argmax_tokens = probs.argmax(dim=-1)
                else:
                    sparse_top = _sparse_probs.max(dim=-1)
                    max_probs = torch.zeros(d_batch_size, block_len, device=device,
                                            dtype=_sparse_probs.dtype)
                    argmax_tokens = torch.full((d_batch_size, block_len), -1,
                                               device=device, dtype=torch.long)
                    max_probs.view(-1)[_masked_indices] = sparse_top.values
                    argmax_tokens.view(-1)[_masked_indices] = sparse_top.indices

                stop_ids = torch.tensor(list(_s[0].stop_words or []),
                                        device=device, dtype=argmax_tokens.dtype)
                if stop_ids.numel() > 0:
                    is_stop = (argmax_tokens.unsqueeze(-1) == stop_ids).any(dim=-1)
                else:
                    is_stop = torch.zeros_like(mask_token_mask)

                qualifies = (
                    (max_probs > _s[0].dynamic_threshold)
                    & mask_token_mask
                    & (~is_stop)
                )
                prefix = (torch.cumprod(qualifies.int(), dim=1) > 0)
                transfer_index = prefix & mask_token_mask
                no_commit = (~transfer_index.any(dim=1)) & mask_available
                if no_commit.any():
                    first_mp = torch.argmax(mask_token_mask.int(), dim=1)
                    rows = torch.arange(d_batch_size, device=device)
                    fallback = torch.zeros_like(transfer_index)
                    fallback[rows, first_mp] = True
                    fallback &= mask_token_mask
                    transfer_index = torch.where(
                        no_commit.unsqueeze(1), fallback, transfer_index
                    )

            elif strategy == "relative_top":
                # Commit leftmost contiguous prefix of positions whose P(argmax)
                # is at least rel_top_alpha × (sequence-wise max P(argmax)) AND
                # above an absolute floor. Adaptive to distribution scale.
                if use_dense:
                    max_probs = probs.max(dim=-1).values
                    argmax_tokens = probs.argmax(dim=-1)
                else:
                    sparse_top = _sparse_probs.max(dim=-1)
                    max_probs = torch.zeros(d_batch_size, block_len, device=device,
                                            dtype=_sparse_probs.dtype)
                    argmax_tokens = torch.full((d_batch_size, block_len), -1,
                                               device=device, dtype=torch.long)
                    max_probs.view(-1)[_masked_indices] = sparse_top.values
                    argmax_tokens.view(-1)[_masked_indices] = sparse_top.indices

                stop_ids = torch.tensor(list(_s[0].stop_words or []),
                                        device=device, dtype=argmax_tokens.dtype)
                if stop_ids.numel() > 0:
                    is_stop = (argmax_tokens.unsqueeze(-1) == stop_ids).any(dim=-1)
                else:
                    is_stop = torch.zeros_like(mask_token_mask)

                masked_mp = torch.where(
                    mask_token_mask, max_probs, torch.full_like(max_probs, -1.0)
                )
                seq_max = masked_mp.max(dim=1, keepdim=True).values.clamp_min(EPS)
                # Require relative ratio AND absolute floor of 0.5
                qualifies = (
                    (max_probs >= _s[0].rel_top_alpha * seq_max)
                    & (max_probs > 0.5)
                    & mask_token_mask
                    & (~is_stop)
                )
                prefix = (torch.cumprod(qualifies.int(), dim=1) > 0)
                transfer_index = prefix & mask_token_mask
                no_commit = (~transfer_index.any(dim=1)) & mask_available
                if no_commit.any():
                    first_mp = torch.argmax(mask_token_mask.int(), dim=1)
                    rows = torch.arange(d_batch_size, device=device)
                    fallback = torch.zeros_like(transfer_index)
                    fallback[rows, first_mp] = True
                    fallback &= mask_token_mask
                    transfer_index = torch.where(
                        no_commit.unsqueeze(1), fallback, transfer_index
                    )

            else:
                raise ValueError(f"Unsupported remasking strategy: {strategy}")

            # --- Deferred entropy: only compute for transferred positions ---
            final_transfer_index = transfer_index & mask_token_mask

            # Lazy entropy: skip computation in chain intermediate steps for non-entropy strategies
            # Entropy is only used for: (1) entropy_bounded strategy, (2) recording in committed blocks
            # For non-entropy strategies, use zeros during chain and compute at SAVING step
            can_skip_entropy = (not needs_entropy and chain_state is not None and
                                chain_state.get('cpu_step', None) is not None and
                                chain_state['cpu_step'] + 1 < _s[0].denoising_steps)

            if needs_entropy and strategy == "entropy_bounded":
                # entropy_bounded needs entropy BEFORE transfer_index is decided
                if use_dense:
                    log_probs_full = probs.clamp_min(EPS).log()
                    entropies_all = -(probs * log_probs_full).sum(dim=-1)
                else:
                    sparse_log_p = _sparse_probs.clamp_min(EPS).log()
                    sparse_ent = -(_sparse_probs * sparse_log_p).sum(dim=-1)
                    entropies_all = torch.zeros(d_batch_size, block_len, device=device, dtype=sparse_ent.dtype)
                    entropies_all.view(-1)[_masked_indices] = sparse_ent

                # Now do entropy_bounded strategy
                masked_entropies = torch.where(
                    mask_token_mask, entropies_all, torch.full_like(entropies_all, torch.inf)
                )
                ent_sorted, order = torch.sort(masked_entropies, dim=1)
                ent_sorted_masked = torch.where(torch.isfinite(ent_sorted), ent_sorted, torch.zeros_like(ent_sorted))
                cumsum = torch.cumsum(ent_sorted_masked, dim=1)
                thresholds = torch.full((d_batch_size, 1), _s[0].eb_threshold, device=device)
                k_tensor = torch.searchsorted(cumsum, thresholds, right=False)
                k_tensor = torch.where(
                    mask_available.unsqueeze(1), k_tensor.clamp_min(1), torch.zeros_like(k_tensor)
                ).squeeze(1)
                # Use block_len as k (small, avoids GPU→CPU sync)
                k_mask = torch.arange(block_len, device=device).unsqueeze(0) < k_tensor.unsqueeze(1)
                transfer_index = torch.zeros_like(mask_token_mask)
                transfer_index.scatter_(1, order[:, :block_len], k_mask)
                transfer_index &= mask_token_mask
                final_transfer_index = transfer_index & mask_token_mask

            elif needs_entropy and strategy == "least_entropy":
                # least_entropy needs entropy to select lowest-entropy (most certain) position
                if use_dense:
                    log_probs_full = probs.clamp_min(EPS).log()
                    entropies_all = -(probs * log_probs_full).sum(dim=-1)
                else:
                    sparse_log_p = _sparse_probs.clamp_min(EPS).log()
                    sparse_ent = -(_sparse_probs * sparse_log_p).sum(dim=-1)
                    entropies_all = torch.zeros(d_batch_size, block_len, device=device, dtype=sparse_ent.dtype)
                    entropies_all.view(-1)[_masked_indices] = sparse_ent

                # Select positions with lowest entropy among masked positions
                masked_entropies = torch.where(
                    mask_token_mask, entropies_all, torch.full_like(entropies_all, torch.inf)
                )
                _, top_indices = torch.topk(masked_entropies, k=block_len, dim=1, largest=False)
                k_mask = torch.arange(block_len, device=device).unsqueeze(0) < effective_num_to_transfer.unsqueeze(1)
                transfer_index = torch.zeros_like(mask_token_mask, dtype=torch.bool)
                transfer_index.scatter_(1, top_indices, k_mask)
                transfer_index &= mask_token_mask
                final_transfer_index = transfer_index & mask_token_mask

            elif _pos_fp_done:
                # Position fast-path: entropy was computed only at the target position
                entropies_all = _pos_fp_entropies
            elif can_skip_entropy:
                # Lazy entropy: skip computation in chain intermediate steps
                entropies_all = batch_entropies  # reuse previous (zeros propagate)
            else:
                # Compute entropy (sparse when not all masked)
                if use_dense:
                    log_probs_full = probs.clamp_min(EPS).log()
                    entropies_all = -(probs * log_probs_full).sum(dim=-1)
                else:
                    sparse_log_p = _sparse_probs.clamp_min(EPS).log()
                    sparse_ent = -(_sparse_probs * sparse_log_p).sum(dim=-1)
                    entropies_all = torch.zeros(d_batch_size, block_len, device=device, dtype=sparse_ent.dtype)
                    entropies_all.view(-1)[_masked_indices] = sparse_ent

            # --- Signal logging: record per-step commit signals (only when enabled) ---
            if self.log_step_signals:
                committed_positions = final_transfer_index.cpu().tolist()
                confidence_vals = batch_seq_x0_p.cpu().tolist()
                step_idx = chain_state['cpu_step'] if chain_state is not None and 'cpu_step' in chain_state else None
                for j, seq in enumerate(denoising_seqs):
                    committed = [pos for pos, c in enumerate(committed_positions[j]) if c]
                    if committed:
                        sig = {
                            'step': step_idx if step_idx is not None else seq.current_denoising_step,
                            'strategy': strategy,
                            'committed_positions': committed,
                            'confidence': [confidence_vals[j][p] for p in committed],
                        }
                        seq.block_step_signals.append(sig)

            batch_new_tokens = torch.where(final_transfer_index, batch_seq_x0, batch_current_tokens)
            batch_new_trajectory = torch.where(
                final_transfer_index & (batch_trajectory == 0), batch_global_step_plus_1, batch_trajectory
            )
            batch_new_logprobs = torch.where(final_transfer_index, batch_seq_x0_logp, batch_logprobs)
            batch_new_entropies = torch.where(final_transfer_index, entropies_all, batch_entropies)

            new_denoising_steps = cs_denoising_steps + 1
            new_global_steps = cs_global_steps + 1

            has_saving_in_batch = bool(saving_seq_indices)

            # Fast prediction: if all sequences are mid-denoising (step+1 < limit),
            # no SAVING can happen — skip the GPU→CPU sync entirely
            can_predict_no_saving = False
            if chain_state is not None and not has_saving_in_batch:
                cpu_step = chain_state.get('cpu_step', None)
                if cpu_step is not None and cpu_step + 1 < _s[0].denoising_steps:
                    can_predict_no_saving = True

            if can_predict_no_saving:
                # Zero-sync chain fast path: no GPU→CPU synchronization needed
                schedule = _s[0].num_transfer_tokens_per_step
                total_transferred = d_batch_size * schedule[cpu_step]
                self._chain_state_out = {
                    'tokens': batch_new_tokens,
                    'logprobs': batch_new_logprobs,
                    'entropies': batch_new_entropies,
                    'trajectory': batch_new_trajectory,
                    'denoising_steps': new_denoising_steps,
                    'global_steps': new_global_steps,
                    'cpu_step': cpu_step + 1,
                }
                for j, seq in enumerate(_s):
                    seq.intermediate_block_tokens = batch_new_tokens[j]
                seq._chain_total_transferred = total_transferred
            else:
                # Need GPU sync to determine SAVING transitions
                remaining_masks = (batch_new_tokens == self.mask_token_id).any(dim=1)
                step_limits = torch.tensor([seq.denoising_steps for seq in _s], device=device)
                is_fully_denoised = (~remaining_masks) | (new_denoising_steps >= step_limits)
                any_saving = is_fully_denoised.any().item()

                if not any_saving and not has_saving_in_batch:
                    # No SAVING transitions — provide chain_state
                    cpu_step_val = chain_state['cpu_step'] + 1 if chain_state is not None and 'cpu_step' in chain_state else None
                    self._chain_state_out = {
                        'tokens': batch_new_tokens,
                        'logprobs': batch_new_logprobs,
                        'entropies': batch_new_entropies,
                        'trajectory': batch_new_trajectory,
                        'denoising_steps': new_denoising_steps,
                        'global_steps': new_global_steps,
                    }
                    if cpu_step_val is not None:
                        self._chain_state_out['cpu_step'] = cpu_step_val
                    if chain_state is not None:
                        # Chain continuation: minimal CPU state update
                        schedule = _s[0].num_transfer_tokens_per_step
                        step_idx = chain_state.get('cpu_step', _s[0].current_denoising_step)
                        total_transferred = d_batch_size * schedule[min(step_idx, len(schedule)-1)]
                        for j, seq in enumerate(_s):
                            seq.intermediate_block_tokens = batch_new_tokens[j]
                        seq._chain_total_transferred = total_transferred
                    else:
                        # First step in chain: do CPU update + provide chain_state
                        self._chain_state_out['cpu_step'] = _s[0].current_denoising_step
                        num_to_transfer_per_seq = final_transfer_index.sum(dim=1)
                        cpu_batch = torch.stack([
                            new_denoising_steps.to(torch.int64),
                            new_global_steps.to(torch.int64),
                            num_to_transfer_per_seq.to(torch.int64),
                        ]).cpu().tolist()  # [3, D] -> single sync
                        for j, seq in enumerate(_s):
                            seq.intermediate_block_tokens = batch_new_tokens[j]
                            seq.intermediate_block_tokens_entropy = batch_new_entropies[j]
                            seq.block_trajectory = batch_new_trajectory[j]
                            seq.block_logprobs = batch_new_logprobs[j]
                            seq.block_entropies = batch_new_entropies[j]
                            seq.current_denoising_step = cpu_batch[0][j]
                            seq.global_denoising_step = cpu_batch[1][j]
                            seq.num_to_transfer = cpu_batch[2][j]
                else:
                    # SAVING transitions: full sync path
                    num_to_transfer_per_seq = final_transfer_index.sum(dim=1)
                    cpu_batch = torch.stack([
                        new_denoising_steps.to(torch.int64),
                        new_global_steps.to(torch.int64),
                        num_to_transfer_per_seq.to(torch.int64),
                        is_fully_denoised.to(torch.int64),
                    ]).cpu().tolist()  # [4, D] -> single sync
                    new_denoising_steps_cpu = cpu_batch[0]
                    new_global_steps_cpu = cpu_batch[1]
                    num_to_transfer_cpu = cpu_batch[2]
                    is_fully_denoised_cpu = cpu_batch[3]

                    for j, seq in enumerate(_s):
                        seq.intermediate_block_tokens = batch_new_tokens[j]
                        seq.intermediate_block_tokens_entropy = batch_new_entropies[j]
                        seq.block_trajectory = batch_new_trajectory[j]
                        seq.block_logprobs = batch_new_logprobs[j]
                        seq.block_entropies = batch_new_entropies[j]

                        seq.current_denoising_step = new_denoising_steps_cpu[j]
                        seq.global_denoising_step = new_global_steps_cpu[j]
                        seq.num_to_transfer = num_to_transfer_cpu[j]

                        if is_fully_denoised_cpu[j]:
                            seq.status = SequenceStatus.SAVING
                    self._chain_state_out = None

        finished_seqs = [seq for seq in self.running if seq.is_finished]
        if finished_seqs:
            for seq in finished_seqs:
                self.block_manager.deallocate(seq)
            self.running = [seq for seq in self.running if not seq.is_finished]
        if self.prefill_ready:
            self.prefill_ready = deque(seq for seq in self.prefill_ready if not seq.is_finished)
        return finished_seqs

    def postprocess(self, seqs: list[Sequence], logits: torch.Tensor, run_type: RunType,
                    chain_state: dict | None = None) -> list[Sequence]:
        if run_type == RunType.PREFILL:
            for seq in seqs:
                seq.num_cached_tokens = seq.num_prefill_tokens
                seq.status = SequenceStatus.DENOISING
        elif run_type == RunType.DENOISE:
            device = logits.device
            batch_size = len(seqs)
            block_len = seqs[0].block_length # Assume all are same

            # --- 1. Batched Sampling & Initial Calculations ---
            if self.consistent_sampling_params:
                probs = self.sample_pipe(logits, temperature=seqs[0].temperature).view(batch_size, block_len, -1)
                self.apply_repetition_penalty(probs, seqs)
            else:
                # Handle diverse temperatures
                # Logits: (B*L, V) or (B, L, V)
                if logits.dim() == 3:
                    logits = logits.view(-1, logits.shape[-1])
                
                # Create temperature tensor (B*L, 1)
                temps = torch.tensor([seq.temperature for seq in seqs], device=device, dtype=torch.float)
                temps = temps.repeat_interleave(block_len).unsqueeze(1)
                
                # Apply temperature
                logits = logits / temps
                probs = F.softmax(logits, dim=-1).view(batch_size, block_len, -1)
                self.apply_repetition_penalty(probs, seqs)

            entropies_all = -(probs.clamp_min(EPS) * (probs.clamp_min(EPS)).log()).sum(dim=-1)
            
            if self.consistent_sampling_params:
                batch_top_k = seqs[0].top_k
                batch_top_p = seqs[0].top_p
            else:
                batch_top_p = torch.tensor([seq.top_p for seq in seqs], device=device, dtype=torch.float)
                batch_top_k = torch.tensor([seq.top_k for seq in seqs], device=device, dtype=torch.long)
            
            batch_seq_x0 = top_k_top_p_sampling_from_probs(
                probs.view(-1, probs.shape[-1]), 
                top_k=batch_top_k, 
                top_p=batch_top_p
            ).to(torch.int64).view(batch_size, block_len)
            
            batch_seq_x0_p = torch.gather(probs, -1, batch_seq_x0.unsqueeze(-1)).squeeze(-1)
            batch_seq_x0_logp = torch.log(batch_seq_x0_p.clamp_min(EPS))
            
            # Stack tensors from sequences
            batch_current_tokens = torch.stack([seq.intermediate_block_tokens.to(device) for seq in seqs])
            batch_logprobs = torch.stack([seq.block_logprobs.to(device) for seq in seqs])
            batch_entropies = torch.stack([seq.block_entropies.to(device) for seq in seqs])
            batch_trajectory = torch.stack([seq.block_trajectory.to(device) for seq in seqs])
            
            num_to_transfer_list = [
                seq.num_transfer_tokens_per_step[seq.current_denoising_step] 
                if seq.status == SequenceStatus.DENOISING else 0 
                for seq in seqs
            ]
            batch_num_to_transfer = torch.tensor(num_to_transfer_list, device=device, dtype=torch.long)
            
            batch_global_step_plus_1 = torch.tensor(
                [seq.global_denoising_step + 1 for seq in seqs], device=device, dtype=torch.long).unsqueeze(1) # (B, 1)
            
            all_statuses = [seq.status for seq in seqs]
            denoising_mask_bool = torch.tensor([s == SequenceStatus.DENOISING for s in all_statuses], device=device)
            saving_mask_bool = torch.tensor([s == SequenceStatus.SAVING for s in all_statuses], device=device)
            
            denoising_mask = denoising_mask_bool.unsqueeze(1) 
            
            mask_token_mask = (batch_current_tokens == self.mask_token_id) & denoising_mask

            strategies = [seq.remasking_strategy if status == SequenceStatus.DENOISING else '' for seq, status in zip(seqs, all_statuses)]
            if self.diversity_enforce:
                strategies = [strategy if (seq.num_generated_tokens > self.barrier) else 'sequential' for strategy, seq in zip(strategies, seqs)]
            elif self.epsilon_greedy:
                strategies = ['random' if random.random() < self.epsilon else strategy for strategy in strategies ]
            
            seq_mask = torch.tensor([s == 'sequential' for s in strategies], device=device).unsqueeze(1)
            low_conf_static_mask = torch.tensor(['low_confidence_static' in s for s in strategies], device=device).unsqueeze(1)
            low_conf_dynamic_mask = torch.tensor(['low_confidence_dynamic' in s for s in strategies], device=device).unsqueeze(1)
            entropy_bounded_mask = torch.tensor(['entropy_bounded' in s for s in strategies], device=device).unsqueeze(1)
            random_mask = torch.tensor([s == 'random' for s in strategies], device=device).unsqueeze(1)
            
            transfer_index = torch.zeros((batch_size, block_len), dtype=torch.bool, device=device)
            
            if seq_mask.any():
                first_mask_pos = torch.argmax(mask_token_mask.int(), dim=1, keepdim=True)
                range_tensor = torch.arange(block_len, device=device).unsqueeze(0)
                start_pos_b = first_mask_pos
                end_pos_b = (start_pos_b + batch_num_to_transfer.unsqueeze(1)).clamp_max(block_len)
                seq_transfer_index = (range_tensor >= start_pos_b) & (range_tensor < end_pos_b) & mask_token_mask
                transfer_index = torch.where(seq_mask, seq_transfer_index, transfer_index)

            if low_conf_static_mask.any():
                confidence = torch.where(mask_token_mask, batch_seq_x0_p, -torch.inf)
                # Use block_len as k (small, avoids GPU→CPU sync)
                _, top_indices = torch.topk(confidence, k=block_len, dim=1)
                k_mask = torch.arange(block_len, device=device).unsqueeze(0) < batch_num_to_transfer.unsqueeze(1)
                static_transfer_index = torch.zeros_like(confidence, dtype=torch.bool).scatter_(1, top_indices, k_mask)
                transfer_index = torch.where(low_conf_static_mask, static_transfer_index, transfer_index)

            if low_conf_dynamic_mask.any():
                dyn_thresholds = torch.tensor([seq.dynamic_threshold for seq in seqs], device=device).unsqueeze(1)
                confidence = torch.where(mask_token_mask, batch_seq_x0_p, -torch.inf)
                dyn_transfer_index = (confidence > dyn_thresholds)
                num_transferred_dyn = dyn_transfer_index.sum(dim=1)
                needs_fallback = (num_transferred_dyn < batch_num_to_transfer) & low_conf_dynamic_mask.squeeze()
                
                if needs_fallback.any():
                    fallback_mask_token_mask = mask_token_mask[needs_fallback]
                    fallback_num_to_transfer = batch_num_to_transfer[needs_fallback].unsqueeze(1)
                    first_mask_pos = torch.argmax(fallback_mask_token_mask.int(), dim=1, keepdim=True)
                    range_tensor = torch.arange(block_len, device=device).unsqueeze(0)
                    start_pos_b = first_mask_pos
                    end_pos_b = (start_pos_b + fallback_num_to_transfer).clamp_max(block_len)
                    fallback_indices = (range_tensor >= start_pos_b) & (range_tensor < end_pos_b) & fallback_mask_token_mask
                    dyn_transfer_index[needs_fallback] = fallback_indices
                
                batch_num_to_transfer = torch.where(
                    low_conf_dynamic_mask.squeeze(), 
                    dyn_transfer_index.sum(dim=1), 
                    batch_num_to_transfer
                )
                transfer_index = torch.where(low_conf_dynamic_mask, dyn_transfer_index, transfer_index)

            if entropy_bounded_mask.any():
                masked_entropies = torch.where(mask_token_mask, entropies_all, torch.inf)
                ent_sorted, order = torch.sort(masked_entropies, dim=1, descending=False)
                ent_sorted_masked = torch.where(ent_sorted == torch.inf, 0.0, ent_sorted)
                cumsum = torch.cumsum(ent_sorted_masked, dim=1)
                eb_thresholds = torch.tensor([seq.eb_threshold for seq in seqs], device=device).unsqueeze(1)
                k_tensor = torch.searchsorted(cumsum, eb_thresholds, right=False)
                k_tensor.clamp_min_(1) 
                k_mask_eb = torch.arange(block_len, device=device).unsqueeze(0) < k_tensor
                # Fix: Use mask_token_mask or similar to initialize eb_transfer_index, not confidence which is undefined here
                eb_transfer_index = torch.zeros_like(mask_token_mask, dtype=torch.bool).scatter_(1, order, k_mask_eb)
                
                batch_num_to_transfer = torch.where(
                    entropy_bounded_mask.squeeze(), 
                    k_tensor.squeeze(1), 
                    batch_num_to_transfer
                )
                transfer_index = torch.where(entropy_bounded_mask, eb_transfer_index, transfer_index)
                
            if random_mask.any():
                B, L = mask_token_mask.shape
                scores = torch.rand((B, L), device=device)
                scores = scores.masked_fill(~mask_token_mask, -1.0)
                # Use block_len as k (small, avoids GPU→CPU sync)
                _, top_indices = scores.topk(block_len, dim=-1)
                k_mask = torch.arange(block_len, device=device).unsqueeze(0) < batch_num_to_transfer.unsqueeze(1)
                random_transfer_index = torch.zeros_like(mask_token_mask, dtype=torch.bool).scatter_(1, top_indices, k_mask)
                transfer_index = torch.where(random_mask, random_transfer_index, transfer_index)
                
            final_transfer_index = transfer_index & mask_token_mask

            batch_new_tokens = torch.where(final_transfer_index, batch_seq_x0, batch_current_tokens)
            batch_new_trajectory = torch.where(
                final_transfer_index & (batch_trajectory == 0), 
                batch_global_step_plus_1, 
                batch_trajectory
            )
            batch_new_logprobs = torch.where(final_transfer_index, batch_seq_x0_logp, batch_logprobs)
            batch_new_entropies = torch.where(final_transfer_index, entropies_all, batch_entropies)

            new_denoising_steps = torch.tensor([seq.current_denoising_step for seq in seqs], device=device) + denoising_mask_bool.int()
            new_global_steps = torch.tensor([seq.global_denoising_step for seq in seqs], device=device) + denoising_mask_bool.int()
            
            is_fully_denoised = (~(batch_new_tokens == self.mask_token_id).any(dim=1)) | \
                                (new_denoising_steps >= torch.tensor([seq.denoising_steps for seq in seqs], device=device))

            # Optimization: Move scalars to CPU in batch to avoid synchronization in loop
            new_denoising_steps_cpu = new_denoising_steps.tolist()
            new_global_steps_cpu = new_global_steps.tolist()
            num_to_transfer_cpu = final_transfer_index.sum(dim=1).tolist()
            is_fully_denoised_cpu = is_fully_denoised.tolist()
            denoising_mask_cpu = denoising_mask_bool.tolist()
            saving_mask_cpu = saving_mask_bool.tolist()

            for i, seq in enumerate(seqs):
                if denoising_mask_cpu[i]:
                    # Views into small batch tensors (batch_size × block_len); no clone needed.
                    seq.intermediate_block_tokens = batch_new_tokens[i]
                    seq.intermediate_block_tokens_entropy = batch_new_entropies[i]
                    seq.block_trajectory = batch_new_trajectory[i]
                    seq.block_logprobs = batch_new_logprobs[i]
                    seq.block_entropies = batch_new_entropies[i]
                    
                    seq.current_denoising_step = new_denoising_steps_cpu[i]
                    seq.global_denoising_step = new_global_steps_cpu[i]
                    seq.num_to_transfer = num_to_transfer_cpu[i]
                    
                    if is_fully_denoised_cpu[i]:
                        seq.status = SequenceStatus.SAVING

                elif saving_mask_cpu[i]:
                    seq.commit_block(seq.intermediate_block_tokens)
                    seq.num_to_transfer = 0
                    if not seq.is_finished:
                        seq.start_new_block()
                    else:
                        self.block_manager.deallocate(seq)

        # Optimization: Filter running list efficiently
        # We already deallocated finished sequences in the loop (if they finished in SAVING state).
        # But we might have sequences that finished in other ways (e.g. max tokens check outside loop? No, commit_block handles it).
        # However, to be safe and avoid double deallocation overhead (even if safe), we can track indices.
        
        # Actually, the list comprehension [seq for seq in self.running if seq.is_finished] iterates ALL running seqs.
        # If running has 64 items, it's fast.
        # But calling deallocate again is redundant.
        # Let's just filter.
        
        # User suggested logic:
        finished_seqs = [seq for seq in self.running if seq.is_finished]
        self.running = [seq for seq in self.running if not seq.is_finished]
        for seq in finished_seqs:
            self.block_manager.deallocate(seq)
        
        if self.prefill_ready:
            self.prefill_ready = deque(
                seq for seq in self.prefill_ready if not seq.is_finished
            )
        return finished_seqs
