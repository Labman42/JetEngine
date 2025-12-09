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

    def _prepare_denoise_batch(self, prefill_batch: list[Sequence]) -> list[Sequence]:
        batch: list[Sequence] = []
        prefill_ids = {seq.seq_id for seq in prefill_batch}
        
        requests: list[tuple[Sequence, int]] = []
        total_needed = 0
        # We need to check availability dynamically or conservatively.
        # Since we are single-threaded here, we can check current free blocks.
        available_blocks = len(self.block_manager.free_block_ids)
        
        for seq in self.running:
            if seq.seq_id in prefill_ids:
                continue
            if seq.status not in (SequenceStatus.DENOISING, SequenceStatus.SAVING):
                continue
                
            num_new_blocks = seq.num_new_blocks_needed(self.block_manager.block_size)
            if num_new_blocks > 0:
                if total_needed + num_new_blocks > available_blocks:
                    print(f"[Warning] Cannot append {num_new_blocks} blocks for seq {seq.seq_id}. Not enough memory.")
                    continue
                requests.append((seq, num_new_blocks))
                total_needed += num_new_blocks
            
            batch.append(seq)
            
        if requests:
            self.block_manager.append_blocks_batch(requests)
            
        return batch

    def schedule(self) -> ScheduleResult:
        self._release_finished_sequences()
        self._fill_slots_from_waiting()

        prefill_batch = self._prepare_prefill_batch()
        denoise_batch = self._prepare_denoise_batch(prefill_batch)

        if not prefill_batch and not denoise_batch:
            if self.running or self.waiting_prefill:
                print("[Warning] Scheduler idle: no batches prepared despite pending sequences.")
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
                        # print(k)
                        selected_token_indices = mask_index.nonzero(as_tuple=True)[0][order[:k]]
                        # print(selected_token_indices)
                        transfer_index[selected_token_indices] = True
                        num_to_transfer = k

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
            
    def postprocess_unify(self, seqs: list[Sequence], logits: torch.Tensor, run_type: RunType) -> list[Sequence]:
        # This function seems to be a duplicate or alternative to postprocess. 
        # I will update it to match the logic but it seems unused in the main path if consistent_sampling_params is False.
        # For now, I'll focus on updating it to be tensor-compatible.
        if run_type == RunType.PREFILL:
            for seq in seqs:
                seq.num_cached_tokens = seq.num_prefill_tokens
                seq.status = SequenceStatus.DENOISING
            return []

        if run_type != RunType.DENOISE or not seqs:
            return []

        device = logits.device
        batch_size = len(seqs)
        block_len = seqs[0].block_length

        probs = self.sample_pipe(logits, temperature=seqs[0].temperature).view(batch_size, block_len, -1)
        self.apply_repetition_penalty(probs, seqs)
        entropies_all = -(probs.clamp_min(EPS) * (probs.clamp_min(EPS)).log()).sum(dim=-1)

        flat_probs = probs.view(-1, probs.shape[-1])
        flat_samples = top_k_top_p_sampling_from_probs(
            flat_probs, top_k=seqs[0].top_k, top_p=seqs[0].top_p
        ).to(torch.int64)
        batch_seq_x0 = flat_samples.view(batch_size, block_len)
        batch_seq_x0_p = torch.gather(flat_probs, 1, flat_samples.unsqueeze(-1)).view(batch_size, block_len)
        batch_seq_x0_logp = torch.log(batch_seq_x0_p.clamp_min(EPS))

        # Stack tensors from sequences
        batch_current_tokens = torch.stack([seq.intermediate_block_tokens.to(device) for seq in seqs])
        batch_logprobs = torch.stack([seq.block_logprobs.to(device) for seq in seqs])
        batch_entropies = torch.stack([seq.block_entropies.to(device) for seq in seqs])
        batch_trajectory = torch.stack([seq.block_trajectory.to(device) for seq in seqs])
        
        batch_global_step_plus_1 = torch.tensor(
            [seq.global_denoising_step + 1 for seq in seqs], device=device, dtype=torch.long
        ).unsqueeze(1)

        denoising_mask_bool = torch.tensor(
            [seq.status == SequenceStatus.DENOISING for seq in seqs], device=device
        )
        saving_mask_bool = torch.tensor(
            [seq.status == SequenceStatus.SAVING for seq in seqs], device=device
        )
        denoising_mask = denoising_mask_bool.unsqueeze(1)

        num_to_transfer_list = [
            seq.num_transfer_tokens_per_step[seq.current_denoising_step] if seq.status == SequenceStatus.DENOISING else 0
            for seq in seqs
        ]
        batch_num_to_transfer = torch.tensor(num_to_transfer_list, device=device, dtype=torch.long)

        mask_token_mask = (batch_current_tokens == self.mask_token_id) & denoising_mask
        mask_available = mask_token_mask.any(dim=1)
        effective_num_to_transfer = torch.where(
            mask_available, batch_num_to_transfer, torch.zeros_like(batch_num_to_transfer)
        )

        strategy = seqs[0].remasking_strategy
        transfer_index = torch.zeros((batch_size, block_len), dtype=torch.bool, device=device)

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
            max_k = int(effective_num_to_transfer.max().item())
            if max_k > 0:
                _, top_indices = torch.topk(confidence, k=max_k, dim=1)
                k_mask = torch.arange(max_k, device=device).unsqueeze(0) < effective_num_to_transfer.unsqueeze(1)
                transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
                transfer_index.scatter_(1, top_indices, k_mask)
                transfer_index &= mask_token_mask

        elif strategy == "low_confidence_dynamic":
            confidence = torch.where(
                mask_token_mask, batch_seq_x0_p, torch.full_like(batch_seq_x0_p, -torch.inf)
            )
            dyn_transfer_index = confidence > seqs[0].dynamic_threshold
            dyn_transfer_index &= mask_token_mask
            num_transferred_dyn = dyn_transfer_index.sum(dim=1)
            needs_fallback = (num_transferred_dyn < effective_num_to_transfer) & mask_available & denoising_mask_bool
            if needs_fallback.any():
                fallback_mask = mask_token_mask[needs_fallback]
                fallback_num = effective_num_to_transfer[needs_fallback]
                range_tensor = torch.arange(block_len, device=device).unsqueeze(0)
                first_mask_pos = torch.argmax(fallback_mask.int(), dim=1).unsqueeze(1)
                end = (first_mask_pos + fallback_num.unsqueeze(1)).clamp_max(block_len)
                fallback_index = (range_tensor >= first_mask_pos) & (range_tensor < end)
                fallback_index &= fallback_mask
                dyn_transfer_index[needs_fallback] = fallback_index
            transfer_index = dyn_transfer_index

        elif strategy == "entropy_bounded":
            masked_entropies = torch.where(
                mask_token_mask, entropies_all, torch.full_like(entropies_all, torch.inf)
            )
            ent_sorted, order = torch.sort(masked_entropies, dim=1)
            ent_sorted_masked = torch.where(torch.isfinite(ent_sorted), ent_sorted, torch.zeros_like(ent_sorted))
            cumsum = torch.cumsum(ent_sorted_masked, dim=1)
            thresholds = torch.full((batch_size, 1), seqs[0].eb_threshold, device=device)
            k_tensor = torch.searchsorted(cumsum, thresholds, right=False)
            k_tensor = torch.where(
                (mask_available & denoising_mask_bool).unsqueeze(1), k_tensor.clamp_min(1), torch.zeros_like(k_tensor)
            ).squeeze(1)
            max_k = int(k_tensor.max().item())
            if max_k > 0:
                k_mask = torch.arange(max_k, device=device).unsqueeze(0) < k_tensor.unsqueeze(1)
                transfer_index = torch.zeros_like(mask_token_mask)
                transfer_index.scatter_(1, order[:, :max_k], k_mask)
                transfer_index &= mask_token_mask
        elif strategy == "random":
            B, L = mask_token_mask.shape
            scores = torch.rand((B, L), device=device)
            scores = scores.masked_fill(~mask_token_mask, -1.0)
            max_k = int(effective_num_to_transfer.max().item())
            if max_k > 0:
                _, top_indices = scores.topk(max_k, dim=-1)
                k_mask = torch.arange(max_k, device=device).unsqueeze(0) < effective_num_to_transfer.unsqueeze(1)
                transfer_index = torch.zeros_like(mask_token_mask, dtype=torch.bool)
                transfer_index.scatter_(1, top_indices, k_mask)
                transfer_index &= mask_token_mask
        else:
            raise ValueError(f"Unsupported remasking strategy: {strategy}")

        final_transfer_index = transfer_index & mask_token_mask

        batch_new_tokens = torch.where(final_transfer_index, batch_seq_x0, batch_current_tokens)
        batch_new_trajectory = torch.where(
            final_transfer_index & (batch_trajectory == 0), batch_global_step_plus_1, batch_trajectory
        )
        batch_new_logprobs = torch.where(final_transfer_index, batch_seq_x0_logp, batch_logprobs)
        batch_new_entropies = torch.where(final_transfer_index, entropies_all, batch_entropies)

        denoise_increment = denoising_mask_bool.int()
        new_denoising_steps = torch.tensor(
            [seq.current_denoising_step for seq in seqs], device=device
        ) + denoise_increment
        new_global_steps = torch.tensor(
            [seq.global_denoising_step for seq in seqs], device=device
        ) + denoise_increment

        remaining_masks = (batch_new_tokens == self.mask_token_id).any(dim=1)
        step_limits = torch.tensor([seq.denoising_steps for seq in seqs], device=device)
        is_fully_denoised = (~remaining_masks) | (new_denoising_steps >= step_limits)
        is_fully_denoised &= denoising_mask_bool

        for i, seq in enumerate(seqs):
            if seq.status == SequenceStatus.DENOISING:
                seq.intermediate_block_tokens = batch_new_tokens[i]
                seq.intermediate_block_tokens_entropy = batch_new_entropies[i]
                seq.block_trajectory = batch_new_trajectory[i]
                seq.block_logprobs = batch_new_logprobs[i]
                seq.block_entropies = batch_new_entropies[i]
                seq.current_denoising_step = new_denoising_steps[i].item()
                seq.global_denoising_step = new_global_steps[i].item()
                seq.num_to_transfer = final_transfer_index[i].sum().item()
                if is_fully_denoised[i]:
                    seq.status = SequenceStatus.SAVING

            elif seq.status == SequenceStatus.SAVING:
                seq.commit_block(seq.intermediate_block_tokens)
                seq.num_to_transfer = 0
                if not seq.is_finished:
                    seq.start_new_block()
                else:
                    self.block_manager.deallocate(seq)

        finished_seqs = [seq for seq in self.running if seq.is_finished]
        if finished_seqs:
            for seq in finished_seqs:
                self.block_manager.deallocate(seq)
            self.running = [seq for seq in self.running if not seq.is_finished]
        if self.prefill_ready:
            self.prefill_ready = deque(seq for seq in self.prefill_ready if not seq.is_finished)
        return finished_seqs

    def postprocess(self, seqs: list[Sequence], logits: torch.Tensor, run_type: RunType) -> list[Sequence]:
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
                max_k = batch_num_to_transfer.max().item()
                _, top_indices = torch.topk(confidence, k=max_k, dim=1)
                k_mask = torch.arange(max_k, device=device).unsqueeze(0) < batch_num_to_transfer.unsqueeze(1)
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
                max_k = batch_num_to_transfer.max().item()
                _, top_indices = scores.topk(max_k, dim=-1)
                k_mask = torch.arange(max_k, device=device).unsqueeze(0) < batch_num_to_transfer.unsqueeze(1)
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
                    # Update sequence state - Keep as tensors!
                    # Fix: Use .clone() to avoid keeping views of large batch tensors alive
                    seq.intermediate_block_tokens = batch_new_tokens[i].clone()
                    seq.intermediate_block_tokens_entropy = batch_new_entropies[i].clone()
                    seq.block_trajectory = batch_new_trajectory[i].clone()
                    seq.block_logprobs = batch_new_logprobs[i].clone()
                    seq.block_entropies = batch_new_entropies[i].clone()
                    
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
