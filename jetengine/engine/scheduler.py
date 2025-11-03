from collections import deque
import torch
from torch.nn import functional as F
import numpy as np

from jetengine.config import Config
from jetengine.engine.sequence import Sequence, SequenceStatus, RunType
from jetengine.engine.block_manager import BlockManager
from jetengine.layers.sampler import sample_with_temperature_topk_topp
from flashinfer.logits_processor import LogitsPipe, Temperature, Softmax, TopP, TopK, Sample
from flashinfer.sampling import top_p_sampling_from_probs, top_k_top_p_sampling_from_probs
from torch.distributions import Categorical

EPS = 1e-12

class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.mask_token_id = config.mask_token_id
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.running: list[Sequence] = []
        self.sample_pipe = LogitsPipe([
                                Temperature(),      # Scale logits by temperature
                                Softmax(),          # Convert logits to probabilities
                            ])

    def add(self, seq: Sequence):
        self.running.append(seq)

    def is_finished(self):
        return not self.running

    def schedule(self) -> tuple[list[Sequence], RunType] | tuple[None, None]:
        # 1. Schedule new sequences for prefill
        self.running = [s for s in self.running if not s.is_finished]
        prefill_candidates = [s for s in self.running if s.status == SequenceStatus.WAITING]
        if prefill_candidates:
            prefill_batch = []
            # Simple batching: take as many as fit
            for seq in prefill_candidates:
                # num_tokens for a waiting seq is its prefill length
                if len(prefill_batch) < self.max_num_seqs and self.block_manager.can_allocate(seq):
                    self.block_manager.allocate(seq)
                    seq.status = SequenceStatus.PREFILLING
                    prefill_batch.append(seq)
            if prefill_batch:
                return prefill_batch, RunType.PREFILL   
        # 2. If no prefilling, create a DENOISE batch.
        denoise_candidates = [s for s in self.running if s.status == SequenceStatus.DENOISING or s.status == SequenceStatus.SAVING]
        if denoise_candidates:
            denoise_batch = []
            for seq in denoise_candidates:
                num_new_blocks = seq.num_new_blocks_needed(self.block_manager.block_size)
                if len(denoise_batch) < self.max_num_seqs and self.block_manager.can_append_blocks(num_new_blocks):
                    self.block_manager.append_blocks(seq, num_new_blocks)
                    denoise_batch.append(seq)
                else:
                    print(f"[Warning] Can not denoise seq with {len(seq)} and block_manager with {self.block_manager.free_block_ids}")
            if denoise_batch:
                return denoise_batch, RunType.DENOISE
        left = len(self.running)
        if left > 0:
            print("[Warning] No progress can be made .")
            print(f"[Warning]: Left {left}")
        return None, None     

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
                    
                    current_block_tensor = torch.tensor(seq.intermediate_block_tokens, device=logits.device)
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
                        block_probs = probs[start_idx : start_idx + block_len]
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

                    # update
                    new_block_list = current_block_tensor.tolist()
                    accepted_tokens = seq_x0[transfer_index].tolist()
                    accepted_tokens_entropy = seq_entropies[transfer_index].tolist()
                    accepted_tokens_logprobs = seq_x0_logp[transfer_index].tolist()
                    new_block_list_entropy = torch.tensor(seq.intermediate_block_tokens_entropy, device=logits.device).tolist()
                    original_indices = transfer_index.nonzero(as_tuple=True)[0].tolist()
                    
                    # track trajectory
                    if seq.block_trajectory is None or len(seq.block_trajectory) != block_len:
                        seq.block_trajectory = [0] * block_len
                        
                    if seq.block_logprobs is None or len(seq.block_logprobs) != block_len:
                        seq.block_logprobs = [0.0] * block_len
                    if seq.block_entropies is None or len(seq.block_entropies) != block_len:
                        seq.block_entropies = [0.0] * block_len
                        
                    first_time_global = seq.global_denoising_step + 1
                    for idx, token, entropy, logprob in zip(
                        original_indices, 
                        accepted_tokens, 
                        accepted_tokens_entropy,
                        accepted_tokens_logprobs
                    ):
                        new_block_list[idx] = token
                        new_block_list_entropy[idx] = entropy
                        # Track trajectory (only mark once)
                        if seq.block_trajectory[idx] == 0:
                            seq.block_trajectory[idx] = first_time_global
                        # NEW: Store logprobs and entropies (update every time)
                        seq.block_logprobs[idx] = logprob
                        seq.block_entropies[idx] = entropy
                        
                    seq.intermediate_block_tokens = new_block_list
                    seq.intermediate_block_tokens_entropy = new_block_list_entropy
                    
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
            
    def postprocess(self, seqs: list[Sequence], logits: torch.Tensor, run_type: RunType):
        if run_type == RunType.PREFILL:
            for seq in seqs:
                seq.num_cached_tokens = seq.num_prefill_tokens
                seq.status = SequenceStatus.DENOISING
        elif run_type == RunType.DENOISE:
            device = logits.device
            batch_size = len(seqs)
            block_len = seqs[0].block_length # Assume all are same

            # --- 1. Batched Sampling & Initial Calculations ---
            # Reshape logits to (B, L, V) for easier processing
            # if logits.dim() == 2:
            #     logits = logits.view(batch_size, block_len, -1) # (B*L, V) -> (B, L, V)

            probs = self.sample_pipe(logits, temperature=seqs[0].temperature).view(batch_size, block_len, -1)
            entropies_all = -(probs.clamp_min(EPS) * (probs.clamp_min(EPS)).log()).sum(dim=-1)
            if self.consistent_sampling_params:
                # These are scalars, apply to all
                batch_top_k = seqs[0].top_k
                batch_top_p = seqs[0].top_p
            else:
                # These are tensors
                batch_top_p = torch.tensor([seq.top_p for seq in seqs], device=device, dtype=torch.float)
                batch_top_k = torch.tensor([seq.top_k for seq in seqs], device=device, dtype=torch.long)
            # Perform sampling for the entire batch
            # Shape: (B, L)
            batch_seq_x0 = top_k_top_p_sampling_from_probs(
                probs.view(-1, probs.shape[-1]), # Sampler might expect (N, V)
                top_k=batch_top_k, 
                top_p=batch_top_p
            ).to(torch.int64).view(batch_size, block_len)
            
            # Get probabilities and log-probabilities of the sampled tokens
            # Shape: (B, L)
            batch_seq_x0_p = torch.gather(probs, -1, batch_seq_x0.unsqueeze(-1)).squeeze(-1)
            batch_seq_x0_logp = torch.log(batch_seq_x0_p.clamp_min(EPS))
            # Create tensors for all sequence states
            batch_current_tokens = torch.tensor(
                [seq.intermediate_block_tokens for seq in seqs], device=device, dtype=torch.int64)
            
            batch_logprobs = torch.tensor(
                [seq.block_logprobs or [0.0]*block_len for seq in seqs], device=device, dtype=torch.float)
            
            batch_entropies = torch.tensor(
                [seq.block_entropies or [0.0]*block_len for seq in seqs], device=device, dtype=torch.float)

            batch_trajectory = torch.tensor(
                [seq.block_trajectory or [0]*block_len for seq in seqs], device=device, dtype=torch.long)
            
            # Get current num_to_transfer for each sequence in DENOISING state
            num_to_transfer_list = [
                seq.num_transfer_tokens_per_step[seq.current_denoising_step] 
                if seq.status == SequenceStatus.DENOISING else 0 
                for seq in seqs
            ]
            batch_num_to_transfer = torch.tensor(num_to_transfer_list, device=device, dtype=torch.long)
            # Get global step for trajectory tracking
            batch_global_step_plus_1 = torch.tensor(
                [seq.global_denoising_step + 1 for seq in seqs], device=device, dtype=torch.long).unsqueeze(1) # (B, 1)
            
            # Status masks (boolean)
            all_statuses = [seq.status for seq in seqs]
            denoising_mask_bool = torch.tensor([s == SequenceStatus.DENOISING for s in all_statuses], device=device)
            saving_mask_bool = torch.tensor([s == SequenceStatus.SAVING for s in all_statuses], device=device)
            
            # Broadcastable (B, 1) mask for filtering tensors
            denoising_mask = denoising_mask_bool.unsqueeze(1) 
            
            # Mask of all MASK tokens in DENOISING sequences
            # Shape: (B, L)
            mask_token_mask = (batch_current_tokens == self.mask_token_id) & denoising_mask

            # Strategy masks
            strategies = [seq.remasking_strategy if status == SequenceStatus.DENOISING else '' for seq, status in zip(seqs, all_statuses)]
            seq_mask = torch.tensor([s == 'sequential' for s in strategies], device=device).unsqueeze(1)
            low_conf_static_mask = torch.tensor(['low_confidence_static' in s for s in strategies], device=device).unsqueeze(1)
            low_conf_dynamic_mask = torch.tensor(['low_confidence_dynamic' in s for s in strategies], device=device).unsqueeze(1)
            entropy_bounded_mask = torch.tensor(['entropy_bounded' in s for s in strategies], device=device).unsqueeze(1)
            
            # Initialize the final index of all tokens to be transferred
            # Shape: (B, L)
            transfer_index = torch.zeros((batch_size, block_len), dtype=torch.bool, device=device)
            
            # --- Strategy: 'sequential' ---
            if seq_mask.any():
                # Find the first mask position for each sequence
                first_mask_pos = torch.argmax(mask_token_mask.int(), dim=1, keepdim=True) # (B, 1)
                
                # Create a range tensor [0, 1, ..., L-1]
                range_tensor = torch.arange(block_len, device=device).unsqueeze(0) # (1, L)
                
                # Create start and end transfer positions for each sequence
                start_pos_b = first_mask_pos
                end_pos_b = (start_pos_b + batch_num_to_transfer.unsqueeze(1)).clamp_max(block_len)
                
                # Create the transfer mask for sequential strategy
                seq_transfer_index = (range_tensor >= start_pos_b) & (range_tensor < end_pos_b) & mask_token_mask
                
                # Apply only to sequences using this strategy
                transfer_index = torch.where(seq_mask, seq_transfer_index, transfer_index)

            # --- Strategy: 'low_confidence_static' ---
            if low_conf_static_mask.any():
                # Calculate confidence, setting non-masked tokens to -inf
                confidence = torch.where(mask_token_mask, batch_seq_x0_p, -torch.inf)
                
                # Get the max K needed across all sequences
                max_k = batch_num_to_transfer.max().item()
                
                # Get the top-k indices (B, max_k)
                _, top_indices = torch.topk(confidence, k=max_k, dim=1)
                
                # Create a mask to select the *correct* number of K for each sequence
                # Shape: (B, max_k)
                k_mask = torch.arange(max_k, device=device).unsqueeze(0) < batch_num_to_transfer.unsqueeze(1)
                
                # Scatter the 'True' values into a (B, L) tensor
                static_transfer_index = torch.zeros_like(confidence, dtype=torch.bool).scatter_(1, top_indices, k_mask)
                
                # Apply only to sequences using this strategy
                transfer_index = torch.where(low_conf_static_mask, static_transfer_index, transfer_index)

            # --- Strategy: 'low_confidence_dynamic' ---
            if low_conf_dynamic_mask.any():
                dyn_thresholds = torch.tensor([seq.dynamic_threshold for seq in seqs], device=device).unsqueeze(1)
                confidence = torch.where(mask_token_mask, batch_seq_x0_p, -torch.inf)
                
                # Initial transfer index based on threshold
                dyn_transfer_index = (confidence > dyn_thresholds)
                
                # Check which sequences didn't meet the minimum `num_to_transfer`
                num_transferred_dyn = dyn_transfer_index.sum(dim=1)
                needs_fallback = (num_transferred_dyn < batch_num_to_transfer) & low_conf_dynamic_mask.squeeze()
                
                # if needs_fallback.any():
                #     # Get max K for only the sequences that need fallback
                #     fallback_k_values = batch_num_to_transfer[needs_fallback]
                #     max_k_dyn = fallback_k_values.max().item()
                    
                #     # Get top-k for *only* the fallback sequences
                #     _, top_indices_dyn = torch.topk(confidence[needs_fallback], k=max_k_dyn, dim=1)
                    
                #     # Create the K-mask for the fallback sequences
                #     k_mask_dyn = torch.arange(max_k_dyn, device=device).unsqueeze(0) < fallback_k_values.unsqueeze(1)
                    
                #     # Scatter to create the new transfer indices for fallback sequences
                #     fallback_indices = torch.zeros((needs_fallback.sum(), block_len), dtype=torch.bool, device=device)
                #     fallback_indices.scatter_(1, top_indices_dyn, k_mask_dyn)
                    
                #     # *Replace* the indices for the fallback sequences (as per original logic)
                #     dyn_transfer_index[needs_fallback] = fallback_indices
                
                if needs_fallback.any():
                    # Get the tensors for *only* the fallback sequences
                    fallback_mask_token_mask = mask_token_mask[needs_fallback]  # (num_fallback, L)
                    fallback_num_to_transfer = batch_num_to_transfer[needs_fallback].unsqueeze(1) # (num_fallback, 1)

                    # Find the first mask position for each *fallback* sequence
                    first_mask_pos = torch.argmax(fallback_mask_token_mask.int(), dim=1, keepdim=True) # (num_fallback, 1)
                    
                    # Create a range tensor [0, 1, ..., L-1]
                    range_tensor = torch.arange(block_len, device=device).unsqueeze(0) # (1, L)
                    
                    # Create start and end transfer positions for each *fallback* sequence
                    start_pos_b = first_mask_pos
                    end_pos_b = (start_pos_b + fallback_num_to_transfer).clamp_max(block_len)
                    
                    # Create the transfer mask for the *fallback* sequences
                    # Ensure we only select actual mask tokens within the sequential range
                    fallback_indices = (range_tensor >= start_pos_b) & (range_tensor < end_pos_b) & fallback_mask_token_mask
                    
                    # *Replace* the threshold-based indices with the new sequential indices
                    dyn_transfer_index[needs_fallback] = fallback_indices
                
                # Update the batch_num_to_transfer tensor for sequences that used this strategy
                batch_num_to_transfer = torch.where(
                    low_conf_dynamic_mask.squeeze(), 
                    dyn_transfer_index.sum(dim=1), 
                    batch_num_to_transfer
                )
                
                # Apply only to sequences using this strategy
                transfer_index = torch.where(low_conf_dynamic_mask, dyn_transfer_index, transfer_index)

            # --- Strategy: 'entropy_bounded' ---
            if entropy_bounded_mask.any():
                # Get entropies, setting non-masked tokens to +inf for ascending sort
                masked_entropies = torch.where(mask_token_mask, entropies_all, torch.inf)
                
                # Sort entropies ascending
                # We need all L tokens for stable indexing, so we sort over dim=1
                ent_sorted, order = torch.sort(masked_entropies, dim=1, descending=False)
                
                # Replace infs with 0 so cumsum works
                ent_sorted_masked = torch.where(ent_sorted == torch.inf, 0.0, ent_sorted)
                
                # Batched cumsum
                cumsum = torch.cumsum(ent_sorted_masked, dim=1)
                
                # Get thresholds for each sequence
                eb_thresholds = torch.tensor([seq.eb_threshold for seq in seqs], device=device).unsqueeze(1)
                
                # Find k for each sequence: number of tokens where cumsum < threshold
                k_tensor = torch.searchsorted(cumsum, eb_thresholds, right=False)
                
                # Enforce "if k == 0: k = 1"
                k_tensor.clamp_min_(1) 
                
                # Create a (B, L) mask based on k
                k_mask_eb = torch.arange(block_len, device=device).unsqueeze(0) < k_tensor
                
                # Scatter this mask back using the original sorted `order`
                eb_transfer_index = torch.zeros_like(confidence, dtype=torch.bool).scatter_(1, order, k_mask_eb)
                
                # Update the batch_num_to_transfer tensor
                batch_num_to_transfer = torch.where(
                    entropy_bounded_mask.squeeze(), 
                    k_tensor.squeeze(1), 
                    batch_num_to_transfer
                )

                # Apply only to sequences using this strategy
                transfer_index = torch.where(entropy_bounded_mask, eb_transfer_index, transfer_index)
                
            # Final transfer index must be denoising AND a mask token
            final_transfer_index = transfer_index & mask_token_mask

            # Update tokens: (B, L)
            batch_new_tokens = torch.where(final_transfer_index, batch_seq_x0, batch_current_tokens)

            # Update trajectory: (B, L)
            # Only update if trajectory is 0 (first time being accepted)
            batch_new_trajectory = torch.where(
                final_transfer_index & (batch_trajectory == 0), 
                batch_global_step_plus_1, 
                batch_trajectory
            )
            
            # Update logprobs and entropies: (B, L)
            # These are updated *every time* a token is accepted
            batch_new_logprobs = torch.where(final_transfer_index, batch_seq_x0_logp, batch_logprobs)
            batch_new_entropies = torch.where(final_transfer_index, entropies_all, batch_entropies)

            # Calculate final state changes
            new_denoising_steps = torch.tensor([seq.current_denoising_step for seq in seqs], device=device) + denoising_mask_bool.int()
            new_global_steps = torch.tensor([seq.global_denoising_step for seq in seqs], device=device) + denoising_mask_bool.int()
            
            # Check for finished blocks
            is_fully_denoised = (~(batch_new_tokens == self.mask_token_id).any(dim=1)) | \
                                (new_denoising_steps >= torch.tensor([seq.denoising_steps for seq in seqs], device=device))

            # --- 6. Lightweight Disaggregation Loop ---
            # This loop is now very fast, just for updating Python object state.

            for i, seq in enumerate(seqs):
                if denoising_mask_bool[i]:
                    # Update sequence state from the computed batch tensors
                    seq.intermediate_block_tokens = batch_new_tokens[i].tolist()
                    seq.intermediate_block_tokens_entropy = batch_new_entropies[i].tolist() # Use new entropies
                    
                    # Update trajectory and logprobs
                    seq.block_trajectory = batch_new_trajectory[i].tolist()
                    seq.block_logprobs = batch_new_logprobs[i].tolist()
                    seq.block_entropies = batch_new_entropies[i].tolist() # Also update this one
                    
                    # Update step counters
                    seq.current_denoising_step = new_denoising_steps[i].item()
                    seq.global_denoising_step = new_global_steps[i].item()
                    
                    # Update the *actual* number of tokens transferred
                    seq.num_to_transfer = final_transfer_index[i].sum().item()
                    
                    # Check if block is done
                    if is_fully_denoised[i]:
                        seq.status = SequenceStatus.SAVING

                elif saving_mask_bool[i]:
                    # This part remains the same, as it's state-machine logic
                    seq.commit_block(seq.intermediate_block_tokens)
                    seq.num_to_transfer = 0
                    if not seq.is_finished:
                        seq.start_new_block()
                    else:
                        self.block_manager.deallocate(seq)
                        
        # # Filter out finished sequences from the running list
        # finished_seqs = [seq for seq in self.running if seq.is_finished]
        # self.running = [seq for seq in self.running if not seq.is_finished]
        # for seq in finished_seqs:
        #     self.block_manager.deallocate(seq)