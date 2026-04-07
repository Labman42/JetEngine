import atexit
import math
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
from torch import nn
from contextlib import nullcontext

from jetengine.config import Config
from jetengine.sampling_params import SamplingParams
from jetengine.engine.sequence import Sequence, RunType, SequenceStatus
from jetengine.engine.scheduler import Scheduler
from jetengine.engine.model_runner import ModelRunner
from jetengine.utils.loader import load_from_hf_model
from jetengine.utils.statics import _estimate_kv_cache_usage, _actual_estimate_kv_cache_usage
from jetengine.engine.distributed_manager import DistributedManager





class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        
        est_blocks, est_bytes = _estimate_kv_cache_usage(config)
        est_gib = est_bytes / (1024 ** 3)
        print(
            f"[KVCache] Estimating {est_blocks:,} blocks "
            f"({est_gib:.2f} GiB) for up to {config.max_num_seqs:,} active sequences "
            f"of length {config.max_model_len:,} tokens."
        )

        self.dist_manager = DistributedManager(config.tensor_parallel_size)

        self.model_runner = ModelRunner(config, self.dist_manager)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, use_fast=True, trust_remote_code=True)
        config.eos = self.tokenizer.eos_token_id
        if config.mask_token_id == -1:
            config.mask_token_id = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id is not None else self.tokenizer.pad_token_id
        assert config.mask_token_id is not None, "Model tokenizer must have a mask_token_id or pad_token_id"

        self.config = config
        self.scheduler = Scheduler(config)
        self.scheduler.consistent_sampling_params = False
        # Chain statistics for diagnostics
        self._chain_attempts = 0
        self._chain_successes = 0
        self._chain_depth_counts = {}  # max_chain -> count
        atexit.register(self.exit)

    def offload_parameters(self, include_buffers: bool = False):
        """
        Replace all parameter (and buffer) storages with meta tensors.
        Keeps shapes/dtypes, frees GPU/CPU memory.
        """

        def offload_parameters_keep_buffers(model: torch.nn.Module):
            """
            Move *parameters* to meta to free memory while keeping buffers unchanged.
            Works for any module tree.
            """
            # 1) Snapshot real buffers (module reference + buffer name + tensor)
            saved_buffers = []
            for mod in model.modules():
                for bname, buf in list(mod._buffers.items()):
                    if buf is not None:
                        saved_buffers.append((mod, bname, buf))

            # 2) Move everything to meta
            model.to_empty(device=torch.device("meta"))

            # 3) Restore the saved, real buffers
            for mod, bname, buf in saved_buffers:
                # Reattach the original tensor (device/dtype preserved)
                mod._buffers[bname] = buf

            torch.cuda.empty_cache()
        if include_buffers:
            self.model_runner.model.to_empty(device=torch.device("meta"))
        else:
            offload_parameters_keep_buffers(self.model_runner.model)

        print("Successfully cleaned old parameters (buffers kept)." if not include_buffers
              else "Successfully cleaned old parameters and buffers.")

    def free_all_resources(self):
        """
        Completely frees all GPU resources: CUDAGraphs, Model, and KV Cache.
        """
        # print("Freeing CUDAGraphs, Model, and KV Cache...")

        if hasattr(self.model_runner, 'graphs'):
            del self.model_runner.graphs
            self.model_runner.graphs = {}
        if hasattr(self.model_runner, 'graph_pool'):
            del self.model_runner.graph_pool
            self.model_runner.graph_pool = None

        if hasattr(self.model_runner, 'model'):
            del self.model_runner.model

        if hasattr(self.model_runner, 'kv_cache'):
            del self.model_runner.kv_cache

        # 4. Free memory and reset peak stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # print("GPU resources freed.")

    def reload_parameters(self, hf_model: nn.Module):
        self.model_runner.reinit_model()
        load_from_hf_model(self.model_runner.model, hf_model=hf_model)

    def reload_from_hf_model(self, hf_model: nn.Module):
        """
        Reloads the engine from a HuggingFace model.
        This creates a new model shell, loads weights, re-allocates
        the KV cache, and re-captures CUDAGraphs.
        
        Args:
            hf_model (nn.Module): The HF model (CPU or GPU) to load weights from.
        """
        # print("Reloading engine...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        original_device = torch.device(torch.get_default_device())
        original_dtype = torch.get_default_dtype()

        try:
            torch.set_default_device("cuda")
            torch.set_default_dtype(self.config.torch_dtype)

            self.model_runner.reinit_model()
            # print("Created new model shell.")

            # print("Reloading parameters...")
            load_from_hf_model(self.model_runner.model, hf_model=hf_model)

            # print("Allocating new KV cache...")
            self.model_runner.allocate_kv_cache()
            # print("New KV cache allocated.")

            if not self.config.enforce_eager:
                # print("Re-capturing CUDAGraphs...")
                self.model_runner.capture_cudagraph()
            #     print("Graphs re-captured.")

            # print("Engine reload complete.")

        finally:
            torch.set_default_device(original_device)
            torch.set_default_dtype(original_dtype)
            self.scheduler = Scheduler(self.config)
            self.scheduler.consistent_sampling_params = True
            # print(f"Restored default device to {original_device}.")

    def exit(self):
        del self.model_runner

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if isinstance(prompt, list):
            if self.tokenizer.pad_token_id in prompt:
                start = prompt.index(self.tokenizer.pad_token_id) + 1
                prompt = prompt[start:]
        if sampling_params.stop_words is None:
            sampling_params.stop_words = [self.tokenizer.eos_token_id]
        seq = Sequence(prompt, self.config.mask_token_id, sampling_params)
        seq.eos_token_id = self.tokenizer.eos_token_id
        self.scheduler.add(seq)

    def step(self):
        schedule_result = self.scheduler.schedule()
        if not schedule_result.has_work:
            return [], 0  # Nothing to run

        finished_sequences: list[Sequence] = []
        postprocess_fn = (
            self.scheduler.postprocess
            if not getattr(self.scheduler, "consistent_sampling_params", False) or self.config.diversity_enforce or self.config.epsilon_greedy
            else self.scheduler.postprocess_unify
        )

        if schedule_result.prefill:
            logits = self.model_runner.run(schedule_result.prefill, RunType.PREFILL)
            finished_sequences.extend(
                postprocess_fn(schedule_result.prefill, logits, RunType.PREFILL)
            )

        tokens_generated = 0
        if schedule_result.denoise:
            denoise_seqs = schedule_result.denoise
            bs = len(denoise_seqs)
            # Adaptive chain depth:
            # - Small batches (tail): chain=4 for maximum throughput
            # - Large batches with no pending work: chain=5 (full block in one step)
            # - Large batches with pending prefills: chain=3 (balance throughput vs refill latency)
            has_pending = (len(self.scheduler.waiting_prefill) > 0 or
                          len(self.scheduler.prefill_ready) > 0)
            if bs <= 8:
                max_chain = 4
            elif not has_pending:
                max_chain = 5  # full block in one step when nothing is waiting
            else:
                max_chain = 3  # balance throughput vs prefill refill latency
            self._chain_depth_counts[max_chain] = self._chain_depth_counts.get(max_chain, 0) + 1
            _context_valid = False  # Track whether positions/block_tables can be reused
            _prev_bs = bs  # Track batch size for context reuse validation
            _chain_state = None  # Batch tensor state for chain reuse
            for _chain_step in range(max_chain):
                num_denoising = sum(1 for s in denoise_seqs if s.status == SequenceStatus.DENOISING)
                num_need_logits = num_denoising if num_denoising < len(denoise_seqs) else None
                # Reuse context (positions/block_tables) for pure DENOISING chain steps
                # Context becomes invalid when sequences commit blocks (SAVING) or batch changes
                reuse = _context_valid and num_need_logits is None
                # Pass batch tokens directly from chain_state to avoid loop+cat in prepare_denoise
                _chain_tokens = _chain_state['tokens'] if _chain_state is not None and reuse else None
                logits = self.model_runner.run(denoise_seqs, RunType.DENOISE,
                                               num_need_logits=num_need_logits,
                                               reuse_context=reuse,
                                               chain_batch_tokens=_chain_tokens)
                finished_sequences.extend(
                    postprocess_fn(denoise_seqs, logits, RunType.DENOISE,
                                   chain_state=_chain_state)
                )
                # Get chain_state from scheduler (set during postprocess_unify)
                _chain_state = getattr(self.scheduler, '_chain_state_out', None)
                if _chain_state is not None and hasattr(denoise_seqs[-1], '_chain_total_transferred'):
                    tokens_generated += denoise_seqs[-1]._chain_total_transferred
                else:
                    tokens_generated += sum(
                        getattr(seq, "num_to_transfer", 0) for seq in denoise_seqs
                    )
                if _chain_step < max_chain - 1:
                    self._chain_attempts += 1
                    # Filter out finished sequences (blocks already deallocated)
                    denoise_seqs = [s for s in denoise_seqs if s.status in
                                    (SequenceStatus.DENOISING, SequenceStatus.SAVING)]
                    if not denoise_seqs:
                        break
                    # Allocate any new KV blocks needed by just-committed sequences
                    if not self.scheduler.try_allocate_chain_blocks(denoise_seqs):
                        _context_valid = False
                        _chain_state = None
                        break
                    new_bs = len(denoise_seqs)
                    has_saving = any(s.status == SequenceStatus.SAVING for s in denoise_seqs)
                    # Context valid only if same batch size, same sequences, all DENOISING
                    if has_saving or new_bs != _prev_bs:
                        _context_valid = False
                        _chain_state = None  # Batch changed, can't reuse chain state
                        if has_saving:
                            # Re-sort: DENOISING first, SAVING last (for selective logits)
                            denoise_seqs.sort(key=lambda s: 0 if s.status == SequenceStatus.DENOISING else 1)
                    else:
                        _context_valid = True
                    _prev_bs = new_bs
                    self._chain_successes += 1

        # Deduplicate by sequence id to avoid returning the same sequence twice
        seen_seq_ids = set()
        finished_outputs = []
        for seq in finished_sequences:
            if seq.seq_id in seen_seq_ids:
                continue
            seen_seq_ids.add(seq.seq_id)

            if self.config.max_model_len:
                seq.token_ids = seq.token_ids[:self.config.max_model_len]
                response_len = min(
                    self.config.max_model_len - len(seq.prompt_token_ids),
                    len(seq.completion_token_ids),
                )
                seq.trajectory = seq.trajectory[:response_len]
                seq.logprobs = seq.logprobs[:response_len]
                seq.entropies = seq.entropies[:response_len]

            finished_outputs.append(
                (seq.seq_id, seq.token_ids, seq.trajectory, seq.logprobs, seq.entropies, seq.all_step_signals)
            )

        return finished_outputs, tokens_generated

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        ) -> list[str]:
        # ... (This method remains largely the same, but the progress bar will update differently) ...
        # The logic inside the `while not self.is_finished()` loop correctly calls `self.step()`
        # and collects outputs.
        if use_tqdm:
            pbar = tqdm(total=len(prompts),
                        desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
            self.scheduler.consistent_sampling_params = True
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        stall_counter = 0

        total_generated_tokens = 0
        start_time = perf_counter()

        while not self.is_finished():
            output, num_processed = self.step()
            
            if not output and not self.is_finished() and num_processed == 0:
                stall_counter += 1
                if stall_counter > 3:
                    print("\n[Warning] Deadlock detected: No progress can be made because all sequences are waiting for KV cache blocks, but no blocks are free. This can happen if the batch size is too large for the available KV cache. Try reducing the number of concurrent requests.")
                    break
            else:
                stall_counter = 0

            total_generated_tokens += num_processed

            throughput = total_generated_tokens / \
                (perf_counter() - start_time)
            if use_tqdm:
                pbar.set_postfix(
                    {"Throughput": f"{int(throughput)} tok/s"})

            for seq_id, token_ids, trajectory, logprobs, entropies, step_signals in output:
                outputs[seq_id] = {
                    "token_ids": token_ids,
                    "trajectory": trajectory,
                    "logprobs": logprobs,
                    "entropies": entropies,
                    "step_signals": step_signals,
                }
                if use_tqdm:
                    pbar.update(1)

        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]

        safe_outputs = []
        for output in outputs:
            token_ids = output["token_ids"]
            trajectory = output["trajectory"]
            logprobs = output["logprobs"]
            entropies = output["entropies"]
            step_signals = output["step_signals"]
            try:
                text = self.tokenizer.decode(token_ids)
            except Exception as e:
                print(f"[Warning] Decode failed for token_ids={token_ids}. Set the token to EOS.")
                token_ids = [self.tokenizer.eos_token_id]
                text = self.tokenizer.decode(token_ids)
            safe_outputs.append({
                "text": text,
                "token_ids": token_ids,
                "trajectory": trajectory,
                "logprobs": logprobs,
                "entropies": entropies,
                "step_signals": step_signals,
            })
        
        if use_tqdm:
            pbar.close()
        return safe_outputs

    def generate_streaming(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        max_active: int | None = None,
        use_tqdm: bool = True,
        ) -> list[str]:
        """
        Stream prompts through the engine while keeping up to `max_active` sequences running.
        As sequences finish, new prompts are added from the pending list to maximize GPU utilization.
        """
        est_blocks, est_bytes = _actual_estimate_kv_cache_usage(sampling_params.max_tokens, max_active, self.config)
        est_gib = est_bytes / (1024 ** 3)
        print(
            f"[KVCache] Estimating {est_blocks:,} blocks "
            f"({est_gib:.2f} GiB) for up to {max_active:,} active sequences "
            f"of length {sampling_params.max_tokens:,} tokens."
        )
        print(
            f"[logits] Estimating ({4 * self.config.hf_config.vocab_size * max_active * sampling_params.block_length/ (1024 ** 3):.2f}GiB) "
        )
        total = len(prompts)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * total
            self.scheduler.consistent_sampling_params = True

        if max_active is None:
            max_active = getattr(self.scheduler, "max_num_seqs", 32)

        # Interleave duplicate prompts to ensure heterogeneous batches.
        # Homogeneous batches (e.g., 64 copies of the same prompt in gen64) cause
        # numerical instability in batched GEMM/attention kernels, leading to
        # systematically different outputs vs heterogeneous batches. Round-robin
        # through prompt groups so each batch contains diverse prompts.
        from collections import defaultdict
        groups = defaultdict(list)
        for i, p in enumerate(prompts):
            key = tuple(p) if isinstance(p, list) else p
            groups[key].append(i)
        if any(len(g) > 1 for g in groups.values()):
            interleaved_indices: list[int] = []
            iters = [iter(g) for g in groups.values()]
            while iters:
                next_iters = []
                for it in iters:
                    try:
                        interleaved_indices.append(next(it))
                        next_iters.append(it)
                    except StopIteration:
                        pass
                iters = next_iters
            ordered_prompts = [prompts[i] for i in interleaved_indices]
            ordered_sampling_params = [sampling_params[i] for i in interleaved_indices]
        else:
            interleaved_indices = list(range(total))
            ordered_prompts = list(prompts)
            ordered_sampling_params = list(sampling_params)

        if use_tqdm:
            pbar = tqdm(total=total, desc="Generating", dynamic_ncols=True)

        outputs: dict[int, list[int]] = {}
        pending_idx = 0
        stall_counter = 0

        # Track seq_id -> original prompt index for output reordering at the end.
        seq_to_orig: dict[int, int] = {}

        def _add_request_tracked(prompt, sp, orig_idx):
            self.add_request(prompt, sp)
            # The Sequence just created has the highest seq_id; find it.
            # add_request appends to either running or waiting_prefill.
            if self.scheduler.waiting_prefill:
                new_seq = self.scheduler.waiting_prefill[-1]
            else:
                new_seq = self.scheduler.running[-1]
            seq_to_orig[new_seq.seq_id] = orig_idx

        # Prime initial requests up to capacity
        initial = min(max_active, total)
        for i in range(initial):
            _add_request_tracked(ordered_prompts[i], ordered_sampling_params[i], interleaved_indices[i])
        pending_idx = initial

        total_generated_tokens = 0
        start_time = perf_counter()

        while not self.is_finished() or pending_idx < total:
            # Top up to capacity before each step
            running = getattr(self.scheduler, "running", [])
            deficit = max_active - len(running)
            while deficit > 0 and pending_idx < total:
                _add_request_tracked(
                    ordered_prompts[pending_idx], ordered_sampling_params[pending_idx],
                    interleaved_indices[pending_idx])
                pending_idx += 1
                deficit -= 1

            output, num_processed = self.step()

            if not output and not self.is_finished() and num_processed == 0 and pending_idx == total:
                stall_counter += 1
                if stall_counter > 3:
                    print("\n[Warning] Deadlock detected: No progress can be made because all sequences are waiting for KV cache blocks, but no blocks are free. This can happen if the batch size is too large for the available KV cache. Try reducing the number of concurrent requests.")
                    break
            else:
                stall_counter = 0
            
            total_generated_tokens += num_processed

            if use_tqdm:
                throughput = total_generated_tokens / \
                    (perf_counter() - start_time)
                pbar.set_postfix(
                    {"Throughput": f"{int(throughput)} tok/s"})
                pbar.update(len(output))

            for seq_id, token_ids, trajectory, logprobs, entropies, step_signals in output:
                outputs[seq_id] = {
                    "token_ids": token_ids,
                    "trajectory": trajectory,
                    "logprobs": logprobs,
                    "entropies": entropies,
                    "step_signals": step_signals,
                }

        # Reorder outputs to match the original prompt order (undo interleaving)
        outputs_by_orig_idx: dict[int, dict] = {}
        for seq_id, out in outputs.items():
            orig_idx = seq_to_orig.get(seq_id, seq_id)
            outputs_by_orig_idx[orig_idx] = out
        outputs = [outputs_by_orig_idx[i] for i in range(total)]

        safe_outputs = []
        for output in outputs:
            token_ids = output["token_ids"]
            trajectory = output["trajectory"]
            logprobs = output["logprobs"]
            entropies = output["entropies"]
            step_signals = output["step_signals"]
            try:
                text = self.tokenizer.decode(token_ids)
            except Exception as e:
                print(f"[Warning] Decode failed for token_ids={token_ids}. Set the token to EOS.")
                token_ids = [self.tokenizer.eos_token_id]
                text = self.tokenizer.decode(token_ids)
            safe_outputs.append({
                "text": text,
                "token_ids": token_ids,
                "trajectory": trajectory,
                "logprobs": logprobs,
                "entropies": entropies,
                "step_signals": step_signals,
            })

        if use_tqdm:
            pbar.close()
        elapsed = perf_counter() - start_time
        depth_str = " ".join(f"d{k}={v}" for k, v in sorted(self._chain_depth_counts.items()))
        print(f"[Chain] attempts={self._chain_attempts} successes={self._chain_successes} "
              f"hit_rate={self._chain_successes/max(self._chain_attempts,1)*100:.1f}% "
              f"depths=[{depth_str}] "
              f"wall={elapsed:.1f}s tok/s={total_generated_tokens/max(elapsed,1e-9):.0f}")
        return safe_outputs
