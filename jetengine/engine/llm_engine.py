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
from jetengine.engine.sequence import Sequence, RunType
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
            logits = self.model_runner.run(schedule_result.denoise, RunType.DENOISE)
            finished_sequences.extend(
                postprocess_fn(schedule_result.denoise, logits, RunType.DENOISE)
            )
            tokens_generated = sum(
                getattr(seq, "num_to_transfer", 0) for seq in schedule_result.denoise
            )

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
                (seq.seq_id, seq.token_ids, seq.trajectory, seq.logprobs, seq.entropies)
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

            for seq_id, token_ids, trajectory, logprobs, entropies in output:  # Updated unpacking
                outputs[seq_id] = {
                    "token_ids": token_ids, 
                    "trajectory": trajectory,
                    "logprobs": logprobs,  # NEW
                    "entropies": entropies  # NEW
                }
                if use_tqdm:
                    pbar.update(1)

        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        
        safe_outputs = []
        for output in outputs:
            token_ids = output["token_ids"]
            trajectory = output["trajectory"]
            logprobs = output["logprobs"]  # NEW
            entropies = output["entropies"]  # NEW
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
                "logprobs": logprobs,  # NEW
                "entropies": entropies  # NEW
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

        if use_tqdm:
            pbar = tqdm(total=total, desc="Generating", dynamic_ncols=True)

        outputs: dict[int, list[int]] = {}
        pending_idx = 0
        stall_counter = 0

        # Prime initial requests up to capacity
        initial = min(max_active, total)
        for i in range(initial):
            self.add_request(prompts[i], sampling_params[i])
        pending_idx = initial

        total_generated_tokens = 0
        start_time = perf_counter()

        while not self.is_finished() or pending_idx < total:
            # Top up to capacity before each step
            running = getattr(self.scheduler, "running", [])
            deficit = max_active - len(running)
            while deficit > 0 and pending_idx < total:
                self.add_request(
                    prompts[pending_idx], sampling_params[pending_idx])
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

            for seq_id, token_ids, trajectory, logprobs, entropies in output:  # Updated unpacking
                outputs[seq_id] = {
                    "token_ids": token_ids, 
                    "trajectory": trajectory,
                    "logprobs": logprobs,  # NEW
                    "entropies": entropies  # NEW
                }

        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        
        safe_outputs = []
        for output in outputs:
            token_ids = output["token_ids"]
            trajectory = output["trajectory"]
            logprobs = output["logprobs"]  # NEW
            entropies = output["entropies"]  # NEW
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
                "logprobs": logprobs,  # NEW
                "entropies": entropies  # NEW
            })

        if use_tqdm:
            pbar.close()
        return safe_outputs
