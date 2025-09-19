import atexit
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
from jetengine.engine.distributed_manager import DistributedManager


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.dist_manager = DistributedManager(config.tensor_parallel_size)

        self.model_runner = ModelRunner(config, self.dist_manager)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, use_fast=True, trust_remote_code=True)
        config.eos = self.tokenizer.eos_token_id
        config.mask_token_id = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id is not None else self.tokenizer.pad_token_id
        assert config.mask_token_id is not None, "Model tokenizer must have a mask_token_id or pad_token_id"

        self.config = config
        self.scheduler = Scheduler(config)
        self.scheduler.consistent_sampling_params = True
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
            torch.set_default_dtype(self.config.hf_config.torch_dtype)

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
        seq = Sequence(prompt, self.config.mask_token_id, sampling_params)
        seq.eos_token_id = self.tokenizer.eos_token_id
        self.scheduler.add(seq)

    def step(self):
        scheduled_seqs, run_type = self.scheduler.schedule()
        if scheduled_seqs is None:
            return [], 0  # Nothing to run

        logits = self.model_runner.run(scheduled_seqs, run_type)
        self.scheduler.postprocess(scheduled_seqs, logits, run_type)

        finished_outputs = [(seq.seq_id, seq.completion_token_ids)
                            for seq in scheduled_seqs if seq.is_finished]

        # Throughput calculation needs to be adapted for block-wise generation
        num_tokens = [self.scheduler.running[i].num_to_transfer if hasattr(
            self.scheduler.running[i], 'num_to_transfer') else 0 for i in range(len(self.scheduler.running))]
        return finished_outputs, sum(num_tokens)

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

        total_generated_tokens = 0
        start_time = perf_counter()

        while not self.is_finished():
            output, num_processed = self.step()
            total_generated_tokens += num_processed

            throughput = total_generated_tokens / \
                (perf_counter() - start_time)
            if use_tqdm:
                pbar.set_postfix(
                    {"Throughput": f"{int(throughput)} tok/s"})

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(
            token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs

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
            total_generated_tokens += num_processed

            if use_tqdm:
                throughput = total_generated_tokens / \
                    (perf_counter() - start_time + 1e-6)
                pbar.set_postfix(
                    {"Throughput": f"{int(throughput)} tok/s"})
                pbar.update(len(output))

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids

        outputs_list = [outputs[seq_id] for seq_id in sorted(outputs)]
        results = [{"text": self.tokenizer.decode(
            token_ids), "token_ids": token_ids} for token_ids in outputs_list]

        if use_tqdm:
            pbar.close()
        return results
