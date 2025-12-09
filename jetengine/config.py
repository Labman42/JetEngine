import os
import torch
from dataclasses import dataclass
from transformers import AutoConfig


def get_cfg_alias(cfg, name, *candidates):
    # returns the first attribute that exists
    for key in (name, *candidates):
        if hasattr(cfg, key):
            return getattr(cfg, key)
    raise AttributeError(f"{name} not found on config")

@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 1024*128
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    diversity_enforce: bool = False
    epsilon_greedy: bool = False
    epsilon: float = 0.1
    diversity_enforce_barrier: int = 100
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    mask_token_id: int = -1
    block_length: int = 4
    dtype: str = 'auto'

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        cfg = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.hf_config = cfg
        
        # Determine torch_dtype
        if self.dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif self.dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            # auto
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.torch_dtype = torch.bfloat16
            else:
                self.torch_dtype = torch.float16

        if "llada" in cfg.model_type.lower():
            self.head_dim               = get_cfg_alias(cfg, "d_model") // get_cfg_alias(cfg, "n_heads")
            self.hidden_size            = get_cfg_alias(cfg, "d_model")
            self.max_position_embeddings= get_cfg_alias(cfg, "max_sequence_length")
            self.num_attention_heads    = get_cfg_alias(cfg, "n_heads")
            self.num_key_value_heads    = get_cfg_alias(cfg, "n_kv_heads")
            self.num_hidden_layers      = get_cfg_alias(cfg, "n_layers")
            # self.torch_dtype            = torch.bfloat16
        else:
            # standard HF configs
            self.hidden_size            = get_cfg_alias(cfg, "hidden_size")
            self.num_attention_heads    = get_cfg_alias(cfg, "num_attention_heads")
            self.num_key_value_heads    = get_cfg_alias(cfg, "num_key_value_heads")
            self.num_hidden_layers      = get_cfg_alias(cfg, "num_hidden_layers")
            self.max_position_embeddings= get_cfg_alias(cfg, "max_position_embeddings")
            self.head_dim               = get_cfg_alias(cfg, "head_dim")
            # self.torch_dtype            = get_cfg_alias(cfg, "torch_dtype")

        self.max_model_len = min(self.max_model_len, self.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        assert self.mask_token_id != -1, "Mask token ID must be set"
