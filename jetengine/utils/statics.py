import math
from jetengine.config import Config

def _estimate_kv_cache_usage(config: Config) -> tuple[int, int]:
    tokens_per_sequence = config.max_model_len
    blocks_per_sequence = math.ceil(tokens_per_sequence / config.kvcache_block_size)
    total_blocks = blocks_per_sequence * config.max_num_seqs

    num_kv_heads = config.num_key_value_heads // config.tensor_parallel_size
    block_bytes = (
        2
        * config.num_hidden_layers
        * config.kvcache_block_size
        * num_kv_heads
        * config.head_dim
        * config.torch_dtype.itemsize
    )
    total_bytes = total_blocks * block_bytes
    return total_blocks, total_bytes

def _actual_estimate_kv_cache_usage(max_lengths: int, batch_size: int, config: Config) -> tuple[int, int]:
    tokens_per_sequence = max_lengths
    blocks_per_sequence = math.ceil(tokens_per_sequence / config.kvcache_block_size)
    total_blocks = blocks_per_sequence * batch_size

    num_kv_heads = config.num_key_value_heads // config.tensor_parallel_size
    block_bytes = (
        2
        * config.num_hidden_layers
        * config.kvcache_block_size
        * num_kv_heads
        * config.head_dim
        * config.torch_dtype.itemsize
    )
    total_bytes = total_blocks * block_bytes
    return total_blocks, total_bytes