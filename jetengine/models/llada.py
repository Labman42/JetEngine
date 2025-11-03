# Create this new file: jetengine/models/llada.py

import torch
from torch import nn
import torch.distributed as dist

from jetengine.layers.activation import SiluAndMul
from jetengine.layers.attention import LladaBlockAttention
from jetengine.layers.layernorm import RMSNorm
from jetengine.layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear
)
from jetengine.layers.rotary_embedding import get_rope
from jetengine.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class LladaDecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        process_group
    ) -> None:
        super().__init__()
        
        # --- Attention Components (Inlined) ---
        tp_size = dist.get_world_size(group=process_group)
        self.total_num_heads = config.n_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.n_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = config.d_model // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.d_model,
            self.head_dim,
            process_group,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, 'attention_bias', False),
        )

        self.attn_out = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.d_model,
            process_group,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_sequence_length,
            base=getattr(config, "rope_theta", 500000.0),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        # Use the same BlockAttention as SDAR, as requested
        self.attn = LladaBlockAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        
        # --- MLP Components (Inlined) ---
        self.gate_up_proj = MergedColumnParallelLinear(
            config.d_model,
            [config.mlp_hidden_size] * 2,
            process_group,
            bias=False,
        )
        self.ff_out = RowParallelLinear(
            config.mlp_hidden_size,
            config.d_model,
            process_group,
            bias=False,
        )
        assert config.activation_type == "silu"
        self.act_fn = SiluAndMul()

        # --- Layer-level LayerNorms (from weight_map) ---
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ff_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if residual is None:
            residual = hidden_states
            hidden_states = self.attn_norm(hidden_states)
        else:
            hidden_states, residual = self.attn_norm(hidden_states, residual)
            
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        hidden_states = self.attn_out(o)
        
        hidden_states, residual = self.ff_norm(hidden_states, residual)
        
        gate_up = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states = self.ff_out(hidden_states)
        
        return hidden_states, residual


class LladaModel(nn.Module):
    """
    This class now represents the 'transformer' block from the weight_map.
    """
    def __init__(
        self,
        config,
        process_group
    ) -> None:
        super().__init__()
        # model.transformer.wte.weight
        self.wte = VocabParallelEmbedding(
            config.vocab_size, config.d_model, process_group)
        # model.transformer.blocks.N...
        self.blocks = nn.ModuleList([LladaDecoderLayer(
            config, process_group) for _ in range(config.n_layers)])
        # model.transformer.ln_f.weight
        self.ln_f = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        # model.transformer.ff_out.weight
        self.ff_out = ParallelLMHead(
            config.vocab_size, config.d_model, process_group)
            
        if config.weight_tying:
            self.ff_out.weight.data = self.wte.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids)
        residual = None
        for layer in self.blocks:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.ln_f(hidden_states, residual)
        return hidden_states


class LladaForCausalLM(nn.Module):
    """
    This is the top-level model wrapper.
    """
    
    # This mapping tells the loader how to handle the fused weights.
    # The loader will find "model.transformer.blocks.N.ff_proj.weight"
    # and map it to shard 0 of "model.transformer.blocks.N.gate_up_proj.weight".
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "ff_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config,
        process_group
    ) -> None:
        super().__init__()
        # This 'model' attribute matches the 'model.' prefix in the weight_map
        self.model = nn.Module()
        # This 'transformer' attribute matches 'model.transformer.'
        self.model.transformer = LladaModel(config, process_group)
                
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # Get hidden states from the LladaModel
        hidden_states = self.model.transformer(input_ids, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Compute logits using the head inside LladaModel
        logits = self.model.transformer.ff_out(hidden_states)
        return logits