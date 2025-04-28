# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from dataclasses import dataclass, field
from typing import Optional, Union, Dict

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


from multimodal.mm_config import MultimodalModelArgs, QwenVisionModelArgs
from multimodal.qwen2_5vl.preprocessing import get_rope_index

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    rope_scaling: Optional[dict] = None
    wqkv_bias: Optional[bool] = False
    mm_config: Optional[MultimodalModelArgs] = None
    cross_attention_layers: Optional[list] = field(default_factory=list) 
    moe_layers: Optional[Union[int, list]] = field(default_factory=list)
    moe_num_experts: Optional[int] = None
    num_activated_experts: Optional[int] = None

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head
        if isinstance(self.moe_layers, int):
            # This is for LLama 4 when "interleave_moe_layer_step" is provided instead of MoE layers
            self.moe_layers = list(range(self.moe_layers-1, self.n_layer, self.moe_layers))

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config.lower() in str(name).lower()]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match
            
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, rope_base=1000000),
    "AMD-Llama-135m-code": dict(block_size=2048,vocab_size=32000,n_layer=12,n_head=12,dim=768,intermediate_size=2048,n_local_heads=12,head_dim=64,rope_base=10000.0,norm_eps=1e-5),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
    "Mistral-7B": dict(n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=32000),
    "stories15M": dict(n_layer=6, n_head=6, dim=288),
    "stories110M": dict(n_layer=12, n_head=12, dim=768),

    "llama-3-8b": dict(block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000),
    "llama-3-70b": dict(block_size=8192, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000),
    "llama-3.1-8b": dict(block_size=131072, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
    ),
    "Llama-3.1-70B-Instruct": dict(block_size=131072, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
    ),
    "llama-3.1-405b": dict(block_size=131072, n_layer=126, n_head=128, n_local_heads=8, dim=16384, intermediate_size=53248, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
    ),
    "Llama-3.2-1B-instruct": dict(block_size=131072, n_layer=16, n_head=32, n_local_heads=8, dim=2048, intermediate_size=8192, vocab_size=128256,
        rope_base=500000.0, rope_scaling=dict(factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192, rope_type="llama3"),
        norm_eps=1e-5, head_dim=64
    ),
    "Qwen2-0.5B-Instruct":dict(block_size=8192, n_layer=24, n_head=14, n_local_heads=2, dim=896, intermediate_size=4864, vocab_size=151936,
        rope_base=1000000.0, norm_eps=1e-6, head_dim=64, wqkv_bias=True,
    ),
    "Qwen2-7B-Instruct":dict(block_size=131072, n_layer=28, n_head=28, n_local_heads=4, dim=3584, intermediate_size=18944, vocab_size=152064,
        rope_base=1000000.0, norm_eps=1e-6, head_dim=128, wqkv_bias=True,
    ),
    "llava-onevision-qwen2-7b-si":dict(block_size=131072, n_layer=28, n_head=28, n_local_heads=4, dim=3584, intermediate_size=18944, vocab_size=152064,
        rope_base=1000000.0, norm_eps=1e-6, head_dim=128, wqkv_bias=True, mm_config=MultimodalModelArgs.from_name("llava-onevision-qwen2-7b-si")
    ),
    "llava-onevision-qwen2-0.5b-si":dict(block_size=8192, n_layer=24, n_head=14, n_local_heads=2, dim=896, intermediate_size=4864, vocab_size=151936,
        rope_base=1000000.0, norm_eps=1e-6, head_dim=64, wqkv_bias=True, mm_config=MultimodalModelArgs.from_name("llava-onevision-qwen2-0.5b-si")
    ),
    "llava-onevision-qwen2-72b-si":dict(block_size=131072, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=29568, vocab_size=152064,
        rope_base=1000000.0, norm_eps=1e-6, head_dim=128, wqkv_bias=True, mm_config=MultimodalModelArgs.from_name("llava-onevision-qwen2-72b-si")
    ),
    "llama-3.2-11b-vision-instruct": dict(block_size=131072, n_layer=40, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192), cross_attention_layers=[3,8,13,18,23,28,33,38],
        mm_config=MultimodalModelArgs.from_name("llama-3.2-11b-vision-instruct")
    ),
    "llama-3.2-90b-vision-instruct": dict(block_size=131072, n_layer=100, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192), 
        cross_attention_layers=[3,8,13,18,23,28,33,38,43,48,53,58,63,68,73,78,83,88,93,98],
        mm_config=MultimodalModelArgs.from_name("llama-3.2-90b-vision-instruct")
    ),
    "Qwen2.5-VL-3B-Instruct":dict(block_size=32768, n_layer=36, n_head=16, n_local_heads=2, dim=2048, intermediate_size=11008, vocab_size=151936,
        rope_base=1000000.0, norm_eps=1e-6, head_dim=128, wqkv_bias=True, mm_config=MultimodalModelArgs.from_name("Qwen2.5-VL-3B-Instruct")
    ),
    "Qwen2.5-VL-7B-Instruct":dict(block_size=32768, n_layer=28, n_head=28, n_local_heads=4, dim=3584, intermediate_size=18944, vocab_size=152064,
        rope_base=1000000.0, norm_eps=1e-6, head_dim=128, wqkv_bias=True, mm_config=MultimodalModelArgs.from_name("Qwen2.5-VL-7B-Instruct")
    ),
    "Qwen2.5-VL-72B-Instruct":dict(block_size=32768, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=29568, vocab_size=152064,
        rope_base=1000000.0, norm_eps=1e-6, head_dim=128, wqkv_bias=True, mm_config=MultimodalModelArgs.from_name("Qwen2.5-VL-72B-Instruct")
    ),
    "Qwen2.5-3B-Instruct":dict(block_size=32768, n_layer=36, n_head=16, n_local_heads=2, dim=2048, intermediate_size=11008, vocab_size=151936,
        rope_base=1000000.0, norm_eps=1e-6, head_dim=128, wqkv_bias=True,
    ),
    "Qwen2.5-7B-Instruct":dict(block_size=131072, n_layer=28, n_head=28, n_local_heads=4, dim=3584, intermediate_size=18944, vocab_size=152064,
        rope_base=1000000.0, norm_eps=1e-6, head_dim=128, wqkv_bias=True,
    ),
    "Llama-4-Scout-17B-16E-Instruct":dict(block_size=10485760, n_layer=48, n_head=40, n_local_heads=8, dim=5120, intermediate_size=8192, vocab_size=202048,
        rope_base=500000.0, norm_eps=1e-5, head_dim=128, moe_layers=1, moe_num_experts=16, num_activated_experts=1,
        mm_config=MultimodalModelArgs.from_name("Llama-4-Scout-17B-16E-Instruct")
    ),
}

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_cache_size, n_heads, head_dim, device, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_cache_size, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype, device=device))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out
    
    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        
class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        # Create specialized blocks at init time
        cross_attention_layers = getattr(config, "cross_attention_layers", []) if config.mm_config else []
        additional_tokens = 8 if len(cross_attention_layers)>0 else 0
        self.tok_embeddings = nn.Embedding(config.vocab_size+additional_tokens, config.dim)
        
        self.layers = nn.ModuleList()
        for i in range(config.n_layer):
            if i in cross_attention_layers:
                self.layers.append(CrossAttentionBlock(config))
            else:
                self.layers.append(TransformerBlock(config, i))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_cache_size = -1
        self.cross_attention_seq_length = -1
        self.cross_attention_mask = None
        self.cross_attention_mask_out = None
        self.mrope = False
        self._device=next(self.layers[0].parameters()).device
        if hasattr(config, 'mm_config'):
            self.mrope = getattr(config.mm_config, "mrope", False)
            self.image_grid_thw=None
        

    def setup_caches(self, max_batch_size, max_cache_size, prompt=None, cross_attention_seq_length=None, preserve_history=False):
        if self.mrope:
            position_ids, _ = get_rope_index(input_ids=prompt, image_grid_thw=getattr(self,'image_grid_thw', None), mm_config=self.config.mm_config)
        else:
            position_ids = None
        
        dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
            
        # If current cache is sufficient and we're just updating freqs_cis for mrope
        if self.max_cache_size >= max_cache_size and self.max_batch_size >= max_batch_size and (self.cross_attention_seq_length >= cross_attention_seq_length if cross_attention_seq_length is not None else True):
            if self.mrope:
                self.freqs_cis[:max_cache_size] = precompute_freqs_cis_for_qwen2_5_vl(max_cache_size, self.config.dim // self.config.n_head, 
                                                              position_ids, self.config.rope_base, dtype, self.config.rope_scaling).to(self._device)
            return
        
        head_dim = self.config.dim // self.config.n_head
        max_cache_size = find_multiple(max_cache_size, 8)
        
        # Remember old cache sizes for history preservation
        old_max_cache_size = self.max_cache_size
        old_cross_attention_seq_length = self.cross_attention_seq_length
        
        # Update metadata for new caches
        self.max_cache_size = max_cache_size
        self.max_batch_size = max_batch_size
        if cross_attention_seq_length:
            self.cross_attention_seq_length = cross_attention_seq_length
        
        # Create or update KV caches
        for b in self.layers:
            if hasattr(b, 'attention'):
                # Save reference to old cache if it exists
                old_cache = b.attention.kv_cache if hasattr(b.attention, 'kv_cache') else None
                
                # Create new cache
                b.attention.kv_cache = KVCache(max_batch_size, max_cache_size, 
                                              self.config.n_local_heads, head_dim, self._device, dtype)
                
                # Copy old values using efficient tensor operations
                if preserve_history and old_cache is not None and old_max_cache_size > 0:
                    # Use narrow + copy_ for efficient copying without allocating new tensors
                    copy_size = min(old_max_cache_size, max_cache_size)
                    b.attention.kv_cache.k_cache[:, :, :copy_size].copy_(
                        old_cache.k_cache[:, :, :copy_size])
                    b.attention.kv_cache.v_cache[:, :, :copy_size].copy_(
                        old_cache.v_cache[:, :, :copy_size])
                
            if hasattr(b, 'cross_attention'):
                # Save reference to old cross-attention cache if it exists
                old_cross_cache = b.cross_attention.kv_cache if hasattr(b.cross_attention, 'kv_cache') else None
                
                # Create new cross-attention cache
                b.cross_attention.kv_cache = KVCache(max_batch_size, cross_attention_seq_length, 
                                                   self.config.n_local_heads, head_dim, self._device, dtype)
                
                # Copy old values using efficient tensor operations
                if preserve_history and old_cross_cache is not None and old_cross_attention_seq_length > 0:
                    copy_size = min(old_cross_attention_seq_length, cross_attention_seq_length)
                    b.cross_attention.kv_cache.k_cache[:, :, :copy_size].copy_(
                        old_cross_cache.k_cache[:, :, :copy_size])
                    b.cross_attention.kv_cache.v_cache[:, :, :copy_size].copy_(
                        old_cross_cache.v_cache[:, :, :copy_size])
        
        # Update the RoPE embeddings and causal mask
        if self.mrope:
            if position_ids is None:
                raise ValueError('Multimodal Rope requires the position id')
            self.freqs_cis = precompute_freqs_cis_for_qwen2_5_vl(max_cache_size, self.config.dim // self.config.n_head, 
                                                              position_ids, self.config.rope_base, dtype, self.config.rope_scaling).to(self._device)
        else:
            self.freqs_cis = precompute_freqs_cis(max_cache_size, self.config.dim // self.config.n_head,
                                                  self.config.rope_base, dtype, self.config.rope_scaling).to(self._device)
        self.causal_mask = torch.tril(torch.ones(self.max_cache_size, self.max_cache_size, dtype=torch.bool, device=self._device))

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None, cross_states: Optional[Tensor] = None, 
                embedded: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        causal_mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        if embedded is None:
            x = self.tok_embeddings(idx)
        else:
            x = embedded

        if cross_states is not None or self.cross_attention_mask is None:
            cross_attention_mask, cross_attention_mask_out = self.cross_attention_mask, self.cross_attention_mask_out
        else:
            cross_attention_mask=self.cross_attention_mask[:,:,-1,:].repeat(1, 1, len(input_pos), 1)
            cross_attention_mask_out=self.cross_attention_mask_out[:,:,-1,:].repeat(1, 1, len(input_pos), 1)
        # Pass both freqs_cis and cross_states to all layers
        # Each block type will only use what it needs
        for layer in self.layers:
            # Each layer type will use the appropriate mask
            x = layer(x, input_pos=input_pos, freqs_cis=freqs_cis, 
                     cross_states=cross_states, 
                     mask=causal_mask,
                     cross_attention_mask=cross_attention_mask,
                     cross_attention_mask_out=cross_attention_mask_out)
        x = self.norm(x)
        logits = self.output(x)
        return logits
    
    def clear_cache(self):
        for b in self.layers:
            if hasattr(b,'attention'):
                b.attention.kv_cache.clear()
            if hasattr(b,'cross_attention'):
                b.cross_attention.kv_cache.clear()

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int = -1) -> None:
        super().__init__()
        self.attention = Attention(config)
        
        # Determine whether to use MoE or regular FFN based on layer index
        if (hasattr(config, 'moe_layers') and  config.moe_layers is not None and 
        layer_idx in config.moe_layers and config.moe_num_experts is not None):
            self.feed_forward = Llama4MOEFeedForward(config)
        else:
            self.feed_forward = FeedForward(config)
            
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor, cross_states: Optional[Tensor] = None, 
    cross_attention_mask: Optional[Tensor] = None, cross_attention_mask_out: Optional[Tensor] = None) -> Tensor:
        # Regular transformer block only uses causal mask
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=getattr(config, "wqkv_bias", False))
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_rope_scaling(freqs: torch.Tensor, rope_scaling: Optional[dict] = None):
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    rope_scaling: Optional[dict] = None,
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)

    
def flatten_freqs(freq, mrope_section):
    return torch.cat([m[i % 3] for i, m in enumerate(freq.split(mrope_section, dim=-1))], dim=-1).squeeze()

def precompute_freqs_cis_for_qwen2_5_vl(
    seq_len: int, n_elem: int, position_ids: torch.tensor, 
    base: int = 10000, 
    dtype: torch.dtype = torch.bfloat16,
    rope_scaling: Optional[dict] = None,
    mrope_section: Optional[list] = [16,24,24],
    ):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)).to(device=position_ids.device)
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    t = torch.arange(position_ids[0,0,-1]+1, position_ids[0,0,-1] + 1 + seq_len - position_ids.shape[-1]).to(device=position_ids.device)
    t = t.expand(3,position_ids.shape[1],-1)
    position_ids = torch.cat([position_ids, t], dim=-1)
    freqs_expanded = freqs[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
    freqs = torch.matmul(freqs_expanded.float(), position_ids_expanded.float()).transpose(2, 3)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([flatten_freqs(freqs_cis.real, mrope_section), flatten_freqs(freqs_cis.imag, mrope_section)], dim=-1)
    return cache.to(dtype=dtype)
    



def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)



class CrossAttentionBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.cross_attention = CrossAttention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.cross_attn_attn_gate = torch.nn.Parameter(torch.zeros(1))
        self.cross_attn_mlp_gate = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor, cross_states: Optional[Tensor] = None, 
                cross_attention_mask: Optional[Tensor] = None, cross_attention_mask_out: Optional[Tensor] = None) -> Tensor:
        if cross_states is None and cross_attention_mask is None:
            # Layer is not used, language model only
            return x
        # Cross attention block uses cross_attention_mask
        h = x + self.cross_attention(self.attention_norm(x), cross_states, cross_attention_mask, input_pos) * self.cross_attn_attn_gate.tanh()
        out = h + self.feed_forward(self.ffn_norm(h)) * self.cross_attn_mlp_gate.tanh() * cross_attention_mask_out[:,0]
        return out

class CrossAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        # Separate projections for query and cross kv
        self.wq = nn.Linear(config.dim, config.dim, bias=getattr(config, "wqkv_bias", False))
        kv_dim = 2 * config.n_local_heads * config.head_dim  # combined k and v
        self.wkv = nn.Linear(config.dim, kv_dim, bias=getattr(config, "wqkv_bias", False))
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        
        # Add normalizations for query and key
        self.q_norm = RMSNorm(config.head_dim, eps=config.norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.norm_eps)
        
        self.kv_cache = None
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim

    def forward(self, x: Tensor, cross_states: Optional[Tensor] = None, mask: Optional[Tensor] = None, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        # Project and normalize query from input sequence
        q = self.wq(x)
        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)

        if cross_states is not None:
            # Project key and value from cross attention states
            kv_size = self.n_local_heads * self.head_dim
            k, v = self.wkv(cross_states).split([kv_size, kv_size], dim=-1)
            k = k.view(bsz, -1, self.n_local_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, -1, self.n_local_heads, self.head_dim).transpose(1, 2)
            
            # Update cache if using incremental decoding
            cache_size = torch.arange(k.shape[2], device=k.device)
            if self.kv_cache is not None:
                k, v = self.kv_cache.update(cache_size, k, v)
        else:
            # During decoding (after prefill), use cached key/values
            k, v = self.kv_cache.k_cache, self.kv_cache.v_cache
        cache_size = mask.shape[-1]
        k,v = k[:,:,:cache_size,:], v[:,:,:cache_size,:]
        # Repeat KV heads if necessary
        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        # Normalize key states
        k = self.k_norm(k)
        # print(
        #     f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}, mask shape: {mask.shape}, input_pos shape: {input_pos.shape}"
        # )
        # Compute attention
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)

        return y

class Llama4ConditionalFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Replace separate w1, w2, w3 with gate_up_proj and down_proj like Llama4
        self.w1 = nn.Parameter(torch.empty(config.moe_num_experts, config.dim, 2 * config.intermediate_size))
        self.w2 = nn.Parameter(torch.empty(config.moe_num_experts, config.intermediate_size, config.dim))
        
    def forward(self, x: Tensor, expert_indices: Tensor) -> Tensor:
        # Extract expert-specific weights
        gate_up_weights = self.w1[expert_indices]  # [T, A, D, 2*I]
        down_weights = self.w2[expert_indices]  # [T, A, I, D]
        
        # Compute gate_up and split into gate and up parts
        gate_up = torch.einsum('ti,taio->tao', x, gate_up_weights)
        gate, up = gate_up.chunk(2, dim=-1)  # Split along intermediate dim
        
        # Apply SiLU activation to gate and multiply with up
        activated = F.silu(gate) * up
        
        # Project back to hidden dimension
        expert_outs = torch.einsum('tao,taoi->tai', activated, down_weights)
        return expert_outs


class Llama4MOEFeedForward(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.router = nn.Linear(config.dim, config.moe_num_experts, bias=False)
        self.cond_ffn = Llama4ConditionalFeedForward(config)
        self.dim = config.dim
        self.num_activated_experts = config.num_activated_experts
        
        # Add a shared expert like Llama4
        self.shared_expert = FeedForward(config)
        
    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        x = x.view(-1, self.dim)
        
        # Router logits and scores
        scores = self.router(x)  # [T, E]
        expert_weights = F.softmax(scores, dim=-1)
        expert_weights, expert_indices = torch.topk(expert_weights, self.num_activated_experts, dim=-1)  # [T, A], [T, A]
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True)  # [T, A]
        
        # Get outputs from conditional FFN
        expert_outs = self.cond_ffn(x, expert_indices)
        
        # Add shared expert contribution
        shared_out = self.shared_expert(x)
        
        # Combine MoE output with shared expert output
        moe_out = torch.einsum('tai,ta->ti', expert_outs, expert_weights)
        combined_out = moe_out + shared_out
        
        # Restore original shape
        return combined_out.view(original_shape)
