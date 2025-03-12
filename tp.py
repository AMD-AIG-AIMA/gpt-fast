import os
from typing import List, Optional

import torch
import torch.distributed as dist
from torch import nn
from model import Attention, FeedForward, Transformer, CrossAttention
from quantize import WeightOnlyInt4Linear


def _get_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def _get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

def is_local() -> bool:
    return _get_rank() == 0

def local_break():
    if is_local():
        breakpoint()
    dist.barrier()

def init_dist() -> Optional[int]:
    """Initializes the distributed process group if applicable."""
    try:
        rank = _get_rank()
        world_size = _get_world_size()
        
        if world_size < 2:
            return None  # No parallelization needed

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
            torch.cuda.set_device(rank)
        return rank
    except KeyError:
        return None  # Not run via torchrun, no-op

def shard_tensor(tensor: torch.Tensor, dim: int, world_size: int, rank: int) -> torch.Tensor:
    assert tensor.size(dim=dim) % world_size == 0, f"Cannot evenly shard tensor along dim {dim}"
    return tensor.chunk(world_size, dim=dim)[rank]

def shard_qkv(qkv: torch.Tensor, dim: int, weight_splits: List[int], world_size: int, rank: int) -> torch.Tensor:
    if len(weight_splits) == 3:
        q, k, v = qkv.split(weight_splits, dim=dim)
        return torch.cat([shard_tensor(t, dim, world_size, rank) for t in (q, k, v)], dim=dim)
    elif len(weight_splits)==2:
        k, v = qkv.split(weight_splits, dim=dim)
        return torch.cat([shard_tensor(t, dim, world_size, rank) for t in (k, v)], dim=dim)
    else:
        raise NotImplementedError(f"The sharding for size {len(weight_splits)} is not implemented!")

def apply_tp_linear(linear: nn.Linear, style: str, weight_splits: List[int] = []) -> None:
    rank = _get_rank()
    world_size = _get_world_size()
    
    assert style in {"colwise", "rowwise"}, "Invalid style for tensor parallelism"
    shard_dim, size_attr = (0, "out_features") if style == "colwise" else (1, "in_features")
    
    if getattr(linear, size_attr) % world_size != 0:
        raise ValueError(f"Cannot evenly shard {size_attr} ({getattr(linear, size_attr)}) across {world_size} GPUs")
    
    # Shard bias if applicable
    if linear.bias is not None and style == "colwise":
        linear.bias = nn.Parameter(
            shard_qkv(linear.bias, 0, weight_splits, world_size, rank) if weight_splits else shard_tensor(linear.bias, 0, world_size, rank),
            requires_grad=False,
        )
    
    # Shard weights
    if isinstance(linear, WeightOnlyInt4Linear):
        linear.weight = nn.Parameter(shard_qkv(linear.weight, shard_dim, [s // 8 for s in weight_splits], world_size, rank), requires_grad=False)
        linear.scales_and_zeros = shard_qkv(linear.scales_and_zeros, 1 - shard_dim, weight_splits, world_size, rank)
    else:
        linear.weight = nn.Parameter(shard_qkv(linear.weight, shard_dim, weight_splits, world_size, rank) if weight_splits else shard_tensor(linear.weight, shard_dim, world_size, rank), requires_grad=False)
    
    if hasattr(linear, "scales") and style == "colwise":
        linear.scales = shard_qkv(linear.scales, 0, weight_splits, world_size, rank)
    
    setattr(linear, size_attr, getattr(linear, size_attr) // world_size)

def reduce_hook(module, input, output):
    dist.all_reduce(output[0], op=dist.ReduceOp.SUM)

def register_tp_hooks(module):
    world_size = _get_world_size()
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if not hasattr(module, 'process_group'):
        module.process_group = dist.new_group(ranks=list(range(world_size)))
    module.register_forward_hook(reduce_hook)

def apply_tp_ffn(mlp: FeedForward) -> None:
    for layer, style in zip([mlp.w1, mlp.w3, mlp.w2], ["colwise", "colwise", "rowwise"]):
        apply_tp_linear(layer, style)
    register_tp_hooks(mlp)

def apply_tp_attn(attn: Attention) -> None:
    world_size = _get_world_size()
    
    if any(dim % world_size != 0 for dim in [attn.n_head, attn.dim, attn.n_local_heads]):
        raise ValueError(f"Attention dimensions ({attn.n_head}, {attn.dim}, {attn.n_local_heads}) are not divisible by {world_size} GPUs")
    
    apply_tp_linear(attn.wqkv, "colwise", [attn.dim, attn.n_local_heads * attn.head_dim, attn.n_local_heads * attn.head_dim])
    apply_tp_linear(attn.wo, "rowwise")
    
    attn.n_head //= world_size
    attn.dim //= world_size
    attn.head_dim = attn.dim // attn.n_head
    attn.n_local_heads //= world_size
    
    register_tp_hooks(attn)

def apply_tp_cross_attn(cross_attn: CrossAttention) -> None:
    world_size = _get_world_size()
    
    if any(dim % world_size != 0 for dim in [cross_attn.n_head, cross_attn.dim, cross_attn.n_local_heads]):
        raise ValueError(f"Attention dimensions ({cross_attn.n_head}, {cross_attn.dim}, {cross_attn.n_local_heads}) are not divisible by {world_size} GPUs")
    
    apply_tp_linear(cross_attn.wq, "colwise")
    apply_tp_linear(cross_attn.wkv, "colwise", [cross_attn.n_local_heads * cross_attn.head_dim, cross_attn.n_local_heads * cross_attn.head_dim])
    apply_tp_linear(cross_attn.wo, "rowwise")
    
    cross_attn.n_head //= world_size
    cross_attn.dim //= world_size
    cross_attn.head_dim = cross_attn.dim // cross_attn.n_head
    cross_attn.n_local_heads //= world_size
    
    register_tp_hooks(cross_attn)

def apply_tp_transformer(transformer: Transformer) -> None:
    world_size = _get_world_size()
    
    if any(getattr(transformer.config, attr) % world_size != 0 for attr in ["n_head", "dim", "n_local_heads"]):
        raise ValueError(f"Transformer dimensions are not divisible by {world_size} GPUs")
    
    transformer.config.n_head //= world_size
    transformer.config.dim //= world_size
    transformer.config.n_local_heads //= world_size

def apply_tp(model: Transformer) -> None:
    try:
        apply_tp_transformer(model)
        for block in model.layers:
            apply_tp_ffn(block.feed_forward)
            if getattr(block, "attention", False):
                apply_tp_attn(block.attention)
            elif getattr(block, 'cross_attention', False):
                apply_tp_cross_attn(block.cross_attention)
    except ValueError as e:
        print(f"Error applying tensor parallelism: {e}\nEnsure model dimensions are divisible by number of GPUs.")
        raise

def barrier():
    print('Performing barrier')
    torch.cuda.synchronize()
    dist.barrier()
    
def broadcast(tensor, src):
    dist.broadcast(tensor, src=src)
    
def is_dist_initialized():
    return dist.is_initialized()

def reduce(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)