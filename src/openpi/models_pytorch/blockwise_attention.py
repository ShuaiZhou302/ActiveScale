"""Exact block-causal attention using PyTorch's packed FlashAttention kernel."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackedBlockAttentionPlan:
    query_indices: torch.Tensor
    key_indices: torch.Tensor
    cumulative_query: torch.Tensor
    cumulative_key: torch.Tensor
    max_query: int
    max_key: int
    causal_query_indices: torch.Tensor
    causal_key_indices: torch.Tensor
    causal_cumulative_query: torch.Tensor
    causal_cumulative_key: torch.Tensor
    causal_max_query: int
    causal_max_key: int
    invalid_indices: torch.Tensor


def _cumulative_lengths(lengths: list[int], device: torch.device) -> torch.Tensor:
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    return torch.tensor(cumulative, dtype=torch.int32, device=device)


def build_packed_block_attention_plan(
    pad_mask: torch.Tensor,
    boundary_mask: torch.Tensor,
    *,
    device: torch.device,
) -> PackedBlockAttentionPlan:
    """Build packed full-block and causal-tail sequences from PI0.5 masks.

    The compact masks are copied to CPU because their shape is only ``[B, L]``
    and plan construction is reused by every Gemma layer. A suffix containing
    at least two consecutive valid boundaries is the FAST autoregressive
    response. It is represented by one lower-right causal FlashAttention call;
    all earlier blocks retain PI0.5's bidirectional-within-block semantics.
    """
    if pad_mask.ndim != 2 or boundary_mask.ndim != 2:
        raise ValueError(f"compact attention masks must be [B, L], got {pad_mask.shape} and {boundary_mask.shape}")
    if pad_mask.shape != boundary_mask.shape:
        raise ValueError(f"compact attention mask shapes differ: {pad_mask.shape} vs {boundary_mask.shape}")

    valid = pad_mask.detach().to(device="cpu", dtype=torch.bool)
    boundary = boundary_mask.detach().to(device="cpu", dtype=torch.bool)
    block_ids = boundary.to(torch.int64).cumsum(dim=1)
    batch_size, length = valid.shape
    positions = torch.arange(length)

    query_indices: list[torch.Tensor] = []
    key_indices: list[torch.Tensor] = []
    query_lengths: list[int] = []
    key_lengths: list[int] = []
    causal_query_indices: list[torch.Tensor] = []
    causal_key_indices: list[torch.Tensor] = []
    causal_query_lengths: list[int] = []
    causal_key_lengths: list[int] = []
    invalid_indices: list[torch.Tensor] = []

    for batch_index in range(batch_size):
        base = batch_index * length
        sample_valid = valid[batch_index]
        sample_blocks = block_ids[batch_index]
        valid_positions = torch.nonzero(sample_valid, as_tuple=False).flatten()
        if not valid_positions.numel():
            raise ValueError(f"attention sample {batch_index} has no valid tokens")

        invalid_indices.append(torch.nonzero(~sample_valid, as_tuple=False).flatten() + base)
        final_valid = int(valid_positions[-1])
        causal_start = final_valid + 1
        while causal_start > 0 and sample_valid[causal_start - 1] and boundary[batch_index, causal_start - 1]:
            causal_start -= 1
        causal_mask = sample_valid & (positions >= causal_start)
        if int(causal_mask.sum()) < 2:
            causal_mask.zero_()

        noncausal_valid = sample_valid & ~causal_mask
        for block_id in torch.unique(sample_blocks[noncausal_valid], sorted=True).tolist():
            query = torch.nonzero(noncausal_valid & (sample_blocks == block_id), as_tuple=False).flatten()
            key = torch.nonzero(sample_valid & (sample_blocks <= block_id), as_tuple=False).flatten()
            query_indices.append(query + base)
            key_indices.append(key + base)
            query_lengths.append(query.numel())
            key_lengths.append(key.numel())

        if bool(causal_mask.any()):
            causal_query = torch.nonzero(causal_mask, as_tuple=False).flatten()
            causal_key = valid_positions
            causal_query_indices.append(causal_query + base)
            causal_key_indices.append(causal_key + base)
            causal_query_lengths.append(causal_query.numel())
            causal_key_lengths.append(causal_key.numel())

    empty_long = torch.empty(0, dtype=torch.long, device=device)
    return PackedBlockAttentionPlan(
        query_indices=torch.cat(query_indices).to(device=device) if query_indices else empty_long,
        key_indices=torch.cat(key_indices).to(device=device) if key_indices else empty_long,
        cumulative_query=_cumulative_lengths(query_lengths, device),
        cumulative_key=_cumulative_lengths(key_lengths, device),
        max_query=max(query_lengths, default=0),
        max_key=max(key_lengths, default=0),
        causal_query_indices=torch.cat(causal_query_indices).to(device=device) if causal_query_indices else empty_long,
        causal_key_indices=torch.cat(causal_key_indices).to(device=device) if causal_key_indices else empty_long,
        causal_cumulative_query=_cumulative_lengths(causal_query_lengths, device),
        causal_cumulative_key=_cumulative_lengths(causal_key_lengths, device),
        causal_max_query=max(causal_query_lengths, default=0),
        causal_max_key=max(causal_key_lengths, default=0),
        invalid_indices=torch.cat(invalid_indices).to(device=device),
    )


def _packed_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cumulative_query: torch.Tensor,
    cumulative_key: torch.Tensor,
    max_query: int,
    max_key: int,
    *,
    scale: float,
    is_causal: bool,
) -> torch.Tensor:
    if not hasattr(torch.ops.aten, "_flash_attention_forward"):
        raise RuntimeError("this PyTorch build does not expose aten._flash_attention_forward")
    return torch.ops.aten._flash_attention_forward(
        query,
        key,
        value,
        cumulative_query,
        cumulative_key,
        max_query,
        max_key,
        0.0,
        is_causal,
        False,
        scale=scale,
        window_size_left=-1,
        window_size_right=-1,
    )[0]


def blockwise_varlen_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: PackedBlockAttentionPlan,
    *,
    scale: float,
) -> torch.Tensor:
    """Apply the exact PI0.5 mask and return ``[B, Hq, L, D]``."""
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("packed FlashAttention requires CUDA tensors")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key and value must be [B, H, L, D]")
    if key.shape != value.shape:
        raise ValueError(f"key/value shapes differ: {key.shape} vs {value.shape}")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError(f"query/key sequence shapes differ: {query.shape} vs {key.shape}")
    if query.shape[1] % key.shape[1]:
        raise ValueError(f"query heads {query.shape[1]} must be divisible by KV heads {key.shape[1]}")

    batch_size, query_heads, length, head_dim = query.shape
    key_value_heads = key.shape[1]
    flat_query = query.transpose(1, 2).reshape(batch_size * length, query_heads, head_dim)
    flat_key = key.transpose(1, 2).reshape(batch_size * length, key_value_heads, head_dim)
    flat_value = value.transpose(1, 2).reshape(batch_size * length, key_value_heads, head_dim)

    flat_output = torch.zeros_like(flat_query)
    if plan.query_indices.numel():
        packed_output = _packed_flash_attention(
            flat_query.index_select(0, plan.query_indices),
            flat_key.index_select(0, plan.key_indices),
            flat_value.index_select(0, plan.key_indices),
            plan.cumulative_query,
            plan.cumulative_key,
            plan.max_query,
            plan.max_key,
            scale=scale,
            is_causal=False,
        )
        flat_output = flat_output.index_copy(0, plan.query_indices, packed_output)

    if plan.causal_query_indices.numel():
        causal_output = _packed_flash_attention(
            flat_query.index_select(0, plan.causal_query_indices),
            flat_key.index_select(0, plan.causal_key_indices),
            flat_value.index_select(0, plan.causal_key_indices),
            plan.causal_cumulative_query,
            plan.causal_cumulative_key,
            plan.causal_max_query,
            plan.causal_max_key,
            scale=scale,
            is_causal=True,
        )
        flat_output = flat_output.index_copy(0, plan.causal_query_indices, causal_output)

    if plan.invalid_indices.numel():
        invalid_value = flat_value.index_select(0, plan.invalid_indices)
        invalid_value = invalid_value.repeat_interleave(query_heads // key_value_heads, dim=1)
        flat_output = flat_output.index_copy(0, plan.invalid_indices, invalid_value)

    return flat_output.reshape(batch_size, length, query_heads, head_dim).transpose(1, 2)
