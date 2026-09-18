import math

import pytest
import torch

from openpi.models_pytorch.blockwise_attention import blockwise_varlen_flash_attention
from openpi.models_pytorch.blockwise_attention import build_packed_block_attention_plan


def _reference_mask(pad_mask: torch.Tensor, boundary_mask: torch.Tensor) -> torch.Tensor:
    block_ids = boundary_mask.cumsum(dim=1)
    visible = block_ids[:, None, :] <= block_ids[:, :, None]
    visible &= pad_mask[:, None, :] & pad_mask[:, :, None]
    return visible | torch.eye(pad_mask.shape[1], dtype=torch.bool)[None]


def _mask_from_plan(plan, batch_size: int, length: int) -> torch.Tensor:
    visible = torch.zeros(batch_size, length, length, dtype=torch.bool)

    for segment in range(plan.cumulative_query.numel() - 1):
        q0, q1 = plan.cumulative_query[segment : segment + 2].tolist()
        k0, k1 = plan.cumulative_key[segment : segment + 2].tolist()
        queries = plan.query_indices[q0:q1].cpu()
        keys = plan.key_indices[k0:k1].cpu()
        batch = int(queries[0]) // length
        visible[batch, queries % length, (keys % length)[:, None]] = True

    for segment in range(plan.causal_cumulative_query.numel() - 1):
        q0, q1 = plan.causal_cumulative_query[segment : segment + 2].tolist()
        k0, k1 = plan.causal_cumulative_key[segment : segment + 2].tolist()
        queries = plan.causal_query_indices[q0:q1].cpu()
        keys = plan.causal_key_indices[k0:k1].cpu()
        batch = int(queries[0]) // length
        offset = keys.numel() - queries.numel()
        for query_offset, query in enumerate(queries):
            visible[batch, int(query) % length, keys[: offset + query_offset + 1] % length] = True

    invalid = plan.invalid_indices.cpu()
    visible[invalid // length, invalid % length, invalid % length] = True
    return visible


@pytest.mark.parametrize(
    ("pad_mask", "boundary_mask"),
    [
        (
            torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool),
            torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1, 0, 0]], dtype=torch.bool),
        ),
        (
            torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool),
            torch.tensor([[1, 0, 1, 0, 0, 1, 1, 1]], dtype=torch.bool),
        ),
        (
            torch.ones(1, 5, dtype=torch.bool),
            torch.ones(1, 5, dtype=torch.bool),
        ),
    ],
)
def test_packed_plan_reconstructs_exact_pi05_mask(pad_mask, boundary_mask):
    plan = build_packed_block_attention_plan(pad_mask, boundary_mask, device=torch.device("cpu"))
    actual = _mask_from_plan(plan, *pad_mask.shape)
    assert torch.equal(actual, _reference_mask(pad_mask, boundary_mask))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlashAttention")
def test_blockwise_flash_matches_eager_output_and_gradients():
    torch.manual_seed(7)
    device = torch.device("cuda")
    pad_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]],
        dtype=torch.bool,
        device=device,
    )
    boundary_mask = torch.tensor(
        [[1, 0, 1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 1, 1, 1]],
        dtype=torch.bool,
        device=device,
    )
    plan = build_packed_block_attention_plan(pad_mask, boundary_mask, device=device)
    scale = 1.0 / math.sqrt(16)

    tensors = [
        torch.randn(2, heads, 8, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
        for heads in (8, 1, 1)
    ]
    query, key, value = tensors
    repeated_key = key.repeat_interleave(8, dim=1)
    repeated_value = value.repeat_interleave(8, dim=1)
    logits = torch.matmul(query, repeated_key.transpose(-2, -1)) * scale
    mask = _reference_mask(pad_mask.cpu(), boundary_mask.cpu()).to(device)
    eager = torch.matmul(torch.softmax(logits.masked_fill(~mask[:, None], -torch.inf), dim=-1), repeated_value)
    eager_loss = eager.float().square().mean()
    eager_grads = torch.autograd.grad(eager_loss, tensors)

    flash = blockwise_varlen_flash_attention(query, key, value, plan, scale=scale)
    flash_loss = flash.float().square().mean()
    flash_grads = torch.autograd.grad(flash_loss, tensors)

    torch.testing.assert_close(flash, eager, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(flash_loss, eager_loss, atol=2e-4, rtol=2e-3)
    for flash_grad, eager_grad in zip(flash_grads, eager_grads, strict=True):
        torch.testing.assert_close(flash_grad, eager_grad, atol=2e-4, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlashAttention")
@pytest.mark.parametrize("autoregressive_tail", [False, True])
def test_blockwise_flash_matches_eager_at_training_sequence_length(autoregressive_tail):
    torch.manual_seed(17)
    device = torch.device("cuda")
    batch_size, length, head_dim = 2, 1536, 32
    pad_mask = torch.ones(batch_size, length, dtype=torch.bool, device=device)
    pad_mask[0, :37] = False
    pad_mask[1, -29:] = False
    boundary_mask = torch.zeros_like(pad_mask)
    for boundary in (37, 293, 549, 805, 1061, 1317):
        boundary_mask[0, boundary] = True
    for boundary in (0, 256, 512, 768, 1024, 1280):
        boundary_mask[1, boundary] = True
    if autoregressive_tail:
        boundary_mask[0, -96:] = True
        boundary_mask[1, -125:-29] = True

    plan = build_packed_block_attention_plan(pad_mask, boundary_mask, device=device)
    scale = 1.0 / math.sqrt(head_dim)
    tensors = [
        torch.randn(batch_size, heads, length, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        for heads in (8, 1, 1)
    ]
    query, key, value = tensors
    repeated_key = key.repeat_interleave(8, dim=1)
    repeated_value = value.repeat_interleave(8, dim=1)
    logits = torch.matmul(query, repeated_key.transpose(-2, -1)) * scale
    mask = _reference_mask(pad_mask.cpu(), boundary_mask.cpu()).to(device)
    eager = torch.matmul(torch.softmax(logits.masked_fill(~mask[:, None], -torch.inf), dim=-1), repeated_value)
    eager_loss = eager.float().square().mean()
    eager_grads = torch.autograd.grad(eager_loss, tensors)

    flash = blockwise_varlen_flash_attention(query, key, value, plan, scale=scale)
    flash_loss = flash.float().square().mean()
    flash_grads = torch.autograd.grad(flash_loss, tensors)

    torch.testing.assert_close(flash, eager, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(flash_loss, eager_loss, atol=2e-4, rtol=2e-3)
    for flash_grad, eager_grad in zip(flash_grads, eager_grads, strict=True):
        assert torch.isfinite(flash_grad).all()
        torch.testing.assert_close(flash_grad, eager_grad, atol=2e-4, rtol=3e-2)
