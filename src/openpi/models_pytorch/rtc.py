"""Inference-time guidance utilities for Real-Time Chunking (RTC).

RTC is training-free. It guides a flow policy toward the unconsumed suffix of
the previous action chunk while leaving a fresh tail unconstrained.
"""

from __future__ import annotations

import math

import torch


def soft_prefix_weights(
    horizon: int,
    inference_delay: int,
    execution_horizon: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the exponential RTC mask from Eq. 5 of Black et al. (2025).

    The first ``inference_delay`` actions are frozen, the overlap then decays
    smoothly, and the final ``execution_horizon`` actions are unconstrained.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if not 0 <= inference_delay <= horizon:
        raise ValueError(f"inference_delay must be in [0, {horizon}], got {inference_delay}")
    if not 0 <= execution_horizon <= horizon:
        raise ValueError(f"execution_horizon must be in [0, {horizon}], got {execution_horizon}")

    overlap_end = horizon - execution_horizon
    frozen_end = min(inference_delay, overlap_end)
    indices = torch.arange(horizon, device=device, dtype=dtype)
    denominator = overlap_end - frozen_end + 1
    linear = torch.clamp(
        (frozen_end - 1 - indices) / denominator + 1,
        min=0.0,
        max=1.0,
    )
    weights = linear * torch.expm1(linear) / (math.e - 1.0)
    return torch.where(indices < overlap_end, weights, torch.zeros_like(weights))


def guidance_scale(
    reverse_flow_time: torch.Tensor,
    max_guidance_weight: float,
) -> torch.Tensor:
    """Return Pi-GDM guidance strength in the repository's reverse-time convention.

    The model samples from ``time=1`` (noise) to ``time=0`` (actions), whereas
    the RTC paper uses tau=0 -> 1. Substituting tau=1-time into Eq. 2 gives the
    expression below.
    """
    if max_guidance_weight <= 0:
        raise ValueError(f"max_guidance_weight must be positive, got {max_guidance_weight}")
    time = reverse_flow_time.to(torch.float32)
    tau = 1.0 - time
    numerator = tau.square() + time.square()
    denominator = tau * time
    unclipped = numerator / denominator
    unclipped = torch.nan_to_num(
        unclipped,
        nan=max_guidance_weight,
        posinf=max_guidance_weight,
        neginf=max_guidance_weight,
    )
    return torch.clamp(unclipped, max=max_guidance_weight)


def guidance_vjp(
    clean_action_estimate: torch.Tensor,
    noisy_actions: torch.Tensor,
    previous_actions: torch.Tensor,
    time_weights: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the weighted vector-Jacobian product in RTC Eq. 2."""
    if clean_action_estimate.shape != noisy_actions.shape:
        raise ValueError("clean_action_estimate and noisy_actions must have equal shape")
    if previous_actions.shape != noisy_actions.shape:
        raise ValueError("previous_actions and noisy_actions must have equal shape")
    if action_mask.shape != noisy_actions.shape:
        raise ValueError("action_mask and noisy_actions must have equal shape")
    if time_weights.ndim != 1 or time_weights.shape[0] != noisy_actions.shape[-2]:
        raise ValueError(
            f"time_weights must be [action_horizon], got {tuple(time_weights.shape)} for {tuple(noisy_actions.shape)}"
        )

    weights = time_weights.view(1, -1, 1).to(
        device=noisy_actions.device,
        dtype=noisy_actions.dtype,
    )
    error = (previous_actions - clean_action_estimate) * weights
    error = torch.where(action_mask, error, torch.zeros_like(error))
    return torch.autograd.grad(
        outputs=clean_action_estimate,
        inputs=noisy_actions,
        grad_outputs=error,
        retain_graph=False,
        create_graph=False,
        only_inputs=True,
    )[0]
