import math

import torch

from openpi.models_pytorch.rtc import guidance_scale
from openpi.models_pytorch.rtc import guidance_vjp
from openpi.models_pytorch.rtc import soft_prefix_weights


def test_soft_prefix_weights_freeze_decay_and_release_tail():
    weights = soft_prefix_weights(
        50,
        inference_delay=6,
        execution_horizon=25,
        device="cpu",
        dtype=torch.float32,
    )

    assert torch.equal(weights[:6], torch.ones(6))
    assert torch.all(weights[6:25] < 1)
    assert torch.all(weights[6:24] > weights[7:25])
    assert torch.equal(weights[25:], torch.zeros(25))


def test_soft_prefix_weights_matches_paper_equation_five():
    horizon, delay, execution_horizon = 10, 2, 4
    weights = soft_prefix_weights(
        horizon,
        delay,
        execution_horizon,
        device="cpu",
        dtype=torch.float64,
    )
    expected = []
    overlap_end = horizon - execution_horizon
    for index in range(horizon):
        if index < delay:
            expected.append(1.0)
        elif index < overlap_end:
            c_i = (overlap_end - index) / (overlap_end - delay + 1)
            expected.append(c_i * math.expm1(c_i) / math.expm1(1.0))
        else:
            expected.append(0.0)
    torch.testing.assert_close(weights, torch.tensor(expected, dtype=torch.float64))


def test_guidance_scale_handles_flow_endpoints_and_midpoint():
    times = torch.tensor([1.0, 0.5, 0.0])
    scales = guidance_scale(times, max_guidance_weight=5.0)

    torch.testing.assert_close(scales, torch.tensor([5.0, 2.0, 5.0]))


def test_guidance_vjp_identity_denoiser_matches_weighted_error():
    noisy = torch.zeros(1, 3, 2, requires_grad=True)
    clean = noisy
    previous = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    weights = torch.tensor([1.0, 0.5, 0.0])
    mask = torch.tensor([[[True, True], [True, False], [True, True]]])

    vjp = guidance_vjp(clean, noisy, previous, weights, mask)

    expected = torch.tensor([[[1.0, 2.0], [1.5, 0.0], [0.0, 0.0]]])
    torch.testing.assert_close(vjp, expected)
