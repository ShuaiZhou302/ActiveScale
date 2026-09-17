import numpy as np
import torch

from openpi.policies.policy import Policy
from openpi.shared.normalize import NormStats
import openpi.transforms as transforms


class _FakeRtcModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_rtc = None

    def sample_actions(self, device, observation, **kwargs):
        del device, observation, kwargs
        return torch.zeros(1, 3, 4)

    def sample_actions_rtc(self, device, observation, **kwargs):
        del device, observation
        self.last_rtc = kwargs
        return torch.ones(1, 3, 4)


def _observation():
    return {
        "image": {"base_0_rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
        "image_mask": {"base_0_rgb": np.asarray(True)},
        "state": np.zeros(4, dtype=np.float32),
    }


def test_policy_routes_rtc_request_and_previous_actions_to_rtc_sampler():
    model = _FakeRtcModel()
    policy = Policy(model, is_pytorch=True, pytorch_device="cpu")
    observation = _observation()
    observation["actions"] = np.full((3, 4), 2.0, dtype=np.float32)
    observation["rtc"] = {
        "inference_delay_steps": 1,
        "execution_horizon_steps": 2,
        "max_guidance_weight": 5.0,
        "num_steps": 5,
    }

    result = policy.infer(observation)

    assert np.array_equal(result["actions"], np.ones((3, 4), dtype=np.float32))
    assert model.last_rtc is not None
    assert model.last_rtc["previous_actions"].shape == (1, 3, 4)
    assert model.last_rtc["inference_delay_steps"] == 1
    assert model.last_rtc["execution_horizon_steps"] == 2
    assert model.last_rtc["num_steps"] == 5


def test_policy_normalizes_physical_rtc_suffix_then_pads_model_dimensions():
    model = _FakeRtcModel()
    physical_dim = 2
    model_dim = 4
    stats = {
        "state": NormStats(
            mean=np.zeros(physical_dim, dtype=np.float32),
            std=np.ones(physical_dim, dtype=np.float32),
        ),
        "actions": NormStats(
            mean=np.asarray([10.0, 20.0], dtype=np.float32),
            std=np.asarray([2.0, 4.0], dtype=np.float32),
        ),
    }
    policy = Policy(
        model,
        transforms=[
            transforms.Normalize(stats),
            transforms.PadStatesAndActions(model_dim),
        ],
        is_pytorch=True,
        pytorch_device="cpu",
    )
    observation = _observation()
    observation["state"] = np.zeros(physical_dim, dtype=np.float32)
    observation["actions"] = np.asarray(
        [[12.0, 24.0], [8.0, 16.0], [10.0, 20.0]],
        dtype=np.float32,
    )
    observation["rtc"] = {
        "inference_delay_steps": 1,
        "execution_horizon_steps": 2,
    }

    policy.infer(observation)

    expected = np.asarray(
        [[[1.0, 1.0, 0.0, 0.0], [-1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    assert model.last_rtc is not None
    assert np.allclose(model.last_rtc["previous_actions"].numpy(), expected, atol=1e-5)
