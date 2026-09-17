import json

import numpy as np
import pytest
import torch

from openpi.models.tokenizer import TokenizedVLMView
from openpi.training.human_piper_cotrain_dataset import HumanPiperCotrainDataset
from openpi.training.human_piper_cotrain_dataset import PIPER_NORM_LAYOUT
from openpi.training.human_piper_cotrain_dataset import PIPER_SOURCE_ID
from openpi.training.human_piper_cotrain_dataset import PiperJointObjectiveAdapter


class _Tokenizer:
    def tokenize_fast_action(self, prompt, state, actions):
        assert prompt == "pick object"
        assert state.shape == (32,)
        assert actions.shape == (3, 23)
        return TokenizedVLMView(
            tokens=np.array([1, 2, 0], dtype=np.int32),
            token_mask=np.array([True, True, False]),
            ar_mask=np.array([0, 1, 0], dtype=np.int32),
            loss_mask=np.array([False, True, False]),
            truncated=False,
        )

    def tokenize_action_condition(self, prompt, state):
        del prompt, state
        return TokenizedVLMView(
            tokens=np.array([1, 0, 0], dtype=np.int32),
            token_mask=np.array([True, False, False]),
            ar_mask=np.zeros(3, dtype=np.int32),
            loss_mask=np.zeros(3, dtype=np.bool_),
            truncated=False,
        )


class _Piper(torch.utils.data.Dataset):
    action_horizon = 3

    def __len__(self):
        return 3

    def has_full_action_horizon(self, index):
        return index != 1

    def __getitem__(self, index):
        del index
        return {
            "image": {
                "base_0_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                "left_wrist_0_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                "right_wrist_0_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
            },
            "image_mask": {
                "base_0_rgb": np.asarray(True),
                "left_wrist_0_rgb": np.asarray(True),
                "right_wrist_0_rgb": np.asarray(True),
            },
            "state": np.full(23, 5.0, dtype=np.float32),
            "actions": np.full((3, 23), 5.0, dtype=np.float32),
            "action_dim_mask": np.ones((3, 23), dtype=np.bool_),
            "action_time_valid_mask": np.ones(3, dtype=np.bool_),
            "prompt": "pick object",
            "front_history_images": np.zeros((4, 4, 4, 3), dtype=np.uint8),
            "front_history_masks": np.ones(4, dtype=np.bool_),
            "camera_extrinsics": np.broadcast_to(np.eye(4), (4, 4, 4)).copy(),
            "camera_pose_valid": np.ones(4, dtype=np.bool_),
            "camera_fov": np.ones((4, 2), dtype=np.float32),
            "camera_fov_valid": np.ones(4, dtype=np.bool_),
            "camera_image_hw": np.ones((4, 2), dtype=np.float32),
        }


class _Human(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {"domain": "human", "index": index}


class _HumanFive(_Human):
    def __len__(self):
        return 5


class _VisionSampleDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        del index
        history = np.stack([np.full((4, 4, 3), value, dtype=np.uint8) for value in range(4)])
        return {
            "image": {
                "left_wrist_0_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                "right_wrist_0_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
            },
            "image_mask": {
                "left_wrist_0_rgb": np.asarray(True),
                "right_wrist_0_rgb": np.asarray(True),
            },
            "front_history_images": history,
        }


def _write_norm(path):
    path.write_text(
        json.dumps(
            {
                "action_layout": PIPER_NORM_LAYOUT,
                "state_q01": [0.0] * 23,
                "state_q99": [10.0] * 23,
                "action_q01": [0.0] * 23,
                "action_q99": [10.0] * 23,
            }
        )
    )
    return path


def test_piper_joint_adapter_filters_tail_and_emits_weighted_joint_schema(tmp_path):
    adapter = PiperJointObjectiveAdapter(
        _Piper(), tokenizer=_Tokenizer(), norm_stats_path=_write_norm(tmp_path / "norm.json")
    )
    assert len(adapter) == 2
    item = adapter[0]
    assert item["source_id"].item() == PIPER_SOURCE_ID
    assert item["action_loss_weight"].item() == pytest.approx(1.0)
    assert item["flow_actions"].shape == (3, 32)
    assert item["actions"].shape == (50, 32)
    assert item["action_dim_mask"][:3, :23].all()
    assert not item["action_dim_mask"][3:].any()
    assert np.allclose(item["state"], 0.0)
    assert item["flow_action_mask"][:, :23].all()
    assert not item["flow_action_mask"][:, 23:].any()
    assert item["vlm_view_type"].item() == 1


def test_human_piper_router_is_deterministic_one_to_one(tmp_path):
    piper = PiperJointObjectiveAdapter(
        _Piper(), tokenizer=_Tokenizer(), norm_stats_path=_write_norm(tmp_path / "norm.json")
    )
    dataset = HumanPiperCotrainDataset(_Human(), [piper], human_slots=1, piper_slots=1)
    assert len(dataset) == 8
    assert dataset[0]["domain"] == "human"
    assert dataset[2]["index"] == 1
    assert dataset[1]["source_id"].item() == PIPER_SOURCE_ID
    assert dataset[3]["source_id"].item() == PIPER_SOURCE_ID


def test_cotrain_epoch_can_be_padded_to_global_batch_without_dropping_domain_tail(tmp_path):
    piper = PiperJointObjectiveAdapter(
        _Piper(), tokenizer=_Tokenizer(), norm_stats_path=_write_norm(tmp_path / "norm.json")
    )
    dataset = HumanPiperCotrainDataset(
        _HumanFive(), [piper], human_slots=1, piper_slots=1, epoch_size_multiple=8
    )
    assert len(dataset) == 16
    assert len(dataset) % 8 == 0


def test_original_pi05_adapter_maps_current_history_frame_to_base_image():
    source = _VisionSampleDataset()
    dataset = HumanPiperCotrainDataset(
        source,
        [source],
        human_slots=1,
        piper_slots=1,
        current_front_history_slot=3,
    )

    for index in (0, 1):
        sample = dataset[index]
        assert np.all(sample["image"]["base_0_rgb"] == 3)
        assert sample["image_mask"]["base_0_rgb"].item()
        assert "base_0_rgb" not in source[0]["image"]
