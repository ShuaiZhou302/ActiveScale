from __future__ import annotations

from collections.abc import Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from openpi.training.human_vlm_dataset import HumanVLMPairedOfflineDataset
from openpi.training.human_vlm_dataset import ACTION_HORIZON
from openpi.training.human_vlm_dataset import MODEL_ACTION_DIM
from openpi.training.human_vlm_dataset import VIEW_LOW_LEVEL
from openpi.training.human_vlm_dataset import _normalize_q01_q99
from openpi.training.human_vlm_dataset import _pad_last_dim
from openpi.training.human_vlm_dataset import source_id_for
from openpi.training.piper_camera_token_dataset import PiperCameraTokenLeRobotDataset


PIPER_SOURCE_ID = source_id_for("Piper")
PIPER_ACTION_DIM = 23
PIPER_NORM_LAYOUT = "left_pose7_right_pose7_camera_pose7_gripper2"


class PiperJointObjectiveAdapter(torch.utils.data.Dataset):
    """Convert Piper camera-token samples to the human joint-objective schema."""

    def __init__(
        self,
        dataset: PiperCameraTokenLeRobotDataset,
        *,
        tokenizer,
        norm_stats_path: str | Path,
        action_loss_weight: float = 1.0,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.action_loss_weight = float(action_loss_weight)
        if not math.isfinite(self.action_loss_weight) or self.action_loss_weight < 0:
            raise ValueError("Piper action_loss_weight must be finite and non-negative")
        payload = json.loads(Path(norm_stats_path).read_text())
        if payload.get("action_layout") != PIPER_NORM_LAYOUT:
            raise ValueError(
                f"Piper norm layout must be {PIPER_NORM_LAYOUT!r}, got {payload.get('action_layout')!r}"
            )
        self.state_q01 = self._vector(payload, "state_q01")
        self.state_q99 = self._vector(payload, "state_q99")
        self.action_q01 = self._vector(payload, "action_q01")
        self.action_q99 = self._vector(payload, "action_q99")
        # FAST CE has no per-timestep target mask. Exclude episode-tail anchors
        # up front instead of teaching repeated padding as a real trajectory.
        self._valid_indices = [
            index for index in range(len(dataset)) if dataset.has_full_action_horizon(index)
        ]
        if not self._valid_indices:
            raise ValueError("Piper dataset has no anchors with a fully valid action horizon")

    @staticmethod
    def _vector(payload: dict[str, Any], key: str) -> np.ndarray:
        value = np.asarray(payload[key], dtype=np.float32)
        if value.shape != (PIPER_ACTION_DIM,) or not np.isfinite(value).all():
            raise ValueError(f"Piper {key} must be finite [{PIPER_ACTION_DIM}], got {value.shape}")
        return value

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = self._valid_indices[int(index)]
        sample = self.dataset[source_index]
        state23 = np.asarray(sample["state"], dtype=np.float32).reshape(PIPER_ACTION_DIM)
        actions23 = np.asarray(sample["actions"], dtype=np.float32).reshape(
            self.dataset.action_horizon, PIPER_ACTION_DIM
        )
        state = _pad_last_dim(
            _normalize_q01_q99(state23, self.state_q01, self.state_q99), MODEL_ACTION_DIM
        ).astype(np.float32)
        normalized_actions = _normalize_q01_q99(
            actions23, self.action_q01[None, :], self.action_q99[None, :]
        )
        normalized_actions = _pad_last_dim(normalized_actions, MODEL_ACTION_DIM).astype(np.float32)
        action_mask = _pad_last_dim(
            np.asarray(sample["action_dim_mask"], dtype=np.bool_), MODEL_ACTION_DIM, value=False
        ).astype(np.bool_)
        time_valid = np.asarray(sample["action_time_valid_mask"], dtype=np.bool_)
        # Human packs retain their physical H50 compatibility field even when
        # the active flow horizon is shorter. Match that collate schema while
        # keeping all padded Piper timesteps outside every loss.
        collate_actions = np.zeros((ACTION_HORIZON, MODEL_ACTION_DIM), dtype=np.float32)
        collate_action_mask = np.zeros((ACTION_HORIZON, MODEL_ACTION_DIM), dtype=np.bool_)
        collate_time_valid = np.zeros(ACTION_HORIZON, dtype=np.bool_)
        collate_actions[: len(normalized_actions)] = normalized_actions
        collate_action_mask[: len(action_mask)] = action_mask
        collate_time_valid[: len(time_valid)] = time_valid
        prompt = str(sample["prompt"])
        tokenized = self.tokenizer.tokenize_fast_action(prompt, state, normalized_actions[:, :PIPER_ACTION_DIM])
        condition = self.tokenizer.tokenize_action_condition(prompt, state)
        if tokenized.truncated:
            raise ValueError(f"Piper FAST target truncated at dataset index {source_index}")
        if condition.truncated:
            raise ValueError(f"Piper flow condition truncated at dataset index {source_index}")

        return {
            "image": {
                "left_wrist_0_rgb": sample["image"]["left_wrist_0_rgb"],
                "right_wrist_0_rgb": sample["image"]["right_wrist_0_rgb"],
            },
            "image_mask": {
                "left_wrist_0_rgb": sample["image_mask"]["left_wrist_0_rgb"],
                "right_wrist_0_rgb": sample["image_mask"]["right_wrist_0_rgb"],
            },
            "state": state,
            "state_pose_valid": np.ones(3, dtype=np.bool_),
            "actions": collate_actions,
            "action_dim_mask": collate_action_mask,
            "action_time_valid_mask": collate_time_valid,
            "tokenized_prompt": tokenized.tokens,
            "tokenized_prompt_mask": tokenized.token_mask,
            "token_ar_mask": tokenized.ar_mask,
            "token_loss_mask": tokenized.loss_mask,
            "vlm_view_type": np.asarray(VIEW_LOW_LEVEL, dtype=np.int32),
            "vlm_token_truncated": np.asarray(False, dtype=np.bool_),
            "source_id": np.asarray(PIPER_SOURCE_ID, dtype=np.int32),
            "action_loss_weight": np.asarray(self.action_loss_weight, dtype=np.float32),
            "flow_actions": normalized_actions,
            "flow_action_mask": action_mask,
            "flow_time_valid_mask": time_valid,
            "flow_condition_tokens": condition.tokens,
            "flow_condition_mask": condition.token_mask,
            "flow_condition_ar_mask": condition.ar_mask,
            "has_flow_target": np.asarray(True, dtype=np.bool_),
            "front_history_images": sample["front_history_images"],
            "front_history_masks": sample["front_history_masks"],
            "camera_extrinsics": sample["camera_extrinsics"],
            "camera_pose_valid": sample["camera_pose_valid"],
            "camera_fov": sample["camera_fov"],
            "camera_fov_valid": sample["camera_fov_valid"],
            "camera_image_hw": sample["camera_image_hw"],
        }


class HumanPiperCotrainDataset(torch.utils.data.Dataset):
    """Deterministic integer-ratio routing over model-ready human and Piper samples."""

    is_human_vlm_offline_pack = True

    def __init__(
        self,
        human: HumanVLMPairedOfflineDataset,
        piper_datasets: Sequence[PiperJointObjectiveAdapter],
        *,
        public_robot_datasets: Sequence[torch.utils.data.Dataset] = (),
        human_slots: int = 1,
        piper_slots: int = 1,
        epoch_size_multiple: int = 1,
        current_front_history_slot: int | None = None,
    ) -> None:
        robot_datasets = [*piper_datasets, *public_robot_datasets]
        if not robot_datasets:
            raise ValueError("At least one Robot dataset is required")
        if human_slots <= 0 or piper_slots <= 0:
            raise ValueError("human_slots and piper_slots must be positive")
        self.human = human
        self.piper = torch.utils.data.ConcatDataset(robot_datasets)
        self.current_front_history_slot = current_front_history_slot
        self.schedule = ("human",) * int(human_slots) + ("piper",) * int(piper_slots)
        # One finite pass exposes every sample from the larger domain at least
        # once; the smaller domain repeats deterministically to realize ratio.
        cycles = max(math.ceil(len(human) / human_slots), math.ceil(len(self.piper) / piper_slots))
        if epoch_size_multiple <= 0:
            raise ValueError("epoch_size_multiple must be positive")
        cycle_multiple = int(epoch_size_multiple) // math.gcd(int(epoch_size_multiple), len(self.schedule))
        cycles = math.ceil(cycles / cycle_multiple) * cycle_multiple
        self.num_frames = cycles * len(self.schedule)

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index) % self.num_frames
        cycle, slot = divmod(index, len(self.schedule))
        domain = self.schedule[slot]
        domain_slot = sum(1 for item in self.schedule[: slot + 1] if item == domain) - 1
        if domain == "human":
            domain_index = cycle * self.schedule.count("human") + domain_slot
            sample = self.human[domain_index % len(self.human)]
        else:
            domain_index = cycle * self.schedule.count("piper") + domain_slot
            sample = self.piper[domain_index % len(self.piper)]
        if self.current_front_history_slot is None:
            return sample

        history = np.asarray(sample["front_history_images"])
        current_front = history[self.current_front_history_slot]
        result = dict(sample)
        result["image"] = {**sample["image"], "base_0_rgb": current_front}
        result["image_mask"] = {**sample["image_mask"], "base_0_rgb": np.asarray(True)}
        return result
