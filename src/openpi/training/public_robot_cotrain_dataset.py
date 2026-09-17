from __future__ import annotations

import bisect
from collections.abc import Sequence
import functools
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch

from openpi.training.human_vlm_dataset import ACTION_HORIZON
from openpi.training.human_vlm_dataset import IMAGE_SIZE
from openpi.training.human_vlm_dataset import MODEL_ACTION_DIM
from openpi.training.human_vlm_dataset import VIEW_LOW_LEVEL
from openpi.training.human_vlm_dataset import _normalize_q01_q99
from openpi.training.human_vlm_dataset import _pad_last_dim
from openpi.training.human_vlm_dataset import source_id_for


CANONICAL_ROBOT_DIM = 16
MODEL_ROBOT_DIM = 23
ROBOT_NORM_LAYOUT = "left_pose7_right_pose7_gripper2"
INACTIVE_SUBTASKS = {"", "null", "none", "end", "abnormal"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _normalized_label(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.strip().lower())


def _is_active_label(value: str) -> bool:
    return _normalized_label(value) not in INACTIVE_SUBTASKS


def _euler_xyz_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """Convert intrinsic xyz Euler radians to scalar-last xyzw quaternion."""
    euler = np.asarray(euler, dtype=np.float64)
    x, y, z = np.moveaxis(euler, -1, 0)
    cx, sx = np.cos(x / 2), np.sin(x / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    cz, sz = np.cos(z / 2), np.sin(z / 2)
    return np.stack(
        (
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        ),
        axis=-1,
    ).astype(np.float32)


def _canonical_beta(row: dict[str, Any], prefix: str) -> np.ndarray:
    position = np.asarray(row[f"{prefix}.end.position"], dtype=np.float32).reshape(2, 3)
    orientation = np.asarray(row[f"{prefix}.end.orientation"], dtype=np.float32).reshape(2, 4)
    gripper = np.asarray(row[f"{prefix}.effector.position"], dtype=np.float32).reshape(2)
    return np.concatenate((position[0], orientation[0], position[1], orientation[1], gripper))


def _canonical_robocoin(row: dict[str, Any], *, action: bool) -> np.ndarray:
    suffix = "action" if action else "state"
    eef = np.asarray(row[f"eef_sim_pose_{suffix}"], dtype=np.float32).reshape(2, 6)
    gripper = np.asarray(row[f"gripper_open_scale_{suffix}"], dtype=np.float32).reshape(2)
    left = np.concatenate((eef[0, :3], _euler_xyz_to_quaternion(eef[0, 3:])))
    right = np.concatenate((eef[1, :3], _euler_xyz_to_quaternion(eef[1, 3:])))
    return np.concatenate((left, right, gripper)).astype(np.float32)


def _field_indices(info: dict[str, Any], feature: str, field: str) -> np.ndarray:
    descriptions = info["features"][feature]["field_descriptions"]
    return np.asarray(descriptions[field]["indices"], dtype=np.int64)


def _canonical_agibotworld2026(
    row: dict[str, Any], info: dict[str, Any], *, action: bool
) -> np.ndarray:
    feature = "action" if action else "observation.state"
    values = np.asarray(row[feature], dtype=np.float32)
    if action:
        position_field = "action/end/position"
        orientation_field = "action/end/orientation"
        gripper_prefix = "action"
    else:
        position_field = "state/end/arm_position"
        orientation_field = "state/end/arm_orientation"
        gripper_prefix = "state"
    position = values[_field_indices(info, feature, position_field)]
    orientation = values[_field_indices(info, feature, orientation_field)]
    left_gripper = values[_field_indices(info, feature, f"{gripper_prefix}/left_effector/position")]
    right_gripper = values[
        _field_indices(info, feature, f"{gripper_prefix}/right_effector/position")
    ]
    return np.concatenate(
        (
            position[:3],
            orientation[:4],
            position[3:6],
            orientation[4:8],
            left_gripper,
            right_gripper,
        )
    ).astype(np.float32)


def _unwrap_quaternions(actions: np.ndarray, state: np.ndarray, valid: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32).copy()
    for start in (3, 10):
        reference = state[start : start + 4]
        for index in range(len(actions)):
            if not valid[index]:
                continue
            if float(np.dot(actions[index, start : start + 4], reference)) < 0:
                actions[index, start : start + 4] *= -1
            reference = actions[index, start : start + 4]
    return actions


def _to_model23(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    out = np.zeros((*value.shape[:-1], MODEL_ROBOT_DIM), dtype=value.dtype)
    out[..., :14] = value[..., :14]
    out[..., 21:23] = value[..., 14:16]
    return out


def robot_cache_frame_path(
    cache_root: str | Path,
    source: str,
    dataset: str,
    camera: str,
    episode_index: int,
    frame_index: int,
) -> Path:
    safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", source)
    safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset)
    safe_camera = re.sub(r"[^A-Za-z0-9_.-]+", "_", camera)
    return (
        Path(cache_root)
        / safe_source
        / safe_dataset
        / safe_camera
        / f"episode_{episode_index:06d}"
        / f"frame_{frame_index:06d}.jpg"
    )


class PublicRobotLeRobotDataset(torch.utils.data.Dataset):
    """Read selected AgiBot or RoboCOIN episodes at the configured stride.

    Poses stay in each source's declared body/base frame. Public sources have
    no admitted camera-pose target, so only image availability is returned.
    """

    def __init__(
        self,
        *,
        source: str,
        manifest_path: str | Path,
        action_horizon: int,
        frame_cache_dir: str | Path | None,
        root: str | Path | None = None,
        anchor_stride: int = 6,
        history_offsets: Sequence[int] = (-48, -32, -16, 0),
        fake_images: bool = False,
    ) -> None:
        if source not in {"AgiBotWorld-Beta", "AgiBotWorld2026", "RoboCOIN"}:
            raise ValueError(f"Unsupported public Robot source: {source}")
        self.source = source
        self.action_horizon = int(action_horizon)
        self.anchor_stride = int(anchor_stride)
        self.history_offsets = tuple(int(value) for value in history_offsets)
        self.frame_cache_dir = None if frame_cache_dir is None else Path(frame_cache_dir)
        self.fake_images = bool(fake_images)
        if self.anchor_stride <= 0 or self.action_horizon <= 0:
            raise ValueError("anchor_stride and action_horizon must be positive")
        if not self.fake_images and self.frame_cache_dir is None:
            raise ValueError("Public Robot training requires frame_cache_dir unless fake_images=True")
        self._root = None if root is None else Path(root)
        rows = _read_jsonl(Path(manifest_path))
        if not rows:
            raise ValueError(f"Empty Robot manifest: {manifest_path}")
        self._dataset_meta: dict[str, dict[str, Any]] = {}
        self._episodes = [self._normalize_episode(row) for row in rows]
        self._cum_anchors = np.cumsum(
            [math.ceil(int(episode["length"]) / self.anchor_stride) for episode in self._episodes],
            dtype=np.int64,
        ).tolist()

    def _normalize_episode(self, row: dict[str, Any]) -> dict[str, Any]:
        if self.source == "AgiBotWorld-Beta":
            return {
                **row,
                "dataset": str(row["task_dataset"]),
                "length": int(row["length"]),
                "episode_index": int(row["episode_index"]),
                "embodiment": str(row["embodiment"]),
                "task": str(row["task"]),
            }
        if self.source == "AgiBotWorld2026":
            dataset = str(row["dataset"])
            if dataset not in self._dataset_meta:
                self._dataset_meta[dataset] = {
                    "info": json.loads(Path(row["info"]).read_text())
                }
            return {
                **row,
                "dataset": dataset,
                "length": int(row["length"]),
                "episode_index": int(row["episode_index"]),
                "embodiment": str(row.get("embodiment", "g2a")),
                "task": str(row["task"]),
            }
        if self._root is None:
            raise ValueError("RoboCOIN requires root")
        dataset = str(row["dataset"])
        dataset_root = self._root / dataset
        meta = self._dataset_meta.get(dataset)
        if meta is None:
            info = json.loads((dataset_root / "meta/info.json").read_text())
            vocab = {
                int(item["subtask_index"]): str(item["subtask"])
                for item in _read_jsonl(dataset_root / "annotations/subtask_annotations.jsonl")
            }
            meta = {"info": info, "vocab": vocab}
            self._dataset_meta[dataset] = meta
        info = meta["info"]
        episode_index = int(row["episode_index"])
        chunk = episode_index // int(info["chunks_size"])
        videos = {
            camera: str(
                dataset_root
                / str(info["video_path"]).format(
                    episode_chunk=chunk,
                    episode_index=episode_index,
                    video_key=camera,
                )
            )
            for camera in (
                "observation.images.cam_head_rgb",
                "observation.images.cam_left_wrist_rgb",
                "observation.images.cam_right_wrist_rgb",
            )
        }
        return {
            **row,
            "dataset": dataset,
            "length": int(row["frames"]),
            "episode_index": episode_index,
            "embodiment": str(row["robot_type"]),
            "task": str(row["tasks"][0]),
            "parquet": str(
                dataset_root
                / str(info["data_path"]).format(episode_chunk=chunk, episode_index=episode_index)
            ),
            "videos": videos,
        }

    def __len__(self) -> int:
        return int(self._cum_anchors[-1])

    def representative_indices_by_embodiment(self) -> dict[str, int]:
        """Return one deterministic, non-tail anchor index per embodiment."""
        result: dict[str, int] = {}
        previous = 0
        for episode, cumulative in zip(self._episodes, self._cum_anchors, strict=True):
            embodiment = str(episode["embodiment"])
            if embodiment not in result:
                episode_anchors = int(cumulative) - previous
                result[embodiment] = previous + episode_anchors // 2
            previous = int(cumulative)
        return result

    def _locate(self, index: int) -> tuple[int, int]:
        episode_pos = bisect.bisect_right(self._cum_anchors, int(index))
        previous = self._cum_anchors[episode_pos - 1] if episode_pos else 0
        frame_index = (int(index) - previous) * self.anchor_stride
        return episode_pos, frame_index

    @functools.lru_cache(maxsize=8)
    def _load_rows(self, episode_pos: int) -> list[dict[str, Any]]:
        episode = self._episodes[episode_pos]
        if self.source == "AgiBotWorld-Beta":
            columns = [
                "observation.states.end.position",
                "observation.states.end.orientation",
                "observation.states.effector.position",
                "actions.end.position",
                "actions.end.orientation",
                "actions.effector.position",
            ]
        elif self.source == "AgiBotWorld2026":
            columns = ["observation.state", "action"]
        else:
            columns = [
                "eef_sim_pose_state",
                "eef_sim_pose_action",
                "gripper_open_scale_state",
                "gripper_open_scale_action",
                "subtask_annotation",
            ]
        return pq.read_table(episode["parquet"], columns=columns).to_pylist()

    def _prompt_at(self, episode_pos: int, frame_index: int, row: dict[str, Any]) -> str:
        episode = self._episodes[episode_pos]
        if self.source == "AgiBotWorld-Beta":
            for segment in episode.get("action_config", ()):
                if int(segment["start_frame"]) <= frame_index < int(segment["end_frame"]):
                    return str(segment["action_text"]).strip()
            return str(episode["task"]).strip()
        if self.source == "AgiBotWorld2026":
            return str(episode["task"]).strip()
        labels = row.get("subtask_annotation") or []
        vocab = self._dataset_meta[str(episode["dataset"])]["vocab"]
        active: list[str] = []
        for label_index in labels:
            label = str(vocab.get(int(label_index), "")).strip()
            if _is_active_label(label) and (not active or _normalized_label(label) != _normalized_label(active[-1])):
                active.append(label)
        # The final non-null slot is the most specific annotation when both a
        # generic arm label and an operation label are present.
        return active[-1] if active else str(episode["task"]).strip()

    def _canonical(self, episode_pos: int, row: dict[str, Any], *, action: bool) -> np.ndarray:
        if self.source == "AgiBotWorld-Beta":
            return _canonical_beta(row, "actions" if action else "observation.states")
        if self.source == "AgiBotWorld2026":
            episode = self._episodes[episode_pos]
            info = self._dataset_meta[str(episode["dataset"])]["info"]
            return _canonical_agibotworld2026(row, info, action=action)
        return _canonical_robocoin(row, action=action)

    def _read_frame(self, episode: dict[str, Any], camera: str, frame_index: int) -> np.ndarray:
        if self.fake_images:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        path = robot_cache_frame_path(
            self.frame_cache_dir,
            self.source,
            str(episode["dataset"]),
            camera,
            int(episode["episode_index"]),
            frame_index,
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing public Robot cached frame: {path}")
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.uint8)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_pos, frame_index = self._locate(index)
        episode = self._episodes[episode_pos]
        rows = self._load_rows(episode_pos)
        row = rows[frame_index]
        prompt = self._prompt_at(episode_pos, frame_index, row)
        state = self._canonical(episode_pos, row, action=False)

        future = []
        valid = []
        last = self._canonical(episode_pos, row, action=True)
        for offset in range(self.action_horizon):
            future_index = frame_index + offset
            is_valid = future_index < len(rows)
            if is_valid:
                future_row = rows[future_index]
                is_valid = self._prompt_at(episode_pos, future_index, future_row) == prompt
            if is_valid:
                last = self._canonical(episode_pos, future_row, action=True)
            future.append(last)
            valid.append(is_valid)
            if not is_valid:
                valid.extend([False] * (self.action_horizon - len(valid)))
                future.extend([last] * (self.action_horizon - len(future)))
                break
        valid_array = np.asarray(valid, dtype=np.bool_)
        actions = _unwrap_quaternions(np.stack(future), state, valid_array)

        if self.source == "AgiBotWorld-Beta":
            cameras = {
                "front": "observation.images.head",
                "left": "observation.images.hand_left",
                "right": "observation.images.hand_right",
            }
        elif self.source == "AgiBotWorld2026":
            cameras = {
                "front": "observation.images.top_head",
                "left": "observation.images.hand_left",
                "right": "observation.images.hand_right",
            }
        else:
            cameras = {
                "front": "observation.images.cam_head_rgb",
                "left": "observation.images.cam_left_wrist_rgb",
                "right": "observation.images.cam_right_wrist_rgb",
            }
        current_front = self._read_frame(episode, cameras["front"], frame_index)
        history_images = []
        history_valid = []
        for offset in self.history_offsets:
            history_index = frame_index + offset
            available = 0 <= history_index < len(rows)
            history_valid.append(available)
            history_images.append(
                self._read_frame(episode, cameras["front"], history_index)
                if available
                else np.zeros_like(current_front)
            )
        image_hw = np.asarray([current_front.shape[0], current_front.shape[1]], dtype=np.float32)
        return {
            "source": self.source,
            "embodiment": str(episode["embodiment"]),
            "prompt": prompt,
            "state": state.astype(np.float32),
            "actions": actions.astype(np.float32),
            "action_time_valid_mask": valid_array,
            "image": {
                "left_wrist_0_rgb": self._read_frame(episode, cameras["left"], frame_index),
                "right_wrist_0_rgb": self._read_frame(episode, cameras["right"], frame_index),
            },
            "front_history_images": np.stack(history_images),
            "front_history_masks": np.asarray(history_valid, dtype=np.bool_),
            "camera_image_hw": np.broadcast_to(image_hw, (len(self.history_offsets), 2)).copy(),
        }


class PublicRobotJointObjectiveAdapter(torch.utils.data.Dataset):
    """Map public bimanual EEF datasets into the APVLA joint objective."""

    def __init__(
        self,
        dataset: PublicRobotLeRobotDataset,
        *,
        tokenizer,
        norm_stats_dir: str | Path,
        action_loss_weight: float = 1.0,
        allow_partial_fast_horizon: bool = True,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.action_loss_weight = float(action_loss_weight)
        self.allow_partial_fast_horizon = bool(allow_partial_fast_horizon)
        self.norm_stats_dir = Path(norm_stats_dir)
        self._norms: dict[tuple[str, str], tuple[np.ndarray, ...]] = {}
        if not math.isfinite(self.action_loss_weight) or self.action_loss_weight < 0:
            raise ValueError("Robot action_loss_weight must be finite and non-negative")

    def __len__(self) -> int:
        return len(self.dataset)

    def _norm(self, source: str, embodiment: str) -> tuple[np.ndarray, ...]:
        key = (source, embodiment)
        cached = self._norms.get(key)
        if cached is not None:
            return cached
        safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", source)
        safe_embodiment = re.sub(r"[^A-Za-z0-9_.-]+", "_", embodiment)
        path = self.norm_stats_dir / f"{safe_source}__{safe_embodiment}.json"
        payload = json.loads(path.read_text())
        if payload.get("action_layout") != ROBOT_NORM_LAYOUT:
            raise ValueError(f"Unexpected Robot norm layout in {path}: {payload.get('action_layout')}")
        arrays = tuple(np.asarray(payload[name], dtype=np.float32) for name in (
            "state_q01", "state_q99", "action_q01", "action_q99"
        ))
        if any(array.shape != (CANONICAL_ROBOT_DIM,) or not np.isfinite(array).all() for array in arrays):
            raise ValueError(f"Invalid Robot norm vectors: {path}")
        self._norms[key] = arrays
        return arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        source = str(sample["source"])
        embodiment = str(sample["embodiment"])
        state_q01, state_q99, action_q01, action_q99 = self._norm(source, embodiment)
        state16 = np.asarray(sample["state"], dtype=np.float32)
        actions16 = np.asarray(sample["actions"], dtype=np.float32)
        valid = np.asarray(sample["action_time_valid_mask"], dtype=np.bool_)
        valid_prefix = int(np.flatnonzero(~valid)[0]) if (~valid).any() else len(valid)
        if valid_prefix <= 0:
            raise ValueError(f"Robot sample has no valid action target at index {index}")
        if not self.allow_partial_fast_horizon and valid_prefix != len(valid):
            raise ValueError(f"Robot sample has a partial action target at index {index}")
        normalized_state = _normalize_q01_q99(state16, state_q01, state_q99)
        normalized_actions = _normalize_q01_q99(actions16, action_q01[None], action_q99[None])
        state23 = _to_model23(normalized_state)
        actions23 = _to_model23(normalized_actions)
        state = _pad_last_dim(state23, MODEL_ACTION_DIM).astype(np.float32)
        actions = _pad_last_dim(actions23, MODEL_ACTION_DIM).astype(np.float32)
        dim_mask23 = np.zeros((len(valid), MODEL_ROBOT_DIM), dtype=np.bool_)
        dim_mask23[:, :14] = valid[:, None]
        dim_mask23[:, 21:23] = valid[:, None]
        action_mask = _pad_last_dim(dim_mask23, MODEL_ACTION_DIM, value=False)
        prompt = str(sample["prompt"])
        tokenized = self.tokenizer.tokenize_fast_action(prompt, state, actions23[:valid_prefix])
        condition = self.tokenizer.tokenize_action_condition(prompt, state)
        if tokenized.truncated or condition.truncated:
            raise ValueError(f"Robot token sequence truncated at index {index}")
        history_mask = np.asarray(sample["front_history_masks"], dtype=np.bool_)
        history_slots = len(history_mask)
        return {
            "image": sample["image"],
            "image_mask": {
                "left_wrist_0_rgb": np.asarray(True),
                "right_wrist_0_rgb": np.asarray(True),
            },
            "state": state,
            "state_pose_valid": np.asarray([True, True, False], dtype=np.bool_),
            "actions": actions,
            "action_dim_mask": action_mask,
            "action_time_valid_mask": valid,
            "tokenized_prompt": tokenized.tokens,
            "tokenized_prompt_mask": tokenized.token_mask,
            "token_ar_mask": tokenized.ar_mask,
            "token_loss_mask": tokenized.loss_mask,
            "vlm_view_type": np.asarray(VIEW_LOW_LEVEL, dtype=np.int32),
            "vlm_token_truncated": np.asarray(False, dtype=np.bool_),
            "source_id": np.asarray(source_id_for(source), dtype=np.int32),
            "action_loss_weight": np.asarray(self.action_loss_weight, dtype=np.float32),
            "flow_actions": actions,
            "flow_action_mask": action_mask,
            "flow_time_valid_mask": valid,
            "flow_condition_tokens": condition.tokens,
            "flow_condition_mask": condition.token_mask,
            "flow_condition_ar_mask": condition.ar_mask,
            "has_flow_target": np.asarray(True, dtype=np.bool_),
            "front_history_images": sample["front_history_images"],
            "front_history_masks": history_mask,
            "camera_extrinsics": np.broadcast_to(np.eye(4, dtype=np.float32), (history_slots, 4, 4)).copy(),
            "camera_pose_valid": np.zeros(history_slots, dtype=np.bool_),
            "camera_fov": np.zeros((history_slots, 2), dtype=np.float32),
            "camera_fov_valid": np.zeros(history_slots, dtype=np.bool_),
            "camera_image_hw": sample["camera_image_hw"],
        }
