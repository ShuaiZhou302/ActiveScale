from __future__ import annotations

import bisect
from collections.abc import Sequence
import functools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch


PIPER_REPO_ID = "cobot-avp-teleop_lerobot"
IMAGE_SIZE = 224
HISTORY_OFFSETS = (-48, -32, -16, 0)
POSE7 = 7
ROBOT_ACTION_DIM = 23
STATE_GRIPPER_INDICES = (6, 13)
PARQUET_FILE_CACHE_SIZE = int(os.environ.get("OPENPI_PIPER_PARQUET_CACHE_SIZE", "64"))

_CAMERA_CACHE_KEYS = {
    "cam_front": "observation.images.cam_front",
    "cam_left": "observation.images.cam_left",
    "cam_right": "observation.images.cam_right",
}

_COLUMNS = (
    "timestamp",
    "frame_index",
    "episode_index",
    "task_index",
    "observation.state",
    "observation.camera_pose_in_unified.quat.cam_front",
    "observation.camera_pose_in_unified.matrix.cam_front",
    "observation.ee_pose_in_unified.quat.left",
    "observation.ee_pose_in_unified.quat.right",
    "action.hybrid.left_ee_pose_in_unified.quat",
    "action.hybrid.right_ee_pose_in_unified.quat",
    "action.hybrid.mid_camera_pose_in_unified.quat",
    "action.hybrid.gripper",
)


def _as_float32(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _unit_normalize_pose_quats(array: np.ndarray) -> np.ndarray:
    out = np.asarray(array, dtype=np.float32).copy()
    for start in (0, 7, 14):
        q = out[..., start + 3 : start + 7]
        norm = np.linalg.norm(q, axis=-1, keepdims=True)
        out[..., start + 3 : start + 7] = np.divide(q, norm, out=np.zeros_like(q), where=norm > 1e-6)
    return out


def _unwrap_action_quaternion_signs(actions: np.ndarray, state: np.ndarray, valid_time: np.ndarray) -> np.ndarray:
    actions = _unit_normalize_pose_quats(actions)
    state = _unit_normalize_pose_quats(state)
    for start in (0, 7, 14):
        lo, hi = start + 3, start + 7
        reference = state[lo:hi]
        for t in range(actions.shape[0]):
            if not bool(valid_time[t]):
                continue
            if float(np.dot(actions[t, lo:hi], reference)) < 0.0:
                actions[t, lo:hi] = -actions[t, lo:hi]
            reference = actions[t, lo:hi]
    return actions


def _state23(row: dict[str, Any], *, gripper_indices: Sequence[int] = STATE_GRIPPER_INDICES) -> np.ndarray:
    joint_state = _as_float32(row["observation.state"]).reshape(-1)
    grippers = joint_state[list(gripper_indices)]
    state = np.concatenate(
        [
            _as_float32(row["observation.ee_pose_in_unified.quat.left"]).reshape(POSE7),
            _as_float32(row["observation.ee_pose_in_unified.quat.right"]).reshape(POSE7),
            _as_float32(row["observation.camera_pose_in_unified.quat.cam_front"]).reshape(POSE7),
            grippers.astype(np.float32),
        ],
        axis=-1,
    )
    return _unit_normalize_pose_quats(state)


def _action23(row: dict[str, Any]) -> np.ndarray:
    action = np.concatenate(
        [
            _as_float32(row["action.hybrid.left_ee_pose_in_unified.quat"]).reshape(POSE7),
            _as_float32(row["action.hybrid.right_ee_pose_in_unified.quat"]).reshape(POSE7),
            _as_float32(row["action.hybrid.mid_camera_pose_in_unified.quat"]).reshape(POSE7),
            _as_float32(row["action.hybrid.gripper"]).reshape(2),
        ],
        axis=-1,
    )
    return _unit_normalize_pose_quats(action)


class PiperCameraTokenLeRobotDataset(torch.utils.data.Dataset):
    """Reader for Piper LeRobot-v3 robot post-training with APVLA camera tokens.

    Contract:
    - state/action layout is [left pose7, right pose7, front camera pose7, left/right gripper].
    - all poses are absolute in the robot unified frame, with no human-style
      rebase to the earliest history frame.
    - camera_extrinsics stores T_unified_from_front_camera for each history slot.
    - front history outside the episode start is zero-imaged and mask-invalid.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        action_horizon: int,
        anchor_stride: int = 1,
        task_indices: Sequence[int] | None = None,
        frame_cache_dir: str | Path | None = None,
        history_offsets: Sequence[int] = HISTORY_OFFSETS,
        state_gripper_indices: Sequence[int] = STATE_GRIPPER_INDICES,
        included_episodes: Sequence[int] | None = None,
        excluded_anchors: Sequence[tuple[int, int]] = (),
        strict_cache: bool | None = None,
    ) -> None:
        self.root = Path(root)
        self.action_horizon = int(action_horizon)
        self.anchor_stride = int(anchor_stride)
        if self.anchor_stride <= 0:
            raise ValueError("anchor_stride must be positive")
        self.task_indices = None if task_indices is None else frozenset(int(idx) for idx in task_indices)
        self.history_offsets = tuple(int(offset) for offset in history_offsets)
        self.state_gripper_indices = tuple(int(idx) for idx in state_gripper_indices)
        self.included_episodes = (
            None if included_episodes is None else frozenset(int(episode) for episode in included_episodes)
        )
        if self.included_episodes is not None and not self.included_episodes:
            raise ValueError("included_episodes was provided but is empty")
        self.excluded_anchors = frozenset((int(episode), int(frame)) for episode, frame in excluded_anchors)
        self.frame_cache_dir = Path(frame_cache_dir) if frame_cache_dir is not None else None
        if self.frame_cache_dir is None and (env_cache := os.environ.get("OPENPI_PIPER_FRAME_CACHE_DIR")):
            self.frame_cache_dir = Path(env_cache)
        self.strict_cache = (
            os.environ.get("OPENPI_PIPER_FRAME_CACHE_STRICT", "1") != "0" if strict_cache is None else strict_cache
        )

        info = json.loads((self.root / "meta/info.json").read_text())
        self.fps = int(info["fps"])
        self._tasks = self._load_tasks()
        self._camera_fov_yx, self._camera_hw = self._load_front_camera_fov()

        self._data_files = sorted((self.root / "data").glob("chunk-*/file-*.parquet"))
        if not self._data_files:
            raise FileNotFoundError(f"No parquet data files found under {self.root / 'data'}")
        row_counts = [pq.ParquetFile(path).metadata.num_rows for path in self._data_files]
        self._file_offsets = np.cumsum([0, *row_counts]).tolist()
        self._num_rows = int(self._file_offsets[-1])
        self._indices = self._build_indices()
        self._episode_index_array: np.ndarray | None = None

    def _load_tasks(self) -> dict[int, str]:
        table = pq.read_table(self.root / "meta/tasks.parquet")
        rows = table.to_pydict()
        return {int(idx): str(task) for idx, task in zip(rows["task_index"], rows["task"])}

    def _load_front_camera_fov(self) -> tuple[np.ndarray, np.ndarray]:
        payload = json.loads((self.root / "meta/camera_info.json").read_text())
        attrs = payload["cam_front"]["attrs"]
        fov_yx = np.asarray([attrs["vertical_fov_rad"], attrs["horizontal_fov_rad"]], dtype=np.float32)
        image_hw = np.asarray([attrs["height"], attrs["width"]], dtype=np.float32)
        return fov_yx, image_hw

    def _build_indices(self) -> list[int] | None:
        if (
            self.task_indices is None
            and self.included_episodes is None
            and not self.excluded_anchors
            and self.anchor_stride == 1
        ):
            return None
        indices: list[np.ndarray] = []
        identity_parts: list[np.ndarray] = []
        matched_episodes: set[int] = set()
        matched_exclusions: set[tuple[int, int]] = set()
        for file_idx, path in enumerate(self._data_files):
            columns = ["task_index", "episode_index", "frame_index"]
            table = pq.read_table(path, columns=columns)
            task_array = table.column("task_index").combine_chunks().to_numpy(zero_copy_only=False)
            episodes = table.column("episode_index").combine_chunks().to_numpy(zero_copy_only=False)
            frames = table.column("frame_index").combine_chunks().to_numpy(zero_copy_only=False)
            keep = (
                np.ones(len(task_array), dtype=np.bool_)
                if self.task_indices is None
                else np.isin(task_array, list(self.task_indices))
            )
            keep &= frames % self.anchor_stride == 0
            if self.included_episodes is not None:
                episode_keep = np.isin(episodes, list(self.included_episodes))
                matched_episodes.update(int(episode) for episode in episodes[keep & episode_keep])
                keep &= episode_keep
            if self.excluded_anchors:
                for local_idx, (episode, frame) in enumerate(zip(episodes, frames, strict=True)):
                    key = (int(episode), int(frame))
                    if key in self.excluded_anchors:
                        matched_exclusions.add(key)
                        keep[local_idx] = False
            local_indices = np.flatnonzero(keep).astype(np.int64)
            if len(local_indices):
                indices.append(local_indices + self._file_offsets[file_idx])
                identity_parts.append(
                    np.stack(
                        [task_array[local_indices], episodes[local_indices], frames[local_indices]], axis=-1
                    ).astype(np.int64)
                )
        missing_exclusions = self.excluded_anchors - matched_exclusions
        if missing_exclusions:
            raise ValueError(f"Excluded Piper anchors were not found: {sorted(missing_exclusions)}")
        if self.included_episodes is not None:
            missing_episodes = self.included_episodes - matched_episodes
            if missing_episodes:
                raise ValueError(
                    "Included Piper episodes were not found inside the selected tasks: "
                    f"{sorted(missing_episodes)}"
                )
        if not indices:
            selected_tasks = None if self.task_indices is None else sorted(self.task_indices)
            raise ValueError(f"No rows matched task_indices={selected_tasks} under {self.root}")
        self._sample_identities = np.concatenate(identity_parts, axis=0)
        self._episode_frame_bounds: dict[int, tuple[int, int]] = {}
        for episode in np.unique(self._sample_identities[:, 1]):
            episode_frames = self._sample_identities[self._sample_identities[:, 1] == episode, 2]
            self._episode_frame_bounds[int(episode)] = (int(episode_frames.min()), int(episode_frames.max()))
        return np.concatenate(indices).astype(np.int64).tolist()

    def sample_identity(self, index: int) -> dict[str, int | str]:
        """Return lightweight sampling metadata without decoding images or actions."""
        if not hasattr(self, "_sample_identities"):
            global_idx = int(index)
            row = self._get_row(global_idx)
            task_index = int(row["task_index"])
            episode_index = int(row["episode_index"])
            frame_index = int(row["frame_index"])
            frame_start, frame_end = frame_index, frame_index
        else:
            task_index, episode_index, frame_index = (int(value) for value in self._sample_identities[int(index)])
            frame_start, frame_end = self._episode_frame_bounds[episode_index]
        denominator = max(1, frame_end - frame_start + 1)
        progress_bin = min(9, max(0, 10 * (frame_index - frame_start) // denominator))
        return {
            "task_index": task_index,
            "prompt": self._tasks[task_index],
            "episode_index": episode_index,
            "frame_index": frame_index,
            "episode_progress_bin": progress_bin,
        }

    def __len__(self) -> int:
        return self._num_rows if self._indices is None else len(self._indices)

    def global_index(self, index: int) -> int:
        return self._indices[int(index)] if self._indices is not None else int(index)

    def has_full_action_horizon(self, index: int) -> bool:
        global_idx = self.global_index(index)
        future_idx = global_idx + self.action_horizon - 1
        if future_idx >= self._num_rows:
            return False
        if self._episode_index_array is None:
            chunks = []
            for path in self._data_files:
                array = pq.read_table(path, columns=["episode_index"]).column("episode_index").combine_chunks()
                chunks.append(array.to_numpy(zero_copy_only=False).astype(np.int64, copy=False))
            self._episode_index_array = np.concatenate(chunks)
        return bool(self._episode_index_array[global_idx] == self._episode_index_array[future_idx])

    @functools.lru_cache(maxsize=PARQUET_FILE_CACHE_SIZE)
    def _load_file(self, file_idx: int) -> list[dict[str, Any]]:
        return pq.read_table(self._data_files[file_idx], columns=list(_COLUMNS)).to_pylist()

    def _get_row(self, global_idx: int) -> dict[str, Any]:
        file_idx = bisect.bisect_right(self._file_offsets, int(global_idx)) - 1
        if file_idx < 0 or file_idx >= len(self._data_files):
            raise IndexError(global_idx)
        local_idx = int(global_idx) - self._file_offsets[file_idx]
        return self._load_file(file_idx)[local_idx]

    def _row_in_episode(self, global_idx: int, episode_index: int) -> dict[str, Any] | None:
        if global_idx < 0 or global_idx >= self._num_rows:
            return None
        row = self._get_row(global_idx)
        return row if int(row["episode_index"]) == int(episode_index) else None

    def _frame_path(self, camera_key: str, episode_index: int, frame_index: int) -> Path:
        if self.frame_cache_dir is None:
            raise FileNotFoundError("Piper camera-token training requires OPENPI_PIPER_FRAME_CACHE_DIR")
        return (
            self.frame_cache_dir
            / _CAMERA_CACHE_KEYS[camera_key]
            / f"episode_{episode_index:06d}"
            / f"frame_{frame_index:06d}.jpg"
        )

    def _read_cached_frame(self, camera_key: str, episode_index: int, frame_index: int) -> np.ndarray:
        path = self._frame_path(camera_key, episode_index, frame_index)
        if not path.is_file():
            if self.strict_cache:
                raise FileNotFoundError(f"Missing cached Piper frame: {path}")
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.uint8)

    def __getitem__(self, index: int) -> dict[str, Any]:
        global_idx = self.global_index(index)
        row = self._get_row(global_idx)
        episode_index = int(row["episode_index"])
        frame_index = int(row["frame_index"])
        task_index = int(row["task_index"])
        prompt = self._tasks[task_index]

        current_front = self._read_cached_frame("cam_front", episode_index, frame_index)
        left_wrist = self._read_cached_frame("cam_left", episode_index, frame_index)
        right_wrist = self._read_cached_frame("cam_right", episode_index, frame_index)

        history_images = []
        history_masks = []
        camera_extrinsics = []
        for offset in self.history_offsets:
            history_row = self._row_in_episode(global_idx + int(offset), episode_index)
            valid = history_row is not None
            history_masks.append(valid)
            if valid:
                hist_frame = int(history_row["frame_index"])
                history_images.append(self._read_cached_frame("cam_front", episode_index, hist_frame))
                camera_extrinsics.append(
                    _as_float32(history_row["observation.camera_pose_in_unified.matrix.cam_front"]).reshape(4, 4)
                )
            else:
                history_images.append(np.zeros_like(current_front))
                camera_extrinsics.append(np.eye(4, dtype=np.float32))

        future_actions = []
        future_valid = []
        last_valid_action = _action23(row)
        for offset in range(self.action_horizon):
            future_row = self._row_in_episode(global_idx + offset, episode_index)
            valid = future_row is not None
            if valid:
                last_valid_action = _action23(future_row)
            future_actions.append(last_valid_action)
            future_valid.append(valid)

        state = _state23(row, gripper_indices=self.state_gripper_indices)
        actions = _unwrap_action_quaternion_signs(
            np.stack(future_actions, axis=0).astype(np.float32),
            state,
            np.asarray(future_valid, dtype=np.bool_),
        )
        action_time_valid = np.asarray(future_valid, dtype=np.bool_)
        action_dim_mask = np.broadcast_to(action_time_valid[:, None], (self.action_horizon, ROBOT_ACTION_DIM)).copy()
        history_mask = np.asarray(history_masks, dtype=np.bool_)

        return {
            "image": {
                "base_0_rgb": current_front,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.asarray(True),
                "left_wrist_0_rgb": np.asarray(True),
                "right_wrist_0_rgb": np.asarray(True),
            },
            "state": state.astype(np.float32),
            "actions": actions.astype(np.float32),
            "action_dim_mask": action_dim_mask,
            "action_time_valid_mask": action_time_valid,
            "prompt": prompt,
            "front_history_images": np.stack(history_images, axis=0),
            "front_history_masks": history_mask,
            "camera_extrinsics": np.stack(camera_extrinsics, axis=0).astype(np.float32),
            "camera_pose_valid": history_mask,
            "camera_fov": np.broadcast_to(self._camera_fov_yx[None, :], (len(self.history_offsets), 2)).copy(),
            "camera_fov_valid": history_mask.copy(),
            "camera_image_hw": np.broadcast_to(self._camera_hw[None, :], (len(self.history_offsets), 2)).copy(),
            "episode_index": np.asarray(episode_index, dtype=np.int64),
            "frame_index": np.asarray(frame_index, dtype=np.int64),
            "task_index": np.asarray(task_index, dtype=np.int64),
        }
