import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation

from openpi.models.tokenizer import TokenizedVLMView
from openpi.training.public_robot_cotrain_dataset import PublicRobotJointObjectiveAdapter
from openpi.training.public_robot_cotrain_dataset import PublicRobotLeRobotDataset
from openpi.training.public_robot_cotrain_dataset import ROBOT_NORM_LAYOUT
from openpi.training.public_robot_cotrain_dataset import _euler_xyz_to_quaternion


class _Tokenizer:
    def tokenize_fast_action(self, prompt, state, actions):
        assert prompt
        assert state.shape == (32,)
        assert actions.shape[1] == 23
        return TokenizedVLMView(
            tokens=np.asarray([1, 2, 0], dtype=np.int32),
            token_mask=np.asarray([True, True, False]),
            ar_mask=np.asarray([0, 1, 0], dtype=np.int32),
            loss_mask=np.asarray([False, True, False]),
            truncated=False,
        )

    def tokenize_action_condition(self, prompt, state):
        del prompt, state
        return TokenizedVLMView(
            tokens=np.asarray([1, 0, 0], dtype=np.int32),
            token_mask=np.asarray([True, False, False]),
            ar_mask=np.zeros(3, dtype=np.int32),
            loss_mask=np.zeros(3, dtype=np.bool_),
            truncated=False,
        )


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _write_norm(root, source, embodiment):
    safe_source = source.replace("-", "-")
    payload = {
        "action_layout": ROBOT_NORM_LAYOUT,
        "state_q01": [-2.0] * 16,
        "state_q99": [2.0] * 16,
        "action_q01": [-2.0] * 16,
        "action_q99": [2.0] * 16,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{safe_source}__{embodiment}.json").write_text(json.dumps(payload))
    return root


def _beta_dataset(tmp_path):
    rows = []
    for frame in range(12):
        rows.append(
            {
                "observation.states.end.position": [[0.5, 0.2, 0.7], [0.5, -0.2, 0.7]],
                "observation.states.end.orientation": [[0.0, 0.0, 0.0, 1.0]] * 2,
                "observation.states.effector.position": [40.0, 50.0],
                "actions.end.position": [[0.5 + frame / 1000, 0.2, 0.7], [0.5, -0.2, 0.7]],
                "actions.end.orientation": [[0.0, 0.0, 0.0, 1.0]] * 2,
                "actions.effector.position": [0.0, 1.0],
            }
        )
    parquet = tmp_path / "beta.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    manifest = _write_jsonl(
        tmp_path / "beta.jsonl",
        [
            {
                "source": "AgiBotWorld-Beta",
                "task_dataset": "task_1",
                "episode_index": 0,
                "length": 12,
                "embodiment": "a2d",
                "task": "move objects",
                "parquet": str(parquet),
                "videos": {},
                "action_config": [
                    {"start_frame": 2, "end_frame": 8, "action_text": "pick corn", "skill": "Pick"}
                ],
            }
        ],
    )
    return PublicRobotLeRobotDataset(
        source="AgiBotWorld-Beta",
        manifest_path=manifest,
        action_horizon=5,
        anchor_stride=6,
        frame_cache_dir=None,
        fake_images=True,
    )


def test_euler_xyz_conversion_matches_scipy():
    euler = np.asarray([[0.2, -0.4, 1.1], [-1.0, 0.3, 0.7]], dtype=np.float32)
    assert np.allclose(_euler_xyz_to_quaternion(euler), Rotation.from_euler("xyz", euler).as_quat())


def test_beta_adapter_masks_camera_and_subtask_boundary(tmp_path):
    base = _beta_dataset(tmp_path)
    assert len(base) == 2
    assert base.representative_indices_by_embodiment() == {"a2d": 1}
    raw = base[1]
    assert raw["prompt"] == "pick corn"
    assert raw["action_time_valid_mask"].tolist() == [True, True, False, False, False]
    adapter = PublicRobotJointObjectiveAdapter(
        base,
        tokenizer=_Tokenizer(),
        norm_stats_dir=_write_norm(tmp_path / "norms", "AgiBotWorld-Beta", "a2d"),
    )
    item = adapter[1]
    assert item["state_pose_valid"].tolist() == [True, True, False]
    assert not item["action_dim_mask"][:, 14:21].any()
    assert item["action_dim_mask"][:2, :14].all()
    assert not item["camera_pose_valid"].any()
    assert item["front_history_masks"].tolist() == [False, False, False, True]


def test_agibotworld2026_fixed_list_schema_maps_to_canonical_eef(tmp_path):
    info = {
        "features": {
            "observation.state": {
                "field_descriptions": {
                    "state/end/arm_position": {"indices": list(range(0, 6))},
                    "state/end/arm_orientation": {"indices": list(range(6, 14))},
                    "state/left_effector/position": {"indices": [14]},
                    "state/right_effector/position": {"indices": [15]},
                }
            },
            "action": {
                "field_descriptions": {
                    "action/end/position": {"indices": list(range(0, 6))},
                    "action/end/orientation": {"indices": list(range(6, 14))},
                    "action/left_effector/position": {"indices": [14]},
                    "action/right_effector/position": {"indices": [15]},
                }
            },
        }
    }
    info_path = tmp_path / "info.json"
    info_path.write_text(json.dumps(info))
    quaternion_pair = [0.0, 0.0, 0.0, 1.0] * 2
    rows = []
    for frame in range(8):
        rows.append(
            {
                "observation.state": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, *quaternion_pair, 0.2, 0.8],
                "action": [1.0 + frame / 100.0, 2.0, 3.0, 4.0, 5.0, 6.0, *quaternion_pair, 0.3, 0.7],
            }
        )
    parquet = tmp_path / "agibot2026.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    manifest = _write_jsonl(
        tmp_path / "agibot2026.jsonl",
        [
            {
                "source": "AgiBotWorld2026",
                "dataset": "Home/task_1/archive_1",
                "episode_index": 0,
                "length": len(rows),
                "embodiment": "g2a",
                "task": "put corn in a box",
                "info": str(info_path),
                "parquet": str(parquet),
            }
        ],
    )
    dataset = PublicRobotLeRobotDataset(
        source="AgiBotWorld2026",
        manifest_path=manifest,
        action_horizon=5,
        anchor_stride=6,
        frame_cache_dir=None,
        fake_images=True,
    )
    sample = dataset[0]
    assert sample["state"].tolist() == pytest.approx(
        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.8]
    )
    assert sample["actions"][1].tolist() == pytest.approx(
        [1.01, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0, 0.3, 0.7]
    )
    assert sample["prompt"] == "put corn in a box"
    assert sample["front_history_masks"].tolist() == [False, False, False, True]


def test_robocoin_uses_most_specific_active_subtask(tmp_path):
    root = tmp_path / "robocoin"
    dataset_root = root / "robot_task"
    info = {
        "robot_type": "alpha_bot_2",
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta/info.json").write_text(json.dumps(info))
    _write_jsonl(
        dataset_root / "annotations/subtask_annotations.jsonl",
        [
            {"subtask_index": 0, "subtask": "Left gripper"},
            {"subtask_index": 1, "subtask": "Move corn into the box"},
            {"subtask_index": 2, "subtask": "null."},
        ],
    )
    rows = [
        {
            "eef_sim_pose_state": [0.0] * 12,
            "eef_sim_pose_action": [0.0] * 12,
            "gripper_open_scale_state": [0.5, 0.5],
            "gripper_open_scale_action": [0.5, 0.5],
            "subtask_annotation": [0, 1, 2, 2, 2],
        }
        for _ in range(8)
    ]
    parquet = dataset_root / "data/chunk-000/episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    manifest = _write_jsonl(
        tmp_path / "robocoin.jsonl",
        [
            {
                "dataset": "robot_task",
                "episode_index": 0,
                "frames": 8,
                "robot_type": "alpha_bot_2",
                "tasks": ["put an object away"],
            }
        ],
    )
    dataset = PublicRobotLeRobotDataset(
        source="RoboCOIN",
        manifest_path=manifest,
        root=root,
        action_horizon=5,
        anchor_stride=6,
        frame_cache_dir=None,
        fake_images=True,
    )
    assert dataset.representative_indices_by_embodiment() == {"alpha_bot_2": 1}
    assert dataset[0]["prompt"] == "Move corn into the box"
