from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.training.piper_camera_token_dataset import PiperCameraTokenLeRobotDataset


def _write_index_only_repo(tmp_path):
    root = tmp_path / "piper"
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"fps": 30}))
    (root / "meta/camera_info.json").write_text(
        json.dumps(
            {
                "cam_front": {
                    "attrs": {
                        "vertical_fov_rad": 1.0,
                        "horizontal_fov_rad": 1.2,
                        "height": 1080,
                        "width": 1920,
                    }
                }
            }
        )
    )
    pq.write_table(
        pa.table({"task_index": [0, 1], "task": ["bag", "drawer"]}),
        root / "meta/tasks.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "task_index": [0, 0, 1],
                "episode_index": [66, 66, 1],
                "frame_index": [544, 545, 0],
            }
        ),
        root / "data/chunk-000/file-000.parquet",
    )
    return root


def test_excluded_anchor_is_removed_without_compressing_global_indices(tmp_path):
    root = _write_index_only_repo(tmp_path)
    dataset = PiperCameraTokenLeRobotDataset(
        root,
        action_horizon=50,
        task_indices=(0,),
        excluded_anchors=((66, 545),),
    )

    assert len(dataset) == 1
    assert dataset._indices == [0]
    assert dataset._num_rows == 3


def test_configured_excluded_anchor_must_exist(tmp_path):
    root = _write_index_only_repo(tmp_path)
    with pytest.raises(ValueError, match="Excluded Piper anchors were not found"):
        PiperCameraTokenLeRobotDataset(
            root,
            action_horizon=50,
            task_indices=(0,),
            excluded_anchors=((66, 999),),
        )


def test_included_episodes_are_a_fail_closed_allowlist(tmp_path):
    root = _write_index_only_repo(tmp_path)
    dataset = PiperCameraTokenLeRobotDataset(
        root,
        action_horizon=1,
        task_indices=(0, 1),
        included_episodes=(1,),
    )

    assert len(dataset) == 1
    assert dataset.sample_identity(0)["episode_index"] == 1


def test_included_episode_must_exist_inside_selected_tasks(tmp_path):
    root = _write_index_only_repo(tmp_path)
    with pytest.raises(ValueError, match="inside the selected tasks"):
        PiperCameraTokenLeRobotDataset(
            root,
            action_horizon=1,
            task_indices=(0,),
            included_episodes=(1,),
        )


def test_anchor_stride_downsamples_without_changing_dense_horizon(tmp_path):
    root = _write_index_only_repo(tmp_path)
    dataset = PiperCameraTokenLeRobotDataset(
        root,
        action_horizon=50,
        anchor_stride=6,
        task_indices=(0, 1),
    )

    assert len(dataset) == 1
    assert dataset.sample_identity(0)["frame_index"] == 0
    assert dataset.action_horizon == 50
