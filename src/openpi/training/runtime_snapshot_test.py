from __future__ import annotations

import dataclasses
import json

from openpi.training import runtime_snapshot


@dataclasses.dataclass
class _Model:
    action_horizon: int = 10
    front_camera_history_offsets: tuple[int, ...] = (-48, -32, -16, 0)
    front_camera_pose_loss_weight: float = 0.2
    front_camera_pose_loss_weight_trans: float = 1.0
    front_camera_pose_loss_weight_rot: float = 1.0
    front_camera_pose_loss_weight_focal: float = 0.5
    joint_loss_weight_subtask: float = 0.7
    joint_loss_weight_fast: float = 0.8
    joint_loss_weight_flow: float = 0.9


@dataclasses.dataclass
class _Data:
    pack_paths: str = "$HF_LEROBOT_HOME/pack/MANIFEST.json"
    source_weights: str = "EgoDex=4,EgoLive=5,VITRA=1"
    source_views: str = "EgoDex=high_level:1+low_level:1"
    split: str = "train"
    val_basis_points: int = 200
    norm_stats_path: str | None = None
    action_norm_stats_path: str | None = None


@dataclasses.dataclass
class _Optimizer:
    weight_decay: float = 0.01


@dataclasses.dataclass
class _Schedule:
    warmup_steps: int = 0
    peak_lr: float = 5e-5
    decay_steps: int = 175_000
    decay_lr: float = 5e-5


@dataclasses.dataclass
class _Config:
    name: str = "formal"
    exp_name: str = "test"
    resume: bool = False
    seed: int = 42
    num_train_steps: int = 175_000
    save_interval: int = 25_000
    log_interval: int = 50
    batch_size: int = 256
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    pytorch_training_precision: str = "bfloat16"
    data: _Data = dataclasses.field(default_factory=_Data)
    model: _Model = dataclasses.field(default_factory=_Model)
    optimizer: _Optimizer = dataclasses.field(default_factory=_Optimizer)
    lr_schedule: _Schedule = dataclasses.field(default_factory=_Schedule)
    pytorch_weight_path: str | None = None


def test_snapshot_records_physical_and_batch_contract(tmp_path, monkeypatch):
    manifest = tmp_path / "CACHE_MANIFEST.json"
    manifest.write_text('{"version":"v3"}\n')
    monkeypatch.setenv("FRAME_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FRAME_CACHE_MANIFEST", str(manifest))
    monkeypatch.setenv("USE_FRAME_CACHE", "1")

    snapshot = runtime_snapshot.build_runtime_snapshot(_Config(), repo_root=tmp_path, world_size=8)

    assert snapshot["batch"]["per_device_batch"] == 32
    assert snapshot["batch"]["effective_global_batch"] == 256
    assert snapshot["run"]["precision"] == "bfloat16"
    assert snapshot["data"]["history_offsets_seconds"] == [-1.6, -32 / 30, -16 / 30, 0.0]
    assert snapshot["data"]["action_horizon_seconds"] == 10 / 30
    assert snapshot["data"]["frame_cache"]["manifest"]["sha256"] == runtime_snapshot.sha256_file(manifest)
    assert snapshot["loss_weights"] == {
        "subtask": 0.7,
        "fast": 0.8,
        "flow": 0.9,
        "camera": 0.2,
        "camera_translation": 1.0,
        "camera_rotation": 1.0,
        "camera_fov": 0.5,
    }


def test_contract_hash_ignores_extended_train_budget(tmp_path):
    first = runtime_snapshot.build_runtime_snapshot(_Config(), repo_root=tmp_path, world_size=8)
    second = runtime_snapshot.build_runtime_snapshot(
        dataclasses.replace(_Config(), num_train_steps=300_000), repo_root=tmp_path, world_size=8
    )
    assert first["contract_sha256"] == second["contract_sha256"]


def test_contract_hash_changes_with_cache_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "CACHE_MANIFEST.json"
    manifest.write_text('{"version":"v3"}\n')
    monkeypatch.setenv("FRAME_CACHE_DIR", str(tmp_path))
    first = runtime_snapshot.build_runtime_snapshot(_Config(), repo_root=tmp_path, world_size=8)
    manifest.write_text('{"version":"v4"}\n')
    second = runtime_snapshot.build_runtime_snapshot(_Config(), repo_root=tmp_path, world_size=8)
    assert first["contract_sha256"] != second["contract_sha256"]


def test_contract_hash_changes_with_execution_settings(tmp_path, monkeypatch):
    first = runtime_snapshot.build_runtime_snapshot(_Config(), repo_root=tmp_path, world_size=8)
    monkeypatch.setenv("PI05_DISABLE_OUTER_FLOW_CHECKPOINT", "1")
    monkeypatch.setenv("PI05_DISABLE_OUTER_IMAGE_CHECKPOINT", "1")
    monkeypatch.setenv("PI05_TRIM_TRAILING_TOKEN_PADDING", "1")
    monkeypatch.setenv("PI05_DISABLE_LIGHTWEIGHT_CHECKPOINTS", "1")
    second = runtime_snapshot.build_runtime_snapshot(_Config(), repo_root=tmp_path, world_size=8)

    assert first["contract_sha256"] != second["contract_sha256"]
    assert second["attention_execution"] == {
        "outer_flow_checkpoint_disabled": "1",
        "outer_image_checkpoint_disabled": "1",
        "trailing_token_padding_trimmed": "1",
        "lightweight_checkpoints_disabled": "1",
    }


def test_atomic_snapshot_update(tmp_path):
    path = tmp_path / "resolved_config.json"
    runtime_snapshot.write_runtime_snapshot(path, {"contract_sha256": "abc"})
    runtime_snapshot.update_runtime_snapshot(path, {"model_initialization": {"loaded": 3}})
    assert json.loads(path.read_text())["model_initialization"]["loaded"] == 3
