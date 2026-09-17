"""Public, path-agnostic ActiveScale training recipes.

The model and optimizer values mirror the paper runs. Dataset locations are
read from environment variables so this module never embeds lab filesystem
paths or credentials.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
import os
import pathlib

from typing_extensions import override

import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer


def _env(name: str, default: str) -> str:
    return os.path.expandvars(os.environ.get(name, default))


def _csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in _env(name, default).split(",") if item.strip())


def _optional_csv(name: str, default: str) -> tuple[str | None, ...]:
    return tuple(
        None if item.strip() in {"", "-", "none", "None"} else item.strip()
        for item in _env(name, default).split(",")
    )


def _index_group(spec: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending task range: {item}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(item))
    if not values:
        raise ValueError(f"Empty task-index group: {spec!r}")
    return tuple(values)


def _index_groups(name: str, root_count: int) -> tuple[tuple[int, ...], ...]:
    raw = os.environ.get(name)
    if raw is None:
        return tuple(tuple(range(35)) for _ in range(root_count))
    groups = tuple(_index_group(group) for group in raw.split(";"))
    if len(groups) != root_count:
        raise ValueError(f"{name} must contain one semicolon-separated group per Piper root")
    return groups


def _path_slots(name: str, root_count: int) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return ()
    paths = tuple("" if item.strip() in {"", "-", "none", "None"} else item.strip() for item in raw.split(","))
    if len(paths) != root_count:
        raise ValueError(f"{name} must contain one comma-separated slot per Piper root")
    return paths


def _anchor_groups(name: str, root_count: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return ()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} must be a JSON list with one anchor list per Piper root") from error
    if not isinstance(payload, list) or len(payload) != root_count:
        raise ValueError(f"{name} must contain one anchor list per Piper root")
    groups: list[tuple[tuple[int, int], ...]] = []
    for group in payload:
        if not isinstance(group, list):
            raise ValueError(f"{name} anchor groups must be lists")
        anchors: list[tuple[int, int]] = []
        for anchor in group:
            if not isinstance(anchor, list) or len(anchor) != 2:
                raise ValueError(f"{name} anchors must be [episode_index, frame_index] pairs")
            anchors.append((int(anchor[0]), int(anchor[1])))
        groups.append(tuple(anchors))
    return tuple(groups)


@dataclasses.dataclass(frozen=True)
class HumanRobotCotrainDataConfig(_config.DataConfigFactory):
    """Model-ready human packs mixed 1:1 with robot trajectories."""

    repo_id: str = "activescale_human_robot"
    human_pack_paths: str = "$ACTIVESCALE_DATA_ROOT/human/**/*.parquet"
    human_source_views: str | None = None
    human_norm_stats_path: str | None = None
    human_source_norm_stats_paths: str | None = None
    human_selection_manifest: str | None = "$ACTIVESCALE_ASSETS_ROOT/manifests/human.jsonl"
    fast_tokenizer_path: str = "physical-intelligence/fast"
    piper_roots: Sequence[str] = ("$ACTIVESCALE_DATA_ROOT/robot/piper",)
    piper_task_indices: Sequence[Sequence[int]] = (tuple(range(35)),)
    piper_frame_cache_dirs: Sequence[str] = ("$ACTIVESCALE_CACHE_ROOT/piper",)
    piper_included_episodes_paths: Sequence[str] = ()
    piper_excluded_anchors: Sequence[Sequence[tuple[int, int]]] = ()
    piper_norm_stats_path: str = "$ACTIVESCALE_ASSETS_ROOT/norm/piper.json"
    public_robot_sources: Sequence[str] = ("AgiBotWorld2026",)
    public_robot_manifest_paths: Sequence[str] = (
        "$ACTIVESCALE_ASSETS_ROOT/manifests/agibotworld2026.jsonl",
    )
    public_robot_roots: Sequence[str | None] = (None,)
    public_robot_frame_cache_dirs: Sequence[str] = ("$ACTIVESCALE_CACHE_ROOT/agibotworld2026",)
    public_robot_norm_stats_dir: str = "$ACTIVESCALE_ASSETS_ROOT/norm/public_robot"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: pi0_config.Pi0Config) -> _config.DataConfig:
        del assets_dirs, model_config
        return _config.DataConfig(
            repo_id=self.repo_id,
            asset_id=None,
            norm_stats=None,
            offline_human_vlm_pack_paths=os.path.expandvars(self.human_pack_paths),
            offline_human_vlm_source_weights=None,
            offline_human_vlm_fast_tokenizer_path=self.fast_tokenizer_path,
            offline_human_vlm_enable_subtask_view=True,
            offline_human_vlm_enable_fast_view=True,
            offline_human_vlm_source_views=self.human_source_views,
            offline_human_vlm_fast_horizon=50,
            offline_human_vlm_allow_partial_fast_horizon=True,
            offline_human_vlm_norm_stats_path=(
                None if self.human_norm_stats_path is None else os.path.expandvars(self.human_norm_stats_path)
            ),
            offline_human_vlm_source_norm_stats_paths=self.human_source_norm_stats_paths,
            offline_human_vlm_action_norm_stats_path=None,
            offline_human_vlm_action_loss_weight=0.2,
            offline_human_vlm_split="all",
            offline_human_vlm_selection_manifest_path=(
                None
                if self.human_selection_manifest is None
                else os.path.expandvars(self.human_selection_manifest)
            ),
            offline_human_vlm_view_cycle_mode="physical_once",
            human_piper_cotrain=True,
            cotrain_piper_roots=tuple(os.path.expandvars(path) for path in self.piper_roots),
            cotrain_piper_task_indices=tuple(tuple(ids) for ids in self.piper_task_indices),
            cotrain_piper_frame_cache_dirs=tuple(
                os.path.expandvars(path) for path in self.piper_frame_cache_dirs
            ),
            cotrain_piper_included_episodes_paths=tuple(
                os.path.expandvars(path) if path else "" for path in self.piper_included_episodes_paths
            ),
            cotrain_piper_excluded_anchors=tuple(tuple(group) for group in self.piper_excluded_anchors),
            cotrain_piper_norm_stats_path=os.path.expandvars(self.piper_norm_stats_path),
            cotrain_human_slots=1,
            cotrain_piper_slots=1,
            cotrain_piper_action_loss_weight=1.0,
            cotrain_piper_anchor_stride=6,
            cotrain_epoch_size_multiple=256,
            cotrain_public_robot_sources=tuple(self.public_robot_sources),
            cotrain_public_robot_manifest_paths=tuple(
                os.path.expandvars(path) for path in self.public_robot_manifest_paths
            ),
            cotrain_public_robot_roots=tuple(
                None if path is None else os.path.expandvars(path) for path in self.public_robot_roots
            ),
            cotrain_public_robot_frame_cache_dirs=tuple(
                os.path.expandvars(path) for path in self.public_robot_frame_cache_dirs
            ),
            cotrain_public_robot_norm_stats_dir=os.path.expandvars(self.public_robot_norm_stats_dir),
            cotrain_public_robot_anchor_stride=6,
            cotrain_public_robot_action_loss_weight=1.0,
        )


@dataclasses.dataclass(frozen=True)
class PiperPosttrainDataConfig(_config.DataConfigFactory):
    """Piper LeRobot-v3 data with synchronized front-camera history."""

    repo_id: str = "piper"
    task_indices: tuple[int, ...] | None = None
    frame_cache_dir: str | None = "$ACTIVESCALE_CACHE_ROOT/piper"
    default_prompt: str | None = None
    state_gripper_indices: tuple[int, int] = (6, 13)
    excluded_anchors: tuple[tuple[int, int], ...] = ()

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: pi0_config.Pi0Config) -> _config.DataConfig:
        model_transforms = _config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            model_transforms=model_transforms,
            piper_camera_token_dataset=True,
            piper_task_indices=None if self.task_indices is None else tuple(self.task_indices),
            piper_frame_cache_dir=(
                None if self.frame_cache_dir is None else os.path.expandvars(self.frame_cache_dir)
            ),
            piper_state_gripper_indices=self.state_gripper_indices,
            piper_excluded_anchors=self.excluded_anchors,
        )


def _model_config() -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=50,
        max_token_len=512,
        pytorch_compile_mode=None,
        training_objective="human_joint_camera",
        enable_front_camera_tokens=True,
        front_camera_history_offsets=(-48, -32, -16, 0),
        joint_loss_weight_subtask=1.0,
        joint_loss_weight_fast=1.0,
        joint_loss_weight_flow=1.0,
        front_camera_pose_loss_weight=0.2,
        front_camera_pose_loss_weight_trans=1.0,
        front_camera_pose_loss_weight_rot=1.0,
        front_camera_pose_loss_weight_focal=0.5,
        front_camera_pose_loss_normalize_trans=False,
    )


def _deployed_piper_model_config() -> pi0_config.Pi0Config:
    """Architecture used by the released five-task Piper policies."""

    return pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=50,
        max_token_len=512,
        pytorch_compile_mode=None,
        training_objective="flow_camera",
        enable_front_camera_tokens=True,
        front_camera_pose_loss_weight=0.2,
        front_camera_pose_loss_normalize_trans=False,
    )


def _piper_policy_configs() -> tuple[_config.TrainConfig, ...]:
    """Inference configs matching the deployed history-mask checkpoints."""

    model = _deployed_piper_model_config()
    repo_id = _env("ACTIVESCALE_PIPER_REPO_ID", "piper")
    frame_cache_dir = _env("ACTIVESCALE_PIPER_CACHE", "$ACTIVESCALE_CACHE_ROOT/piper")
    tasks = (
        ("bag", (0,), "cobot-avp-teleop_lerobot_task0_bag_camera_clean_v2"),
        ("drawer", tuple(range(1, 11)), "cobot-avp-teleop_lerobot_task1_10_drawer_camera_new_20260906"),
        ("pot", tuple(range(11, 16)), "cobot-avp-teleop_lerobot_task11_15_pot_camera"),
        ("box", tuple(range(16, 26)), "cobot-avp-teleop_lerobot_task16_25_box_camera_new_20260906"),
        (
            "under",
            tuple(range(26, 35)),
            "cobot-avp-teleop_lerobot_task26_34_under_reannotated_camera",
        ),
    )
    return tuple(
        _config.TrainConfig(
            name=f"activescale_pi05_piper_{task}_policy",
            project_name="activescale",
            model=model,
            data=PiperPosttrainDataConfig(
                repo_id=repo_id,
                assets=_config.AssetsConfig(asset_id=asset_id),
                task_indices=task_indices,
                frame_cache_dir=frame_cache_dir,
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=5e-5,
                decay_steps=1_000_000,
                decay_lr=5e-5,
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            batch_size=32,
            num_workers=0,
            ema_decay=None,
            num_train_steps=1,
            policy_metadata={"task_family": task, "history_contract": "mask"},
        )
        for task, task_indices, asset_id in tasks
    )


def _cotrain_data() -> HumanRobotCotrainDataConfig:
    piper_roots = _csv("ACTIVESCALE_PIPER_ROOTS", "$ACTIVESCALE_DATA_ROOT/robot/piper")
    piper_caches = _csv("ACTIVESCALE_PIPER_CACHES", "$ACTIVESCALE_CACHE_ROOT/piper")
    if len(piper_roots) != len(piper_caches):
        raise ValueError("ACTIVESCALE_PIPER_ROOTS and ACTIVESCALE_PIPER_CACHES must have equal lengths")
    piper_tasks = _index_groups("ACTIVESCALE_PIPER_TASK_INDICES", len(piper_roots))
    piper_included = _path_slots("ACTIVESCALE_PIPER_INCLUDED_EPISODES", len(piper_roots))
    piper_excluded = _anchor_groups("ACTIVESCALE_PIPER_EXCLUDED_ANCHORS", len(piper_roots))

    public_sources = _csv("ACTIVESCALE_PUBLIC_ROBOT_SOURCES", "AgiBotWorld2026")
    public_manifests = _csv(
        "ACTIVESCALE_PUBLIC_ROBOT_MANIFESTS",
        "$ACTIVESCALE_ASSETS_ROOT/manifests/agibotworld2026.jsonl",
    )
    public_roots = _optional_csv("ACTIVESCALE_PUBLIC_ROBOT_ROOTS", "-")
    public_caches = _csv(
        "ACTIVESCALE_PUBLIC_ROBOT_CACHES",
        "$ACTIVESCALE_CACHE_ROOT/agibotworld2026",
    )
    if not (len(public_sources) == len(public_manifests) == len(public_roots) == len(public_caches)):
        raise ValueError("ACTIVESCALE_PUBLIC_ROBOT_* lists must have equal comma-separated lengths")

    return HumanRobotCotrainDataConfig(
        human_pack_paths=_env("ACTIVESCALE_HUMAN_PACKS", "$ACTIVESCALE_DATA_ROOT/human/**/*.parquet"),
        human_source_views=_env(
            "ACTIVESCALE_HUMAN_SOURCE_VIEWS",
            "EgoLive=high_level:1+low_level:1,EgoVerse=low_level,"
            "EgoProStandard=high_level:1+low_level:1",
        ),
        human_norm_stats_path=os.environ.get("ACTIVESCALE_HUMAN_NORM_STATS"),
        human_source_norm_stats_paths=_env(
            "ACTIVESCALE_HUMAN_SOURCE_NORM_STATS",
            "EgoLive=$ACTIVESCALE_ASSETS_ROOT/norm/human/EgoLive.json,"
            "EgoVerse=$ACTIVESCALE_ASSETS_ROOT/norm/human/EgoVerse.json,"
            "EgoProStandard=$ACTIVESCALE_ASSETS_ROOT/norm/human/EgoProStandard.json",
        ),
        human_selection_manifest=_env(
            "ACTIVESCALE_HUMAN_SELECTION", "$ACTIVESCALE_ASSETS_ROOT/manifests/human.jsonl"
        ),
        fast_tokenizer_path=_env("ACTIVESCALE_FAST_TOKENIZER", "physical-intelligence/fast"),
        piper_roots=piper_roots,
        piper_task_indices=piper_tasks,
        piper_frame_cache_dirs=piper_caches,
        piper_included_episodes_paths=piper_included,
        piper_excluded_anchors=piper_excluded,
        piper_norm_stats_path=_env(
            "ACTIVESCALE_PIPER_NORM_STATS", "$ACTIVESCALE_ASSETS_ROOT/norm/piper.json"
        ),
        public_robot_sources=public_sources,
        public_robot_manifest_paths=public_manifests,
        public_robot_roots=public_roots,
        public_robot_frame_cache_dirs=public_caches,
        public_robot_norm_stats_dir=_env(
            "ACTIVESCALE_PUBLIC_ROBOT_NORM_DIR", "$ACTIVESCALE_ASSETS_ROOT/norm/public_robot"
        ),
    )


def make_configs() -> tuple[_config.TrainConfig, ...]:
    """Return release configs registered by ``openpi.training.config``."""

    model = _model_config()
    data = _cotrain_data()
    full = _config.TrainConfig(
        name="activescale_pi05_human_robot_h50",
        project_name="activescale",
        model=model,
        data=data,
        batch_size=256,
        gradient_accumulation_steps=1,
        num_workers=4,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-10,
            clip_gradient_norm=1.0,
        ),
        ema_decay=None,
        pytorch_weight_path=_env("ACTIVESCALE_BASE_CHECKPOINT", "./checkpoints/pi05_base_pytorch"),
        # The public launcher always overrides run length and checkpoint cadence.
        num_train_steps=1,
        log_interval=50,
        save_interval=1,
        keep_period=None,
    )
    smoke = dataclasses.replace(
        full,
        name="activescale_pi05_human_robot_h50_smoke",
        num_train_steps=10,
        log_interval=1,
        save_interval=10,
        keep_period=10,
        wandb_enabled=False,
    )
    posttrain = _config.TrainConfig(
        name="activescale_pi05_piper_posttrain_h50",
        project_name="activescale",
        model=_deployed_piper_model_config(),
        data=PiperPosttrainDataConfig(
            repo_id=_env("ACTIVESCALE_PIPER_REPO_ID", "piper"),
            task_indices=None,
            frame_cache_dir=_env("ACTIVESCALE_PIPER_CACHE", "$ACTIVESCALE_CACHE_ROOT/piper"),
        ),
        batch_size=32,
        num_workers=0,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            # The launcher replaces this with the user-selected run length.
            decay_steps=1,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        pytorch_weight_path=_env("ACTIVESCALE_MIDTRAIN_CHECKPOINT", "./checkpoints/activescale_midtrain"),
        num_train_steps=1,
        save_interval=1,
        keep_period=None,
    )
    return smoke, full, posttrain, *_piper_policy_configs()
