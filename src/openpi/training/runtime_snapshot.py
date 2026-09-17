"""Reproducible runtime snapshots for PyTorch training runs."""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "sha": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"sha": None, "branch": None, "dirty": None, "error": str(exc)}


def _artifact(path_text: str | None, *, hash_large_file: bool = True) -> dict[str, Any] | None:
    if not path_text:
        return None
    expanded = Path(os.path.expandvars(os.path.expanduser(path_text))).resolve()
    record: dict[str, Any] = {"path": str(expanded), "exists": expanded.exists()}
    if not expanded.is_file():
        return record
    record["size_bytes"] = expanded.stat().st_size
    if hash_large_file or expanded.stat().st_size <= 64 * 1024 * 1024:
        record["sha256"] = sha256_file(expanded)
    return record


def _pack_inputs(pack_paths: str | None) -> list[dict[str, Any]]:
    if not pack_paths:
        return []
    records = []
    for raw in pack_paths.split(","):
        raw = raw.strip()
        if not raw:
            continue
        expanded = os.path.expandvars(os.path.expanduser(raw))
        path = Path(expanded)
        record: dict[str, Any] = {"configured": raw, "expanded": expanded}
        if not any(char in expanded for char in "*?[") and path.is_file():
            record.update(_artifact(expanded) or {})
        records.append(record)
    return records


def _cache_manifest(cache_root: str | None, explicit_manifest: str | None) -> dict[str, Any] | None:
    if not cache_root:
        return None
    root = Path(os.path.expandvars(os.path.expanduser(cache_root))).resolve()
    candidates = []
    if explicit_manifest:
        candidates.append(Path(os.path.expandvars(os.path.expanduser(explicit_manifest))))
    candidates.extend(root / name for name in ("CACHE_MANIFEST.json", "MANIFEST.json", "cache_manifest.json"))
    manifest = next((path for path in candidates if path.is_file()), None)
    return {
        "root": str(root),
        "exists": root.is_dir(),
        "manifest": _artifact(str(manifest)) if manifest is not None else None,
    }


def _checkpoint_identity(path_text: str | None) -> dict[str, Any] | None:
    if not path_text:
        return None
    root = Path(os.path.expandvars(os.path.expanduser(path_text))).resolve()
    model_path = root / "model.safetensors" if root.is_dir() else root
    sidecars = (
        model_path.with_suffix(model_path.suffix + ".sha256"),
        model_path.parent / "model.safetensors.sha256",
    )
    sidecar = next((path for path in sidecars if path.is_file()), None)
    return {
        "path": str(root),
        "model": _artifact(str(model_path), hash_large_file=False),
        "sha256_sidecar": _artifact(str(sidecar)) if sidecar is not None else None,
    }


def build_runtime_snapshot(config: Any, *, repo_root: Path, world_size: int) -> dict[str, Any]:
    data = getattr(config, "data", None)
    model = getattr(config, "model", None)
    lr = getattr(config, "lr_schedule", None)
    optimizer = getattr(config, "optimizer", None)
    batch_size = int(config.batch_size)
    grad_accum = int(getattr(config, "gradient_accumulation_steps", 1))
    frame_cache_dir = os.environ.get("FRAME_CACHE_DIR")
    frame_cache_fallback_dirs = [
        str(Path(os.path.expandvars(os.path.expanduser(item))).resolve())
        for item in os.environ.get("FRAME_CACHE_FALLBACK_DIRS", "").split(os.pathsep)
        if item.strip()
    ]
    frame_cache_manifest = os.environ.get("FRAME_CACHE_MANIFEST")
    history_offsets = list(getattr(model, "front_camera_history_offsets", (-48, -32, -16, 0)))
    horizon = int(getattr(model, "action_horizon", 0))
    dataset_fps = 30.0

    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": _git_state(repo_root),
        "run": {
            "config_name": config.name,
            "experiment_name": config.exp_name,
            "resume": bool(config.resume),
            "seed": int(config.seed),
            "precision": str(config.pytorch_training_precision),
            "num_train_steps": int(config.num_train_steps),
            "save_interval": int(config.save_interval),
            "log_interval": int(config.log_interval),
        },
        "batch": {
            "world_size": int(world_size),
            "global_micro_batch": batch_size,
            "per_device_batch": batch_size // max(1, world_size),
            "gradient_accumulation_steps": grad_accum,
            "effective_global_batch": batch_size * grad_accum,
            "num_workers": int(config.num_workers),
        },
        "data": {
            "pack_inputs": _pack_inputs(getattr(data, "pack_paths", None)),
            "source_weights": getattr(data, "source_weights", None),
            "source_views": getattr(data, "source_views", None),
            "split": getattr(data, "split", None),
            "val_basis_points": getattr(data, "val_basis_points", None),
            "norm_stats": _artifact(getattr(data, "norm_stats_path", None)),
            "action_norm_stats": _artifact(getattr(data, "action_norm_stats_path", None)),
            "frame_cache": _cache_manifest(frame_cache_dir, frame_cache_manifest),
            "frame_cache_fallback_dirs": frame_cache_fallback_dirs,
            "use_frame_cache": os.environ.get("USE_FRAME_CACHE"),
            "dataset_fps": dataset_fps,
            "action_dt_seconds": 1.0 / dataset_fps,
            "history_offsets_frames": history_offsets,
            "history_offsets_seconds": [offset / dataset_fps for offset in history_offsets],
            "action_horizon": horizon,
            "action_horizon_seconds": horizon / dataset_fps,
        },
        "model": _jsonable(model),
        "initial_checkpoint": _checkpoint_identity(config.pytorch_weight_path),
        "optimizer": {
            "type": type(optimizer).__name__,
            "config": _jsonable(optimizer),
            "lr_schedule": _jsonable(lr),
            "gradient_clip_foreach": os.environ.get("PI05_GRAD_CLIP_FOREACH", "auto"),
            "adamw_foreach": os.environ.get("PI05_ADAMW_FOREACH", "auto"),
        },
        "loss_weights": {
            "subtask": float(getattr(model, "joint_loss_weight_subtask", 1.0)),
            "fast": float(getattr(model, "joint_loss_weight_fast", 1.0)),
            "flow": float(getattr(model, "joint_loss_weight_flow", 1.0)),
            "camera": float(getattr(model, "front_camera_pose_loss_weight", 0.0)),
            "camera_translation": float(getattr(model, "front_camera_pose_loss_weight_trans", 1.0)),
            "camera_rotation": float(getattr(model, "front_camera_pose_loss_weight_rot", 1.0)),
            "camera_fov": float(getattr(model, "front_camera_pose_loss_weight_focal", 1.0)),
        },
        "ddp": {
            "find_unused_parameters": os.environ.get("PI05_DDP_FIND_UNUSED_PARAMETERS", "1"),
            "static_graph": os.environ.get("PI05_DDP_STATIC_GRAPH", "0"),
            "gradient_checkpointing_disabled": os.environ.get("PI05_DISABLE_GRADIENT_CHECKPOINTING", "0"),
        },
        "attention_execution": {
            "outer_flow_checkpoint_disabled": os.environ.get("PI05_DISABLE_OUTER_FLOW_CHECKPOINT", "0"),
            "outer_image_checkpoint_disabled": os.environ.get("PI05_DISABLE_OUTER_IMAGE_CHECKPOINT", "0"),
            "trailing_token_padding_trimmed": os.environ.get("PI05_TRIM_TRAILING_TOKEN_PADDING", "0"),
            "lightweight_checkpoints_disabled": os.environ.get("PI05_DISABLE_LIGHTWEIGHT_CHECKPOINTS", "0"),
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "HF_LEROBOT_HOME",
                "USE_FRAME_CACHE",
                "FRAME_CACHE_DIR",
                "FRAME_CACHE_FALLBACK_DIRS",
                "FRAME_CACHE_MANIFEST",
                "PROFILE_STEP_TIMING",
                "CUDA_VISIBLE_DEVICES",
                "SLURM_JOB_ID",
            )
        },
    }
    snapshot["contract_sha256"] = contract_sha256(snapshot)
    return snapshot


def contract_sha256(snapshot: dict[str, Any]) -> str:
    contract = {
        "config_name": snapshot["run"]["config_name"],
        "batch": snapshot["batch"],
        "data": snapshot["data"],
        "model": snapshot["model"],
        "initial_checkpoint": snapshot["initial_checkpoint"],
        "optimizer": snapshot["optimizer"],
        "loss_weights": snapshot["loss_weights"],
        "ddp": snapshot["ddp"],
        "attention_execution": snapshot["attention_execution"],
        "seed": snapshot["run"]["seed"],
    }
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def write_runtime_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(snapshot, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def update_runtime_snapshot(path: Path, updates: dict[str, Any]) -> None:
    snapshot = json.loads(path.read_text())
    snapshot.update(_jsonable(updates))
    write_runtime_snapshot(path, snapshot)
