#!/usr/bin/env python3
"""Validate ActiveScale data contracts without loading a model."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pyarrow.parquet as pq


HUMAN_COLUMNS = {
    "repo_id",
    "episode_index",
    "video_relpath",
    "task",
    "und_frame_indices",
    "camera_token_history_mask",
    "observation.state",
    "observation.state_pose_valid",
    "action",
    "action_loss_mask",
    "observation.camera_extrinsics",
    "observation.camera_fov",
    "observation.camera_fov_valid",
    "observation.camera_image_hw",
}
PIPER_COLUMNS = {
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
}


def _expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path) for path in glob.glob(pattern, recursive=True))
    return sorted(set(paths))


def _validate_parquet(paths: list[Path], required: set[str], label: str) -> int:
    if not paths:
        raise ValueError(f"no {label} parquet files matched")
    rows = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        missing = sorted(required - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"{path}: missing {label} columns {missing}")
        rows += parquet.metadata.num_rows
    print(f"{label}: files={len(paths)} rows={rows}")
    return rows


def _validate_piper(root: Path) -> None:
    metadata = (
        root / "meta/info.json",
        root / "meta/tasks.parquet",
        root / "meta/camera_info.json",
    )
    missing = [str(path) for path in metadata if not path.is_file()]
    if missing:
        raise ValueError(f"missing Piper metadata: {missing}")
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    _validate_parquet(files, PIPER_COLUMNS, "piper")


def _validate_manifest(path: Path, source: str) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    source_required = {
        "AgiBotWorld-Beta": {
            "task_dataset",
            "length",
            "episode_index",
            "embodiment",
            "task",
            "parquet",
            "videos",
        },
        "AgiBotWorld2026": {
            "dataset",
            "length",
            "episode_index",
            "task",
            "info",
            "parquet",
            "videos",
        },
        "RoboCOIN": {"dataset", "episode_index", "frames", "robot_type", "tasks"},
    }
    if source not in source_required:
        raise ValueError(f"unsupported source {source!r}")
    required = source_required[source]
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"{path}:{index + 1}: missing fields {missing}")
    print(f"public_robot: source={source} episodes={len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-pack", action="append", default=[], help="Parquet glob; repeatable")
    parser.add_argument("--piper-root", type=Path)
    parser.add_argument("--public-manifest", type=Path)
    parser.add_argument(
        "--public-source",
        choices=("AgiBotWorld-Beta", "AgiBotWorld2026", "RoboCOIN"),
    )
    args = parser.parse_args()
    if not (args.human_pack or args.piper_root or args.public_manifest):
        parser.error("provide at least one data input")
    if args.human_pack:
        _validate_parquet(_expand(args.human_pack), HUMAN_COLUMNS, "human")
    if args.piper_root:
        _validate_piper(args.piper_root)
    if args.public_manifest:
        if not args.public_source:
            parser.error("--public-source is required with --public-manifest")
        _validate_manifest(args.public_manifest, args.public_source)
    print("validation: PASS")


if __name__ == "__main__":
    main()
