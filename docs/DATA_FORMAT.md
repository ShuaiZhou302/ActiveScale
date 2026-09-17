# Data format

ActiveScale trains all embodiments through one 32D model action space while
preserving source-specific normalization and validity masks. A missing modality
must be masked; it must not be represented as a supervised zero target.

## Canonical action layout

The first 23 dimensions are:

| Slice | Meaning | Representation |
| --- | --- | --- |
| `0:7` | left end effector | xyz + quaternion xyzw |
| `7:14` | right end effector | xyz + quaternion xyzw |
| `14:21` | head/front camera | xyz + quaternion xyzw |
| `21:23` | left/right gripper | source-normalized scalar |
| `23:32` | padding | masked |

Human packs physically store the first 21 values as
`[head, left, right]`; the loader converts them once to
`[left, right, head]`. Piper data already use the model ordering. Public
robots without an admitted head pose supervise `0:14` and `21:23`; the
camera block is masked.

Every pose is absolute in its declared reference frame. Human trajectories are
rebased to their temporal anchor frame during conversion. Piper trajectories
remain in the shared robot unified frame. Public robot trajectories remain in
the source's declared body/base frame and receive embodiment-specific norm
statistics.

## Human Parquet packs

Each anchor row requires:

| Column | Type/shape | Meaning |
| --- | --- | --- |
| `repo_id` | string | source dataset identifier |
| `episode_index` | int | stable episode id |
| `video_relpath` | string | head-camera video path relative to `HF_LEROBOT_HOME` |
| `task` | string | clip-level subtask text |
| `und_frame_indices` | int[4] | history frame indices |
| `camera_token_history_mask` | bool[4] | valid history slots |
| `observation.state` | float[21+] | current canonical pose state |
| `observation.state_pose_valid` | bool[3] | head/left/right pose validity in packed order |
| `action` | float[50,21+] | future pose chunk |
| `action_loss_mask` | bool[50,21+] | valid supervised action elements |
| `observation.camera_extrinsics` | float[4,4,4] | `T_ref_from_camera` |
| `observation.camera_fov` | float[4,2] | vertical/horizontal FOV in radians |
| `observation.camera_fov_valid` | bool[4] | valid FOV slots |
| `observation.camera_image_hw` | float[4,2] | source image height/width |

Optional synchronized wrist columns are
`left_wrist_video_relpath`, `right_wrist_video_relpath`,
`left_wrist_frame_index`, and `right_wrist_frame_index`. Either provide all
four or none.

An `episode_annotations.parquet` sidecar supplies episode-level instruction
text. Selection manifests may point to Parquet row groups without copying the
underlying packs.

## Piper LeRobot-v3 data

The Piper adapter expects:

- `meta/info.json`, `meta/tasks.parquet`, and `meta/camera_info.json`
- `data/chunk-*/file-*.parquet`
- a JPEG cache for `cam_front`, `cam_left`, and `cam_right`

For multiple Piper roots, `ACTIVESCALE_PIPER_TASK_INDICES` uses semicolons
between roots (for example `0-25;26-34;0-25`). Episode allowlists use aligned
comma-separated slots in `ACTIVESCALE_PIPER_INCLUDED_EPISODES`, with `-` for
an unfiltered root. Exact bad anchors can be supplied as aligned JSON lists in
`ACTIVESCALE_PIPER_EXCLUDED_ANCHORS`.

Required Parquet columns are defined by
`piper_camera_token_dataset._COLUMNS`. Camera matrices are
`T_unified_from_front_camera`. The cache layout is:

```text
<cache>/<video-key>/episode_000123/frame_000456.jpg
```

## Public robot manifests

Built-in adapters support `AgiBotWorld-Beta`, `AgiBotWorld2026`, and
`RoboCOIN`. A manifest has one JSON object per episode and points to the
source Parquet/video metadata; it does not duplicate the source data. Required
fields are documented next to `_normalize_episode` in
`public_robot_cotrain_dataset.py`.

Cached frames use:

```text
<cache>/<source>/<dataset>/<camera>/episode_000123/frame_000456.jpg
```

## Normalization

Compute q01/q99 statistics independently for each dataset and embodiment.
Human stats declare `action_layout=left_right_head_pose7`. Piper stats declare
`left_pose7_right_pose7_camera_pose7_gripper2`; public robot stats declare
`left_pose7_right_pose7_gripper2`. Loaders fail closed on mismatched layouts.

## Admission rules

Before training:

1. Validate every required column, shape, and finite numeric value.
2. Unit-normalize quaternions and unwrap signs across valid future steps.
3. Verify history indices stay inside the episode; mask leading history slots.
4. Verify action chunks never cross episode or subtask boundaries.
5. Decode representative head and wrist frames from every source.
6. Project a sample of admitted EEF/camera poses back into images when
   calibration is available.
7. Run the real loader smoke test before allocating a full multi-GPU job.

Run the schema preflight before the loader smoke test:

```bash
python scripts/validate_activescale_data.py \
  --human-pack '/path/to/human/**/*.parquet' \
  --piper-root /path/to/piper

python scripts/validate_activescale_data.py \
  --public-source AgiBotWorld2026 \
  --public-manifest /path/to/agibotworld2026.jsonl
```

This command validates required metadata and columns. Shape, finite-value,
coordinate-frame, image decode, and kinematic checks remain part of the real
loader/converter admission pipeline; schema validation alone is not enough.
