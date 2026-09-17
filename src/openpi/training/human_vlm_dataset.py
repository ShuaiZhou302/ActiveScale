from __future__ import annotations

import bisect
from collections import Counter, OrderedDict, defaultdict, deque
import dataclasses
import glob
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch
from torch.utils.data import Dataset

from openpi.models.tokenizer import HumanVLMDiscreteTokenizer

LOGGER = logging.getLogger(__name__)

# Canonical view names. The old "subtask"/"fast" spellings remain accepted as
# aliases so existing configs keep working, but formal configs should use
# high_level / low_level: the low-level view now trains BOTH the discrete FAST
# tokens and the continuous flow action, so calling it "fast" understates it.
VIEW_HIGH_LEVEL = 0
VIEW_LOW_LEVEL = 1
VIEW_SUBTASK = VIEW_HIGH_LEVEL  # deprecated alias
VIEW_FAST = VIEW_LOW_LEVEL  # deprecated alias
IMAGE_SIZE = 224
ACTION_HORIZON = 50
HISTORY_SLOTS = 4
ACTION_DIM_21 = 21
MODEL_ACTION_DIM = 32
NORM_EPS = 1e-6


# --- Canonical action layout -------------------------------------------------
# The parquet packs are left physically unchanged and store the 21D pose vector
# as [head, left, right]. The model expects pi0.5's canonical robot ordering,
# [left, right, head], so the permutation happens at the dataset boundary and
# nowhere else. Layout names are explicit and validated: an unknown layout must
# fail closed rather than silently train on a scrambled action space.
PACKED_ACTION_LAYOUT = "head_left_right_pose7"
MODEL_ACTION_LAYOUT = "left_right_head_pose7"
POSE7 = 7

# Source block start offsets in the packed layout, in model-layout order.
_PACKED_TO_MODEL_BLOCKS = (POSE7, 2 * POSE7, 0)  # left, right, head
_PACKED_TO_MODEL_POSE_PARTS = (1, 2, 0)  # left, right, head


def permute_packed_to_model(array: np.ndarray) -> np.ndarray:
    """Reorder the leading 21 dims from [head, left, right] to [left, right, head].

    Operates on the last axis and preserves anything beyond dim 21 (padding),
    so it is safe for state [D], action [T, D] and boolean masks alike.
    """
    array = np.asarray(array)
    if array.shape[-1] < ACTION_DIM_21:
        raise ValueError(f"expected at least {ACTION_DIM_21} action dims, got {array.shape[-1]}")
    out = array.copy()
    for out_block, in_start in enumerate(_PACKED_TO_MODEL_BLOCKS):
        out_start = out_block * POSE7
        out[..., out_start : out_start + POSE7] = array[..., in_start : in_start + POSE7]
    return out


def permute_packed_pose_valid_to_model(array: np.ndarray) -> np.ndarray:
    """Reorder pose-part validity from packed [head,left,right] to model [left,right,head]."""
    array = np.asarray(array, dtype=np.bool_).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"expected 3 pose-part validity flags, got {array.shape}")
    return array[np.asarray(_PACKED_TO_MODEL_POSE_PARTS, dtype=np.int64)]


def zero_invalid_pose_blocks(array: np.ndarray, pose_valid: np.ndarray) -> np.ndarray:
    """Zero invalid 7D pose blocks so placeholders cannot condition either expert."""
    out = np.asarray(array, dtype=np.float32).copy()
    pose_valid = np.asarray(pose_valid, dtype=np.bool_).reshape(-1)
    if pose_valid.shape != (3,):
        raise ValueError(f"expected 3 pose-part validity flags, got {pose_valid.shape}")
    for block, valid in enumerate(pose_valid):
        if not valid:
            out[..., block * POSE7 : (block + 1) * POSE7] = 0.0
    return out


QUAT_IN_POSE7 = slice(3, 7)  # xyzw within a pose7 block
_POSE_BLOCK_STARTS = (0, POSE7, 2 * POSE7)  # left, right, head in model layout


def unit_normalize_quaternions(array: np.ndarray) -> np.ndarray:
    """Normalize the quaternion of every pose7 block in a canonical vector."""
    out = np.asarray(array, dtype=np.float32).copy()
    for start in _POSE_BLOCK_STARTS:
        q = out[..., start + QUAT_IN_POSE7.start : start + QUAT_IN_POSE7.stop]
        norm = np.linalg.norm(q, axis=-1, keepdims=True)
        out[..., start + QUAT_IN_POSE7.start : start + QUAT_IN_POSE7.stop] = np.divide(
            q, norm, out=np.zeros_like(q), where=norm > NORM_EPS
        )
    return out


def unwrap_action_quaternion_signs(
    actions: np.ndarray, state: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """Make the future-action quaternion sign continuous along VALID time.

    q and -q are the same rotation, so a sign flip between consecutive targets
    is a discontinuity the regression head cannot represent smoothly. The
    packs enforce per-frame qw>=0, which is deterministic but NOT temporally
    continuous: a rotation passing through qw=0 flips sign.

    Anchoring, per pose block on the canonical [left, right, head] layout:
    the first VALID future quaternion aligns to the corresponding
    current-state quaternion, and every later VALID quaternion aligns to the
    previous VALID one. Invalid timesteps neither update the reference nor
    affect the sign of anything after them -- letting them participate is what
    left a residual flip rate on sources with mid-sequence mask gaps.

    qw<0 is allowed and expected after unwrapping: temporal continuity and a
    global qw>=0 convention cannot both hold.
    """
    actions = unit_normalize_quaternions(actions)
    state = unit_normalize_quaternions(np.asarray(state, dtype=np.float32))
    mask_arr = None if mask is None else np.asarray(mask, dtype=bool)
    for start in _POSE_BLOCK_STARTS:
        lo, hi = start + QUAT_IN_POSE7.start, start + QUAT_IN_POSE7.stop
        if mask_arr is None:
            valid = np.ones(actions.shape[0], dtype=bool)
        else:
            valid = mask_arr[:, lo:hi].all(axis=-1)
        reference = state[lo:hi]
        for t in range(actions.shape[0]):
            if not valid[t]:
                continue
            if float(np.dot(actions[t, lo:hi], reference)) < 0.0:
                actions[t, lo:hi] = -actions[t, lo:hi]
            reference = actions[t, lo:hi]
    return actions


def validate_action_layouts(packed_layout: str, model_layout: str) -> None:
    """Fail closed on any layout pair this code does not implement."""
    if packed_layout != PACKED_ACTION_LAYOUT or model_layout != MODEL_ACTION_LAYOUT:
        raise ValueError(
            "unsupported action layout conversion "
            f"{packed_layout!r} -> {model_layout!r}; this code implements only "
            f"{PACKED_ACTION_LAYOUT!r} -> {MODEL_ACTION_LAYOUT!r}"
        )


def _view_name(view_type: int) -> str:
    if view_type == VIEW_HIGH_LEVEL:
        return "high_level"
    if view_type == VIEW_LOW_LEVEL:
        return "low_level"
    return f"unknown_{view_type}"


_VIEW_BY_NAME = {
    "high_level": VIEW_HIGH_LEVEL,
    "low_level": VIEW_LOW_LEVEL,
    # Deprecated aliases kept so existing configs and reports still parse.
    "subtask": VIEW_HIGH_LEVEL,
    "fast": VIEW_LOW_LEVEL,
}

# Stable integer ids so the trainer can aggregate per-source metrics across
# DataLoader workers and DDP ranks, where python strings cannot travel.
SOURCE_IDS = {
    "EgoDex": 0,
    "EgoLive": 1,
    "VITRA": 2,
    "EgoVerse": 3,
    "GenRobot": 4,
    "Piper": 5,
    "AgiBotWorld-Beta": 6,
    "RoboCOIN": 7,
    "EgoProStandard": 8,
    "AgiBotWorld2026": 9,
}
UNKNOWN_SOURCE_ID = 99


def source_id_for(source: str) -> int:
    return SOURCE_IDS.get(source, UNKNOWN_SOURCE_ID)


def parse_source_views(spec: str | None) -> dict[str, dict[int, float]]:
    """Parse per-source view specs into `{source: {view: weight}}`.

    Accepts `"EgoDex=subtask+fast,VITRA=fast"` (equal weights) and
    `"EgoDex=subtask:3+fast:1"` (explicit weights). Weights are normalized
    within each source.

    Sources differ in which supervision they can support: a source without a
    genuine episode-level instruction cannot train the subtask view, while a
    source with actions can always train the FAST view. Listing them
    explicitly avoids silently generating samples a source cannot satisfy.
    """
    if not spec or not spec.strip():
        return {}
    out: dict[str, dict[int, float]] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()  # noqa: PLW2901
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"source_views entry must be '<source>=<view>[:<weight>][+...]', got {chunk!r}")
        source, views_spec = chunk.split("=", 1)
        weights: dict[int, float] = {}
        for item in views_spec.split("+"):
            item = item.strip()  # noqa: PLW2901
            if not item:
                continue
            name, _, weight_text = item.partition(":")
            name = name.strip().lower()  # noqa: PLW2901
            if name not in _VIEW_BY_NAME:
                raise ValueError(f"unknown view {name!r}; expected one of {sorted(_VIEW_BY_NAME)}")
            weight = float(weight_text) if weight_text.strip() else 1.0
            if weight < 0:
                raise ValueError(f"view weight must be non-negative, got {weight}")
            weights[_VIEW_BY_NAME[name]] = weight
        total = sum(weights.values())
        out[source.strip()] = {v: w / total for v, w in weights.items() if w > 0} if total > 0 else {}
    return out


def splitmix64(x: int) -> int:
    """Stable integer mix. Deterministic across processes, runs and versions.

    `hash()` is unsuitable: it is salted for str/bytes, and even for int its
    contract is not a stability guarantee across Python versions. View routing
    must reproduce exactly on resume and on every DataLoader worker, so the
    mixing function is written out explicitly.
    """
    mask = 0xFFFFFFFFFFFFFFFF
    x = (x + 0x9E3779B97F4A7C15) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    return (x ^ (x >> 31)) & mask


VAL_SPLIT_SEED = 0x5EED5017
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_ALL = "all"


def episode_split_bucket(repo_id: str, episode_index: int, *, seed: int = VAL_SPLIT_SEED) -> int:
    """Deterministic 0..9999 bucket for an episode.

    Keyed on `(repo_id, episode_index)` -- the same join key the sidecar uses
    -- so every anchor of an episode lands in the same bucket and no frame of
    a validation episode can leak into training. Hashing the episode rather
    than the anchor is the whole point: an anchor-level split would put
    neighbouring frames of one episode on both sides.
    """
    digest = hashlib.sha1(f"{repo_id}\x00{int(episode_index)}".encode()).digest()
    return (int.from_bytes(digest[:8], "big") ^ seed) % 10000


def episode_in_split(repo_id: str, episode_index: int, split: str, val_bp: int) -> bool:
    """Is this episode part of `split`? `val_bp` is basis points (100 = 1%)."""
    if split == SPLIT_ALL:
        return True
    is_val = episode_split_bucket(repo_id, episode_index) < val_bp
    return is_val if split == SPLIT_VAL else not is_val


def build_view_schedule(weights: dict[int, float]) -> tuple[int, ...]:
    """Turn per-view weights into a fixed-length rotation schedule.

    `{high: 1, low: 1}` -> `(high, low)`; `{high: 3, low: 1}` ->
    `(high, high, high, low)`; `{low: 1}` -> `(low,)`.

    An anchor walks this schedule as the cycle advances, so it covers every one
    of its views within `len(schedule)` cycles and its long-run view frequency
    is exactly the configured ratio -- neither of which holds for a static
    partition of the anchor space.
    """
    views = sorted(w for w in weights if weights[w] > 0)
    if not views:
        return ()
    total = sum(weights[v] for v in views)
    fractions = [weights[v] / total for v in views]
    # Smallest span that realizes the ratio exactly for the common cases
    # (1:1, 3:1, 1:0); falls back to largest-remainder for anything else.
    for span in range(len(views), 25):
        counts = _allocate_shares(span, fractions)
        if sum(counts) == span and all(
            abs(counts[i] / span - fractions[i]) < 1e-9 for i in range(len(views))
        ):
            break
    else:
        span = len(views)
        counts = _allocate_shares(span, fractions)
    schedule: list[int] = []
    for view, count in zip(views, counts, strict=True):
        schedule.extend([view] * count)
    return tuple(schedule)


def _allocate_shares(total: int, weights: list[float]) -> list[int]:
    """Split `total` items across `weights` using largest-remainder.

    Every positively weighted entry gets at least one item when `total` is
    large enough to go around, so a configured view can never silently vanish
    (plain rounding drops a 50/50 split of a single item, since Python rounds
    0.5 to 0).
    """
    n = len(weights)
    if n == 0 or total <= 0:
        return [0] * n
    if total < n:
        # Not enough to give everyone one: hand them out by descending weight.
        order = sorted(range(n), key=lambda i: (-weights[i], i))
        shares = [0] * n
        for i in order[:total]:
            shares[i] = 1
        return shares
    exact = [total * w for w in weights]
    shares = [max(1, int(f)) for f in exact]
    while sum(shares) > total:
        i = max(
            (i for i in range(n) if shares[i] > 1),
            key=lambda i: shares[i] - exact[i],
            default=None,
        )
        if i is None:
            break
        shares[i] -= 1
    remainders = sorted(range(n), key=lambda i: (exact[i] - shares[i]), reverse=True)
    idx = 0
    while sum(shares) < total:
        shares[remainders[idx % n]] += 1
        idx += 1
    return shares


def _as_paths(spec: str | Path) -> list[Path]:
    paths: list[Path] = []
    for part in str(spec).split(","):
        part = os.path.expanduser(os.path.expandvars(part.strip()))
        if not part:
            continue
        path = Path(part)
        if path.is_file() and path.suffix == ".json":
            payload = json.loads(path.read_text())
            manifest = payload.get("parquets")
            if isinstance(manifest, list):
                for entry in manifest:
                    resolved = Path(os.path.expanduser(os.path.expandvars(str(entry))))
                    paths.append(resolved if resolved.is_absolute() else path.parent / resolved)
                continue
        if path.is_file() and path.suffix in {".txt", ".lst"}:
            listed = [
                line.strip()
                for line in path.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            if listed:
                paths.extend(Path(os.path.expanduser(os.path.expandvars(p))) for p in listed)
                continue
        expanded = sorted(Path(p) for p in glob.glob(part))
        paths.extend(expanded or [Path(part)])
    return [p for p in paths if p.name != "episode_annotations.parquet"]


def _source_key(path: Path) -> str:
    for part in path.parts:
        if part.startswith("EgoDex"):
            return "EgoDex"
        if part.startswith("EgoLive"):
            return "EgoLive"
        if part.startswith("EgoVerse"):
            return "EgoVerse"
        if part.startswith("GenRobot"):
            return "GenRobot"
        if part.startswith("VITRA"):
            return "VITRA"
        if part.startswith("EgoProStandard"):
            return "EgoProStandard"
    return path.parent.name


def _normalized_text(text: str) -> str:
    return " ".join(str(text).lower().replace("_", " ").split())


def parse_source_weights(source_weights: dict[str, float] | str | None) -> dict[str, float]:
    if not source_weights:
        return {}
    if isinstance(source_weights, dict):
        raw_items = source_weights.items()
    else:
        raw_items = []
        for part in str(source_weights).split(","):
            part = part.strip()
            if not part:
                continue
            sep = "=" if "=" in part else ":"
            if sep not in part:
                raise ValueError(f"Invalid source weight entry {part!r}; expected Source=weight")
            key, value = part.split(sep, 1)
            raw_items.append((key.strip(), value.strip()))
    cleaned = {str(k): float(v) for k, v in raw_items if float(v) > 0}
    total = sum(cleaned.values())
    return {k: v / total for k, v in cleaned.items()} if total > 0 else {}


def _pack_root_for(path: Path) -> Path:
    current = path if path.is_dir() else path.parent
    for parent in (current, *current.parents):
        if (parent / "episode_annotations.parquet").is_file():
            return parent
    return current


def _frame_cache_roots() -> tuple[Path, ...]:
    raw_roots = [os.environ.get("FRAME_CACHE_DIR", "")]
    raw_roots.extend(os.environ.get("FRAME_CACHE_FALLBACK_DIRS", "").split(os.pathsep))
    roots = tuple(Path(raw.strip()) for raw in raw_roots if raw.strip())
    if roots:
        return roots
    if os.environ.get("USE_FRAME_CACHE", "").lower() in {"1", "true", "yes"}:
        raise RuntimeError(
            "USE_FRAME_CACHE is set but FRAME_CACHE_DIR is missing. Human VLM training must pin an explicit "
            "cache root because historical .frame_cache/v2 directories may mix dataset-frame and source-frame "
            "indexing semantics."
        )
    return ()


def _frame_cache_dir_for(video_path: Path, cache_root: Path) -> Path:
    key = hashlib.sha1(os.path.realpath(str(video_path)).encode()).hexdigest()[:16]
    return cache_root / key


def _pad_last_dim(array: np.ndarray, dim: int, *, value: float | bool = 0) -> np.ndarray:
    if array.shape[-1] >= dim:
        return array[..., :dim]
    pad_width = [(0, 0)] * array.ndim
    pad_width[-1] = (0, dim - array.shape[-1])
    return np.pad(array, pad_width, constant_values=value)


@dataclasses.dataclass(frozen=True)
class HumanVLMNormStats:
    state_q01: np.ndarray
    state_q99: np.ndarray
    action_q01: np.ndarray | None = None
    action_q99: np.ndarray | None = None


def _load_human_vlm_norm(path: str | None) -> HumanVLMNormStats | None:
    if not path:
        return None
    path_obj = Path(os.path.expanduser(os.path.expandvars(path)))
    payload = json.loads(path_obj.read_text())
    if "state_q01" not in payload or "state_q99" not in payload:
        raise ValueError(f"Human VLM norm stats must contain state_q01/state_q99: {path_obj}")
    # Stats are per-dimension, so they are only valid for the layout they were
    # computed in. Refuse stats that predate the canonical reordering instead
    # of silently normalizing left-wrist values with head-pose quantiles.
    stats_layout = payload.get("action_layout")
    if stats_layout != MODEL_ACTION_LAYOUT:
        raise ValueError(
            f"norm stats {path_obj} declare action_layout={stats_layout!r} but the dataset emits "
            f"{MODEL_ACTION_LAYOUT!r}. Recompute the stats with compute_human_action_norm_stats.py."
        )
    state_q01 = np.asarray(payload["state_q01"], dtype=np.float32)[:ACTION_DIM_21]
    state_q99 = np.asarray(payload["state_q99"], dtype=np.float32)[:ACTION_DIM_21]
    if state_q01.shape != (ACTION_DIM_21,) or state_q99.shape != (ACTION_DIM_21,):
        raise ValueError(f"State norm stats must contain at least 21 dims: {path_obj}")
    if not np.isfinite(state_q01).all() or not np.isfinite(state_q99).all():
        raise ValueError(f"State norm stats contain non-finite values: {path_obj}")

    action_q01 = payload.get("action_q01", payload.get("q01"))
    action_q99 = payload.get("action_q99", payload.get("q99"))
    if action_q01 is None or action_q99 is None:
        return HumanVLMNormStats(state_q01=state_q01, state_q99=state_q99)
    action_q01_arr = np.asarray(action_q01, dtype=np.float32)[:ACTION_DIM_21]
    action_q99_arr = np.asarray(action_q99, dtype=np.float32)[:ACTION_DIM_21]
    if action_q01_arr.shape != (ACTION_DIM_21,) or action_q99_arr.shape != (ACTION_DIM_21,):
        raise ValueError(f"Action norm stats must contain at least 21 dims: {path}")
    if not np.isfinite(action_q01_arr).all() or not np.isfinite(action_q99_arr).all():
        raise ValueError(f"Action norm stats contain non-finite values: {path}")
    return HumanVLMNormStats(
        state_q01=state_q01,
        state_q99=state_q99,
        action_q01=action_q01_arr,
        action_q99=action_q99_arr,
    )


def _load_source_norms(spec: str | None) -> dict[str, HumanVLMNormStats]:
    """Parse a comma-separated SOURCE=PATH map into validated norm stats."""
    if not spec or not spec.strip():
        return {}
    result: dict[str, HumanVLMNormStats] = {}
    for item in spec.split(","):
        source, separator, path = item.strip().partition("=")
        if not separator or not source.strip() or not path.strip():
            raise ValueError(f"source norm entry must be SOURCE=PATH, got {item!r}")
        source = source.strip()
        if source in result:
            raise ValueError(f"duplicate source norm entry for {source!r}")
        stats = _load_human_vlm_norm(path.strip())
        if stats is None:
            raise ValueError(f"source norm path is empty for {source!r}")
        result[source] = stats
    return result


def _normalize_q01_q99(values: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    q01 = np.asarray(q01, dtype=np.float32)
    q99 = np.asarray(q99, dtype=np.float32)
    span = q99 - q01
    valid_span = np.isfinite(span) & (np.abs(span) > NORM_EPS)
    midpoint = (q01 + q99) * 0.5
    finite_values = np.where(np.isfinite(values), values, midpoint)
    normalized = np.zeros_like(finite_values, dtype=np.float32)
    np.divide(
        (finite_values - q01) * 2.0,
        span,
        out=normalized,
        where=valid_span,
    )
    normalized -= np.where(valid_span, 1.0, 0.0).astype(np.float32)
    normalized = np.where(np.isfinite(normalized), normalized, 0.0)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


class HumanVLMPairedOfflineDataset(Dataset):
    """Offline parquet dataset for π0.5 human VLM discrete pretraining.

    Each anchor row is expanded virtually into up to two examples:
    subtask prediction and FAST action prediction. Pack rows remain the source
    of the current subtask (`task`). Episode sidecars provide the high-level
    instruction and are joined by repo_id + episode_index.
    """

    is_human_vlm_offline_pack = True
    _RUNTIME_PACK_COLUMNS = (
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
    )
    _OPTIONAL_WRIST_COLUMNS = (
        "left_wrist_video_relpath",
        "right_wrist_video_relpath",
        "left_wrist_frame_index",
        "right_wrist_frame_index",
    )
    _REQUIRED_PACK_COLUMNS = frozenset(_RUNTIME_PACK_COLUMNS)

    def __init__(
        self,
        *,
        pack_paths: str | Path,
        lerobot_home: str | Path | None = None,
        source_weights: dict[str, float] | str | None = None,
        max_token_len: int,
        fast_tokenizer_path: str = "physical-intelligence/fast",
        enable_subtask_view: bool = True,
        enable_fast_view: bool = True,
        source_views: str | None = None,
        fast_horizon: int = ACTION_HORIZON,
        allow_partial_fast_horizon: bool = False,
        action_norm_stats_path: str | None = None,
        norm_stats_path: str | None = None,
        source_norm_stats_paths: str | None = None,
        action_loss_weight: float = 1.0,
        fake_images: bool = False,
        row_group_cache_size: int = 4,
        max_filter_attempts: int = 64,
        split: str = SPLIT_ALL,
        val_basis_points: int = 200,
        selection_manifest_path: str | Path | None = None,
        view_cycle_mode: str = "complete_view_cycle",
        view_route_phase: int = 0,
    ) -> None:
        super().__init__()
        if split not in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_ALL):
            raise ValueError(f"split must be train/val/all, got {split!r}")
        self.split = split
        self.val_basis_points = int(val_basis_points)
        if view_cycle_mode not in ("complete_view_cycle", "physical_once"):
            raise ValueError(
                "view_cycle_mode must be complete_view_cycle or physical_once, "
                f"got {view_cycle_mode!r}"
            )
        self.view_cycle_mode = view_cycle_mode
        self.view_route_phase = int(view_route_phase)
        self.pack_paths = _as_paths(pack_paths)
        if not self.pack_paths:
            raise ValueError("HumanVLMPairedOfflineDataset requires at least one parquet pack path")
        missing = [str(p) for p in self.pack_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing parquet paths: {missing[:5]}")
        self.fast_horizon = int(fast_horizon)
        if self.fast_horizon <= 0 or self.fast_horizon > ACTION_HORIZON:
            raise ValueError(f"fast_horizon must be in [1,{ACTION_HORIZON}], got {fast_horizon}")
        self.allow_partial_fast_horizon = bool(allow_partial_fast_horizon)
        self.fake_images = bool(fake_images or os.environ.get("OPENPI_HUMAN_VLM_FAKE_IMAGES", "").lower() in {"1", "true", "yes"})
        lerobot_home_raw = str(lerobot_home or os.environ.get("HF_LEROBOT_HOME", "")).strip()
        if not lerobot_home_raw and not self.fake_images:
            raise ValueError("lerobot_home or HF_LEROBOT_HOME must be set when fake_images=False")
        self.lerobot_home = Path(os.path.expanduser(os.path.expandvars(lerobot_home_raw))) if lerobot_home_raw else Path(".")
        self._row_group_cache_size = max(1, int(row_group_cache_size))
        self._row_group_cache: OrderedDict[tuple[int, int], Any] = OrderedDict()
        self._frame_cache_meta: dict[Path, dict[str, Any] | None] = {}
        self._frame_cache_dir_by_video: dict[Path, Path] = {}
        self._decode_error_counts: defaultdict[str, int] = defaultdict(int)
        self._decode_error_total = 0
        self._resample_total = 0
        self._requested_by_source: Counter[str] = Counter()
        self._returned_by_source: Counter[str] = Counter()
        self._requested_by_view: Counter[str] = Counter()
        self._returned_by_view: Counter[str] = Counter()
        self._requested_by_source_view: Counter[str] = Counter()
        self._returned_by_source_view: Counter[str] = Counter()
        self._invalid_by_source_view_reason: Counter[str] = Counter()
        self._retry_by_source_view: Counter[str] = Counter()
        self._recent_bad_samples: deque[str] = deque(maxlen=16)
        self._max_filter_attempts = max(1, int(max_filter_attempts))
        self._norm_stats = _load_human_vlm_norm(norm_stats_path or action_norm_stats_path)
        self._norm_stats_by_source = _load_source_norms(source_norm_stats_paths)
        if self._norm_stats is None and not self._norm_stats_by_source:
            raise ValueError("Human VLM dataset requires shared or source-specific state/action norm stats")
        self.action_loss_weight = float(action_loss_weight)
        if not math.isfinite(self.action_loss_weight) or self.action_loss_weight < 0:
            raise ValueError(f"action_loss_weight must be finite and non-negative, got {action_loss_weight}")
        self.views = []
        if enable_subtask_view:
            self.views.append(VIEW_SUBTASK)
        if enable_fast_view:
            self.views.append(VIEW_FAST)
        if not self.views:
            raise ValueError("At least one human VLM view must be enabled")
        # Per-source overrides, intersected with the globally enabled views so
        # a source can never request a view the run has switched off.
        self._source_view_overrides = {
            source: {v: w for v, w in weights.items() if v in self.views}
            for source, weights in parse_source_views(source_views).items()
        }
        self.tokenizer = HumanVLMDiscreteTokenizer(
            max_len=max_token_len,
            fast_tokenizer_path=fast_tokenizer_path if enable_fast_view else None,
        )

        # Harvest per-file row-group sizes once (open/close each file), then
        # open ParquetFile handles lazily through a small per-process LRU so
        # workers never need thousands of open descriptors.
        self._row_group_sizes: list[list[int]] = []
        self._read_columns_by_file: list[list[str]] = []
        for path in self.pack_paths:
            pf = pq.ParquetFile(path)
            self._row_group_sizes.append(
                [pf.metadata.row_group(rg).num_rows for rg in range(pf.num_row_groups)]
            )
            schema_names = set(pf.schema_arrow.names)
            missing_columns = sorted(self._REQUIRED_PACK_COLUMNS - schema_names)
            if missing_columns:
                pf.close()
                raise ValueError(f"Training pack {path} is missing required columns: {missing_columns}")
            present_wrist_columns = schema_names.intersection(self._OPTIONAL_WRIST_COLUMNS)
            if present_wrist_columns and present_wrist_columns != set(self._OPTIONAL_WRIST_COLUMNS):
                pf.close()
                missing_wrist_columns = sorted(set(self._OPTIONAL_WRIST_COLUMNS) - schema_names)
                raise ValueError(
                    f"Training pack {path} has an incomplete optional wrist-image schema; "
                    f"missing {missing_wrist_columns}"
                )
            self._read_columns_by_file.append(
                [
                    name
                    for name in (*self._RUNTIME_PACK_COLUMNS, *self._OPTIONAL_WRIST_COLUMNS)
                    if name in schema_names
                ]
            )
            pf.close()
        admitted_sources = {_source_key(path) for path in self.pack_paths}
        missing_norms = sorted(
            source
            for source in admitted_sources
            if source not in self._norm_stats_by_source and self._norm_stats is None
        )
        if missing_norms:
            raise ValueError(f"No norm stats resolved for admitted sources: {missing_norms}")
        if enable_fast_view:
            for source in sorted(admitted_sources):
                stats = self._norm_stats_by_source.get(source, self._norm_stats)
                if stats is None or stats.action_q01 is None or stats.action_q99 is None:
                    raise ValueError(f"FAST view requires action q01/q99 norm stats for source {source}")
        self._parquet_handle_limit = max(4, int(os.environ.get("HUMAN_VLM_PARQUET_HANDLE_LIMIT", "64")))
        self._parquet_handles: OrderedDict[int, pq.ParquetFile] = OrderedDict()
        self._parquet_handles_pid = os.getpid()
        self._sidecars = self._load_sidecars()
        self._source_weights = parse_source_weights(source_weights)
        self._selection_groups = self._load_selection_groups(selection_manifest_path)
        self._groups: list[tuple[int, int, int, int, int]] = self._build_groups()
        total = 0
        self._cum_rows: list[int] = []
        self._anchor_ranges_by_source: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._source_global_starts: dict[str, list[int]] = {}
        self._source_cum_rows: dict[str, list[int]] = {}
        for _file_idx, _rg_idx, _start, n, _stride in self._groups:
            source = _source_key(self.pack_paths[_file_idx])
            self._anchor_ranges_by_source[source].append((total, n))
            total += n
            self._cum_rows.append(total)
        for source, ranges in self._anchor_ranges_by_source.items():
            self._source_global_starts[source] = [global_start for global_start, _length in ranges]
            cumsum: list[int] = []
            subtotal = 0
            for _global_start, n in ranges:
                subtotal += n
                cumsum.append(subtotal)
            self._source_cum_rows[source] = cumsum
        self.num_anchor_rows = total
        # View routing is CYCLE-DEPENDENT, not a static partition of anchors.
        #
        # The previous layout split each source's anchors into disjoint
        # (source, view) blocks, which permanently locked anchor i to one view:
        # the first 75% of EgoDex anchors could only ever train high-level and
        # the rest only low-level. Over long training that means no eligible
        # anchor ever supervises both objectives.
        #
        # Instead the index space is `num_anchor_rows * view_cycle_span`, and
        # the cycle is carried by the index itself:
        #
        #     cycle = idx // num_anchor_rows
        #     slot  = idx %  num_anchor_rows
        #     view  = schedule[source][(mix(slot, seed) + cycle) % len(schedule)]
        #
        # Deriving the view cycle from the index rather than from sampler epoch
        # is deliberate: `persistent_workers=True` means a mutated dataset
        # attribute never reaches worker processes. `set_epoch` is still used by
        # the distributed sampler to reshuffle the virtual indices between
        # passes, but it must not change the routing rule for a particular
        # virtual index. This keeps source/view routing resume-safe while sampler
        # epoch controls ordering only.
        self._views_by_source: dict[str, list[int]] = {}
        self._view_weights_by_source: dict[str, dict[int, float]] = {}
        self._view_schedule_by_source: dict[str, tuple[int, ...]] = {}
        for source in sorted(self._source_cum_rows):
            weights = self._source_view_overrides.get(source)
            if weights is None:
                weights = {v: 1.0 / len(self.views) for v in self.views}
            views = [v for v, w in weights.items() if w > 0]
            self._views_by_source[source] = views
            self._view_weights_by_source[source] = dict(weights)
            if not views:
                LOGGER.warning(
                    "[human_vlm_dataset] source %s has no enabled view and is excluded from sampling", source
                )
                continue
            self._view_schedule_by_source[source] = build_view_schedule({v: weights[v] for v in views})
        if not self._view_schedule_by_source:
            raise ValueError("No source has an enabled view; check enable_*_view and source_views")
        # Complete cycles expose every requested objective for every anchor.
        # Short cotrain runs can instead choose one deterministic objective per
        # physical anchor and use view_route_phase for a complementary pass.
        complete_span = math.lcm(*(len(s) for s in self._view_schedule_by_source.values()))
        self.view_cycle_span = 1 if self.view_cycle_mode == "physical_once" else complete_span
        self._view_route_seed = int(os.environ.get("HUMAN_VLM_VIEW_ROUTE_SEED", "20260727"))
        self.num_frames = self.num_anchor_rows * self.view_cycle_span
        LOGGER.info(
            "[human_vlm_dataset] paths=%d anchors=%d virtual_examples=%d span=%d "
            "mode=%s phase=%d schedules=%s source_weights=%s route_seed=%d",
            len(self.pack_paths),
            self.num_anchor_rows,
            self.num_frames,
            self.view_cycle_span,
            self.view_cycle_mode,
            self.view_route_phase,
            {s: [_view_name(v) for v in sch] for s, sch in self._view_schedule_by_source.items()},
            self._source_weights,
            self._view_route_seed,
        )

    def _load_selection_groups(
        self, selection_manifest_path: str | Path | None
    ) -> list[tuple[int, int, int, int, int]] | None:
        if selection_manifest_path is None:
            return None
        if self._source_weights:
            raise ValueError("source_weights must be unset when a human row-selection manifest is used")
        manifest_path = Path(os.path.expandvars(os.path.expanduser(str(selection_manifest_path))))
        path_to_index = {str(path): index for index, path in enumerate(self.pack_paths)}
        groups = []
        intervals: defaultdict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
        for line_number, line in enumerate(manifest_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            path = str(Path(os.path.expandvars(os.path.expanduser(str(row["path"])))))
            if path not in path_to_index:
                raise ValueError(f"Selection row {line_number} references an unadmitted parquet: {path}")
            file_idx = path_to_index[path]
            expected_source = _source_key(self.pack_paths[file_idx])
            if str(row.get("source")) != expected_source:
                raise ValueError(
                    f"Selection row {line_number} source={row.get('source')!r}, expected {expected_source!r}"
                )
            row_group = int(row["row_group"])
            row_start = int(row["row_start"])
            raw_rows = int(row["num_rows"])
            stride = int(row.get("anchor_stride", 1))
            if row_group < 0 or row_group >= len(self._row_group_sizes[file_idx]):
                raise ValueError(f"Selection row {line_number} has invalid row_group={row_group}")
            row_group_size = self._row_group_sizes[file_idx][row_group]
            if row_start < 0 or raw_rows <= 0 or stride <= 0 or row_start + raw_rows > row_group_size:
                raise ValueError(f"Selection row {line_number} is outside row-group bounds")
            sample_count = (raw_rows + stride - 1) // stride
            groups.append((file_idx, row_group, row_start, sample_count, stride))
            intervals[(file_idx, row_group)].append((row_start, row_start + raw_rows, line_number))
        if not groups:
            raise ValueError(f"Human row-selection manifest is empty: {manifest_path}")
        for key, ranges in intervals.items():
            ranges.sort()
            for previous, current in zip(ranges, ranges[1:], strict=False):
                if current[0] < previous[1]:
                    raise ValueError(
                        f"Human row-selection ranges overlap for file/row-group {key}: "
                        f"lines {previous[2]} and {current[2]}"
                    )
        return groups

    def _build_groups(self) -> list[tuple[int, int, int, int, int]]:
        if self._selection_groups is not None:
            return self._selection_groups
        source_units: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
        for file_idx, rg_sizes in enumerate(self._row_group_sizes):
            source = _source_key(self.pack_paths[file_idx])
            for rg_idx, n in enumerate(rg_sizes):
                if n > 0:
                    source_units[source].append((file_idx, rg_idx, n))
        if not self._source_weights:
            return [
                (file_idx, rg_idx, 0, n, 1)
                for units in source_units.values()
                for file_idx, rg_idx, n in units
            ]

        present = {s: self._source_weights.get(s, 0.0) for s in source_units}
        missing = [s for s, w in present.items() if w <= 0]
        if missing:
            raise ValueError(f"source_weights missing present source(s): {missing}; weights={self._source_weights}")
        weights = {s: w / sum(present.values()) for s, w in present.items()}
        rows_by_source = {s: sum(n for *_rest, n in units) for s, units in source_units.items()}
        virtual_total = max(rows_by_source[s] / weights[s] for s in rows_by_source)
        target_rows = {s: int(round(virtual_total * weights[s])) for s in rows_by_source}
        seed = int(os.environ.get("HUMAN_VLM_PACK_SHUFFLE_SEED", "42"))
        chunk_size = max(1, int(os.environ.get("HUMAN_VLM_PACK_GROUP_CHUNK_SIZE", "1024")))
        rng = random.Random(seed)
        chunks_by_source: dict[str, deque[tuple[int, int, int, int, int]]] = {}
        for source, units in source_units.items():
            chunks: list[tuple[int, int, int, int, int]] = []
            shuffled = list(units)
            rng.shuffle(shuffled)
            for file_idx, rg_idx, n in shuffled:
                for start in range(0, n, chunk_size):
                    chunks.append((file_idx, rg_idx, start, min(chunk_size, n - start), 1))
            expanded: deque[tuple[int, int, int, int, int]] = deque()
            emitted = 0
            while emitted < target_rows[source]:
                for chunk in chunks:
                    remaining = target_rows[source] - emitted
                    if remaining <= 0:
                        break
                    take = min(chunk[3], remaining)
                    expanded.append((chunk[0], chunk[1], chunk[2], take, 1))
                    emitted += take
                rng.shuffle(chunks)
            chunks_by_source[source] = expanded
        emitted = {s: 0 for s in chunks_by_source}
        groups: list[tuple[int, int, int, int, int]] = []
        while any(chunks_by_source.values()):
            available = [s for s, chunks in chunks_by_source.items() if chunks]
            source = min(available, key=lambda s: emitted[s] / target_rows[s])
            chunk = chunks_by_source[source].popleft()
            groups.append(chunk)
            emitted[source] += chunk[3]
        LOGGER.info(
            "[human_vlm_dataset] source_rows=%s target_rows=%s effective_weights=%s",
            rows_by_source,
            target_rows,
            {s: target_rows[s] / sum(target_rows.values()) for s in target_rows},
        )
        return groups

    def _load_sidecars(self) -> dict[Path, dict[tuple[str, int], dict[str, Any]]]:
        sidecars: dict[Path, dict[tuple[str, int], dict[str, Any]]] = {}
        for root in sorted({_pack_root_for(path) for path in self.pack_paths}):
            sidecar = root / "episode_annotations.parquet"
            if not sidecar.is_file():
                LOGGER.warning("[human_vlm_dataset] no episode sidecar found for pack root %s", root)
                sidecars[root] = {}
                continue
            table = pq.read_table(sidecar)
            mapping: dict[tuple[str, int], dict[str, Any]] = {}
            duplicates = 0
            for row in table.to_pylist():
                if "episode_index" not in row:
                    raise ValueError(f"Sidecar missing episode_index: {sidecar}")
                repo_id = str(row.get("repo_id") or "*")
                key = (repo_id, int(row["episode_index"]))
                if key in mapping:
                    duplicates += 1
                mapping[key] = row
            if duplicates:
                raise ValueError(f"Sidecar duplicate repo_id+episode_index keys: {sidecar} duplicates={duplicates}")
            sidecars[root] = mapping
            LOGGER.info("[human_vlm_dataset] loaded sidecar %s rows=%d", sidecar, len(mapping))
        return sidecars

    def __len__(self) -> int:
        return self.num_frames

    def _locate_anchor(self, idx: int) -> tuple[int, int, int]:
        idx %= self.num_anchor_rows
        group_pos = bisect.bisect_right(self._cum_rows, idx)
        prev = self._cum_rows[group_pos - 1] if group_pos > 0 else 0
        file_idx, rg_idx, row_start, _n, stride = self._groups[group_pos]
        return file_idx, rg_idx, row_start + (idx - prev) * stride

    def _source_for_anchor(self, anchor_idx: int) -> str:
        file_idx, _rg_idx, _row_idx = self._locate_anchor(anchor_idx)
        return _source_key(self.pack_paths[file_idx])

    def _anchor_for_source_offset(self, source: str, offset: int) -> int:
        cumsum = self._source_cum_rows[source]
        total = cumsum[-1]
        offset %= total
        pos = bisect.bisect_right(cumsum, offset)
        prev = cumsum[pos - 1] if pos > 0 else 0
        global_start, _n = self._anchor_ranges_by_source[source][pos]
        return global_start + (offset - prev)

    def _retry_anchor_for_source(self, source: str, original_anchor_idx: int, attempt: int) -> int:
        if attempt == 0:
            return original_anchor_idx
        cumsum = self._source_cum_rows[source]
        total = cumsum[-1]
        # Deterministic per-index retry inside the originally requested source.
        offset = (original_anchor_idx * 1_103_515_245 + attempt * 7_919) % total
        return self._anchor_for_source_offset(source, offset)

    def _parquet_file(self, file_idx: int) -> pq.ParquetFile:
        # Per-process LRU: DataLoader workers fork with an inherited handle
        # cache whose descriptors must not be shared, so reset it on pid
        # change and reopen lazily inside each worker.
        if os.getpid() != self._parquet_handles_pid:
            self._parquet_handles = OrderedDict()
            self._parquet_handles_pid = os.getpid()
        handle = self._parquet_handles.get(file_idx)
        if handle is not None:
            self._parquet_handles.move_to_end(file_idx)
            return handle
        handle = pq.ParquetFile(self.pack_paths[file_idx])
        self._parquet_handles[file_idx] = handle
        while len(self._parquet_handles) > self._parquet_handle_limit:
            _idx, stale = self._parquet_handles.popitem(last=False)
            stale.close()
        return handle

    def _read_row_group(self, file_idx: int, rg_idx: int) -> Any:
        key = (file_idx, rg_idx)
        cached = self._row_group_cache.get(key)
        if cached is not None:
            self._row_group_cache.move_to_end(key)
            return cached
        # Keep the row group in Arrow form and read only runtime columns. The
        # old `.to_pydict()` path materialized huge EgoLive action columns into
        # Python lists for every cold row group, which dominated training time.
        data = self._parquet_file(file_idx).read_row_group(
            rg_idx, columns=self._read_columns_by_file[file_idx]
        )
        self._row_group_cache[key] = data
        while len(self._row_group_cache) > self._row_group_cache_size:
            self._row_group_cache.popitem(last=False)
        return data

    def _sidecar_for_row(self, file_idx: int, repo_id: str, episode_index: int) -> dict[str, Any]:
        root = _pack_root_for(self.pack_paths[file_idx])
        mapping = self._sidecars.get(root, {})
        return mapping.get((repo_id, episode_index)) or mapping.get(("*", episode_index)) or {}

    def _load_cached_frame(self, repo_id: str, video_relpath: str, frame_idx: int) -> np.ndarray:
        if self.fake_images:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        video_path = self.lerobot_home / repo_id / video_relpath
        cache_roots = _frame_cache_roots()
        if not cache_roots:
            raise FileNotFoundError("FRAME_CACHE_DIR/USE_FRAME_CACHE is not configured and fake_images=False")
        cache_dir = self._frame_cache_dir_by_video.get(video_path)
        if cache_dir is None:
            candidates = tuple(_frame_cache_dir_for(video_path, root) for root in cache_roots)
            cache_dir = next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])
            self._frame_cache_dir_by_video[video_path] = cache_dir
        candidates: list[Path] = []
        meta_path = cache_dir / "meta.json"
        if meta_path.is_file():
            meta = self._frame_cache_meta.get(cache_dir)
            if cache_dir not in self._frame_cache_meta:
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    meta = None
                self._frame_cache_meta[cache_dir] = meta
            if meta:
                source_fps = float(meta.get("source_fps") or meta.get("fps") or 30.0)
                dataset_fps = float(meta.get("dataset_fps") or 30.0)
                rule = str(meta.get("frame_index_rule") or "").lower()
                source_index_rule = "source" in rule and "frame" in rule and "dataset_frame_index" not in rule
                if source_index_rule and source_fps > 45.0 and abs(source_fps - dataset_fps) < 1.0:
                    raise FileNotFoundError(
                        "Ambiguous 60fps source-frame cache for 30fps human pack; rebuild cache with "
                        f"dataset_frame_index naming and dataset_fps=30: {cache_dir}"
                    )
                if source_index_rule:
                    source_idx = int(round(float(frame_idx) / dataset_fps * source_fps))
                    candidates.extend(cache_dir / f"{source_idx + off:06d}.jpg" for off in (0, 1, -1, 2, -2))
        candidates.append(cache_dir / f"{int(frame_idx):06d}.jpg")
        for path in candidates:
            if path.is_file():
                with Image.open(path) as img:
                    return np.array(img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)), copy=True)
        raise FileNotFoundError(f"Cached frame not found for {video_path} frame={frame_idx} cache={cache_dir}")

    def _bad_sample(self, reason: str) -> dict[str, bool | str]:
        return {"_bad_sample": True, "_bad_reason": reason}

    def _bad_decode_sample(self, repo_id: str, video_relpath: str, exc: Exception) -> dict[str, bool | str]:
        key = f"{repo_id}/{video_relpath}"
        self._decode_error_counts[key] += 1
        self._decode_error_total += 1
        self._recent_bad_samples.append(key)
        if self._decode_error_counts[key] <= 3:
            LOGGER.warning("[human_vlm_dataset] skipping bad sample %s: %s", key, exc)
        return {"_bad_sample": True, "_bad_reason": "decode"}

    def _norm_for_source(self, source: str) -> HumanVLMNormStats:
        stats = self._norm_stats_by_source.get(source, self._norm_stats)
        if stats is None:
            raise ValueError(f"No norm stats resolved for source {source}")
        return stats

    def _normalize_state(self, state: np.ndarray, source: str) -> np.ndarray:
        stats = self._norm_for_source(source)
        state21 = np.asarray(state, dtype=np.float32).reshape(-1)[:ACTION_DIM_21]
        if state21.shape[0] < ACTION_DIM_21:
            state21 = _pad_last_dim(state21, ACTION_DIM_21, value=0.0)
        state21 = _normalize_q01_q99(state21, stats.state_q01, stats.state_q99)
        return _pad_last_dim(state21, MODEL_ACTION_DIM, value=0.0).astype(np.float32)

    def _normalize_actions(self, actions: np.ndarray, source: str) -> np.ndarray:
        stats = self._norm_for_source(source)
        actions = actions[:, :ACTION_DIM_21].astype(np.float32, copy=True)
        if stats.action_q01 is None or stats.action_q99 is None:
            return actions
        return _normalize_q01_q99(actions, stats.action_q01[None, :], stats.action_q99[None, :])

    def _build_item_unchecked(self, anchor_idx: int, view_type: int) -> dict[str, Any]:
        file_idx, rg_idx, row_idx = self._locate_anchor(anchor_idx)
        group = self._read_row_group(file_idx, rg_idx)
        source = _source_key(self.pack_paths[file_idx])

        def value(key: str) -> Any:
            return group.column(key)[row_idx].as_py()

        def has_value(key: str) -> bool:
            return key in group.schema.names

        repo_id = str(value("repo_id"))
        episode_index = int(value("episode_index"))
        video_relpath = str(value("video_relpath"))
        task = str(value("task")).strip()
        sidecar = self._sidecar_for_row(file_idx, repo_id, episode_index)
        # Strict two-level annotation contract: the episode-level instruction
        # comes from the sidecar and ONLY from the sidecar, while the pack row
        # supplies the current subtask. No fallback to pack-row columns and
        # never to `episode_task_description`, which is a different field:
        # substituting it would silently train part of the mixture on another
        # annotation semantics and would mask a broken sidecar join.
        episode_instruction = str(sidecar.get("episode_instruction") or "").strip()
        if not task:
            return self._bad_sample("empty_task")
        # Episode-level split, enforced at read time so train and val can share
        # one pack layout, one index space and one set of source weights.
        if not episode_in_split(repo_id, episode_index, self.split, self.val_basis_points):
            return self._bad_sample("split_mismatch")
        if view_type == VIEW_SUBTASK and not episode_instruction:
            return self._bad_sample("missing_episode_instruction")
        if view_type == VIEW_SUBTASK and _normalized_text(episode_instruction) == _normalized_text(task):
            return self._bad_sample("degenerate_episode_instruction_equals_task")

        # Convert to the canonical model layout once, here at the dataset
        # boundary, so state, actions and masks can never disagree downstream.
        action = permute_packed_to_model(
            np.asarray(value("action"), dtype=np.float32).reshape(ACTION_HORIZON, -1)
        )
        action_mask = permute_packed_to_model(
            np.asarray(value("action_loss_mask"), dtype=np.bool_).reshape(action.shape)
        )
        fast_time_valid = action_mask[: self.fast_horizon, :ACTION_DIM_21].all(axis=-1)
        first_invalid = np.flatnonzero(~fast_time_valid)
        fast_prefix_horizon = int(first_invalid[0]) if len(first_invalid) else self.fast_horizon
        if view_type == VIEW_FAST:
            if fast_prefix_horizon == 0:
                return self._bad_sample("invalid_fast_action_mask")
            if not self.allow_partial_fast_horizon and fast_prefix_horizon != self.fast_horizon:
                return self._bad_sample("invalid_fast_action_mask")
        raw_state = np.asarray(value("observation.state"), dtype=np.float32).reshape(-1)
        if raw_state.shape[-1] < ACTION_DIM_21:
            raw_state = _pad_last_dim(raw_state, ACTION_DIM_21, value=0.0)
        canonical_state = permute_packed_to_model(raw_state)
        state_pose_valid = permute_packed_pose_valid_to_model(value("observation.state_pose_valid"))
        # Anchor the action quaternion signs to the current state, then chain
        # along time. camera_extrinsics, state and actions all live in the same
        # earliest-valid-history reference frame (verified: extrinsics at the
        # first valid slot are exactly identity and state == action[0]), so the
        # anchor is well defined and no rebase is applied here.
        action = unwrap_action_quaternion_signs(action, canonical_state, action_mask)
        state = zero_invalid_pose_blocks(self._normalize_state(canonical_state, source), state_pose_valid)

        try:
            und_frame_indices = [int(x) for x in value("und_frame_indices")]
            history = np.stack(
                [self._load_cached_frame(repo_id, video_relpath, int(frame_idx)) for frame_idx in und_frame_indices],
                axis=0,
            )
        except Exception as exc:
            return self._bad_decode_sample(repo_id, video_relpath, exc)

        # FAST and flow must supervise the SAME canonical h-step trajectory.
        horizon_actions = action[: self.fast_horizon, :ACTION_DIM_21]
        normalized_horizon = self._normalize_actions(horizon_actions, source)
        condition = None
        if view_type == VIEW_HIGH_LEVEL:
            tokenized = self.tokenizer.tokenize_subtask(episode_instruction, state, task)
        else:
            # Two sequences per low-level sample:
            #   tokenized  -> teacher-forced FAST tokens, for next-token CE only
            #   condition  -> identical prefix WITHOUT the GT response, used as
            #                 the Action Expert's conditioning so it can never
            #                 attend to the answer it is asked to regress.
            tokenized = self.tokenizer.tokenize_fast_action(
                task,
                state,
                normalized_horizon[:fast_prefix_horizon],
            )
            condition = self.tokenizer.tokenize_action_condition(task, state)
            if condition.truncated:
                return self._bad_sample("condition_token_truncated")
        if tokenized.truncated:
            return self._bad_sample("token_truncated")

        padded_action = _pad_last_dim(action, MODEL_ACTION_DIM, value=0.0).astype(np.float32)
        padded_action_mask = _pad_last_dim(action_mask, MODEL_ACTION_DIM, value=False).astype(np.bool_)

        # Continuous flow target: normalized, canonical, h-step, zero-padded to
        # the model action dim with padded dims masked out.
        flow_actions = _pad_last_dim(normalized_horizon, MODEL_ACTION_DIM, value=0.0).astype(np.float32)
        flow_mask = _pad_last_dim(
            action_mask[: self.fast_horizon, :ACTION_DIM_21], MODEL_ACTION_DIM, value=False
        ).astype(np.bool_)
        camera_extrinsics = np.asarray(value("observation.camera_extrinsics"), dtype=np.float32).reshape(
            HISTORY_SLOTS, 3, 4
        )
        camera_extrinsics4 = np.zeros((HISTORY_SLOTS, 4, 4), dtype=np.float32)
        camera_extrinsics4[:, :3, :4] = camera_extrinsics
        camera_extrinsics4[:, 3, 3] = 1.0

        zero_wrist = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        has_wrist_images = all(has_value(key) for key in self._OPTIONAL_WRIST_COLUMNS)
        if has_wrist_images:
            try:
                left_wrist = self._load_cached_frame(
                    repo_id,
                    str(value("left_wrist_video_relpath")),
                    int(value("left_wrist_frame_index")),
                )
                right_wrist = self._load_cached_frame(
                    repo_id,
                    str(value("right_wrist_video_relpath")),
                    int(value("right_wrist_frame_index")),
                )
            except Exception as exc:
                return self._bad_decode_sample(repo_id, video_relpath, exc)
        else:
            left_wrist = zero_wrist
            right_wrist = zero_wrist
        camera_token_mask = np.asarray(value("camera_token_history_mask"), dtype=np.bool_).reshape(HISTORY_SLOTS)
        return {
            "image": {
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "left_wrist_0_rgb": np.asarray(has_wrist_images),
                "right_wrist_0_rgb": np.asarray(has_wrist_images),
            },
            "state": state,
            "state_pose_valid": state_pose_valid,
            "actions": padded_action,
            "action_dim_mask": padded_action_mask,
            "action_time_valid_mask": padded_action_mask.any(axis=-1),
            "tokenized_prompt": tokenized.tokens,
            "tokenized_prompt_mask": tokenized.token_mask,
            "token_ar_mask": tokenized.ar_mask,
            "token_loss_mask": tokenized.loss_mask,
            "vlm_view_type": np.asarray(view_type, dtype=np.int32),
            "vlm_token_truncated": np.asarray(tokenized.truncated, dtype=np.bool_),
            # Batch-carried identity so the trainer can aggregate per-source and
            # per-view metrics across workers and DDP ranks; worker-local
            # counters cannot be reduced.
            "source_id": np.asarray(source_id_for(source), dtype=np.int32),
            "action_loss_weight": np.asarray(self.action_loss_weight, dtype=np.float32),
            # Continuous flow supervision over the same canonical h-step chunk.
            "flow_actions": flow_actions,
            "flow_action_mask": flow_mask,
            "flow_time_valid_mask": flow_mask.any(axis=-1),
            # Prefix-only conditioning for the Action Expert (low-level only).
            "flow_condition_tokens": (
                condition.tokens if condition is not None else np.zeros_like(tokenized.tokens)
            ),
            "flow_condition_mask": (
                condition.token_mask if condition is not None else np.zeros_like(tokenized.token_mask)
            ),
            "flow_condition_ar_mask": (
                condition.ar_mask if condition is not None else np.zeros_like(tokenized.ar_mask)
            ),
            "has_flow_target": np.asarray(condition is not None, dtype=np.bool_),
            "front_history_images": history,
            "front_history_masks": camera_token_mask,
            "camera_extrinsics": camera_extrinsics4,
            "camera_pose_valid": camera_token_mask,
            "camera_fov": np.asarray(value("observation.camera_fov"), dtype=np.float32).reshape(HISTORY_SLOTS, 2),
            "camera_fov_valid": np.asarray(value("observation.camera_fov_valid"), dtype=np.bool_).reshape(
                HISTORY_SLOTS
            ),
            "camera_image_hw": np.asarray(value("observation.camera_image_hw"), dtype=np.float32).reshape(
                HISTORY_SLOTS, 2
            ),
        }

    def index_space_counts(self) -> dict[tuple[str, int], int]:
        """Exact (source, view) sample counts over one full index space.

        Analytic rather than enumerated: the real mixture has tens of millions
        of anchors, so brute force is not an option for a production audit.
        Because `view_cycle_span` is the LCM of the per-source schedule
        lengths, `span / len(schedule)` is always an integer and every anchor
        walks its schedule a whole number of times -- so these counts are
        exact, not approximate. `sampling_audit_test.py` cross-checks them
        against brute-force enumeration on small fixtures.
        """
        counts: dict[tuple[str, int], int] = {}
        for source, schedule in self._view_schedule_by_source.items():
            n_anchors = self._source_cum_rows[source][-1]
            if self.view_cycle_mode == "physical_once":
                shift = self._physical_view_shift(source)
                quotient, remainder = divmod(n_anchors, len(schedule))
                for local_offset in range(len(schedule)):
                    count = quotient + int(local_offset < remainder)
                    view = schedule[(local_offset + shift) % len(schedule)]
                    counts[(source, view)] = counts.get((source, view), 0) + count
                continue
            reps = self.view_cycle_span // len(schedule)
            for view in set(schedule):
                counts[(source, view)] = n_anchors * reps * schedule.count(view)
        return counts

    def index_space_ratios(self) -> dict[str, float]:
        """Realized source / view / source-view ratios over the index space."""
        counts = self.index_space_counts()
        total = sum(counts.values())
        out: dict[str, float] = {}
        by_source: dict[str, int] = defaultdict(int)
        by_view: dict[int, int] = defaultdict(int)
        for (source, view), n in counts.items():
            by_source[source] += n
            by_view[view] += n
            out[f"source_view/{source}/{_view_name(view)}"] = n / total
        for source, n in by_source.items():
            out[f"source/{source}"] = n / total
        for view, n in by_view.items():
            out[f"view/{_view_name(view)}"] = n / total
        return out

    def route_view(self, slot: int, cycle: int) -> tuple[str, int, int]:
        """Map (anchor slot, cycle) -> (source, view, global anchor index).

        Pure function of its arguments and the fixed route seed: no RNG state,
        no per-worker state, nothing that a resume could perturb.
        """
        source = self._source_for_anchor(slot)
        schedule = self._view_schedule_by_source[source]
        if self.view_cycle_mode == "physical_once":
            source_offset = self._source_offset_for_anchor(source, slot)
            phase = (source_offset + self._physical_view_shift(source)) % len(schedule)
            return source, schedule[phase], slot
        phase = (splitmix64(slot ^ self._view_route_seed) + cycle) % len(schedule)
        return source, schedule[phase], slot

    def _physical_view_shift(self, source: str) -> int:
        schedule = self._view_schedule_by_source[source]
        source_seed = splitmix64(self._view_route_seed ^ source_id_for(source))
        return (source_seed + self.view_route_phase) % len(schedule)

    def _source_offset_for_anchor(self, source: str, anchor_idx: int) -> int:
        ranges = self._anchor_ranges_by_source[source]
        pos = bisect.bisect_right(self._source_global_starts[source], anchor_idx) - 1
        if pos < 0:
            raise IndexError(f"Anchor {anchor_idx} does not belong to source {source}")
        global_start, length = ranges[pos]
        if anchor_idx >= global_start + length:
            raise IndexError(f"Anchor {anchor_idx} does not belong to source {source}")
        previous = self._source_cum_rows[source][pos - 1] if pos else 0
        return previous + anchor_idx - global_start

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # The index carries the cycle, so an anchor walks its whole view
        # schedule instead of being locked to one view forever.
        flat = idx % self.num_frames
        cycle, slot = divmod(flat, self.num_anchor_rows)
        source, view_type, original_anchor_idx = self.route_view(slot, cycle)
        source_view = f"{source}/{_view_name(view_type)}"
        self._requested_by_source_view[source_view] += 1
        self._requested_by_source[source] += 1
        self._requested_by_view[_view_name(view_type)] += 1
        invalid_reasons: Counter[str] = Counter()
        for attempt in range(self._max_filter_attempts):
            anchor_idx = self._retry_anchor_for_source(source, original_anchor_idx, attempt)
            item = self._build_item_unchecked(anchor_idx, view_type)
            if not item.pop("_bad_sample", False):
                self._returned_by_source_view[source_view] += 1
                self._returned_by_source[source] += 1
                self._returned_by_view[_view_name(view_type)] += 1
                if attempt:
                    self._retry_by_source_view[source_view] += attempt
                return item
            reason = str(item.get("_bad_reason", "unknown"))
            invalid_reasons[reason] += 1
            self._invalid_by_source_view_reason[f"{source_view}/{reason}"] += 1
            self._resample_total += 1
        raise RuntimeError(
            "[human_vlm_dataset] exceeded max resample attempts; "
            f"source={source} view={_view_name(view_type)} invalid_reasons={dict(invalid_reasons)} "
            f"decode_error_total={self._decode_error_total} resamples={self._resample_total} "
            f"unique_bad_videos={len(self._decode_error_counts)} recent={list(self._recent_bad_samples)}"
        )

    @staticmethod
    def _ratio(counter: Counter[str]) -> dict[str, float]:
        total = sum(counter.values())
        return {k: v / total for k, v in sorted(counter.items())} if total else {}

    def runtime_stats(self) -> dict[str, Any]:
        return {
            "configured_source_weights": dict(self._source_weights),
            "configured_view_weights_by_source": {
                s: {_view_name(v): w for v, w in weights.items()}
                for s, weights in self._view_weights_by_source.items()
            },
            "requested_by_source": dict(self._requested_by_source),
            "returned_by_source": dict(self._returned_by_source),
            "realized_source_ratio": self._ratio(self._returned_by_source),
            "requested_by_view": dict(self._requested_by_view),
            "returned_by_view": dict(self._returned_by_view),
            "realized_view_ratio": self._ratio(self._returned_by_view),
            "requested_by_source_view": dict(self._requested_by_source_view),
            "returned_by_source_view": dict(self._returned_by_source_view),
            "invalid_by_source_view_reason": dict(self._invalid_by_source_view_reason),
            "retry_by_source_view": dict(self._retry_by_source_view),
            "decode_error_total": self._decode_error_total,
            "resample_total": self._resample_total,
            "unique_bad_videos": len(self._decode_error_counts),
            "recent_bad_samples": list(self._recent_bad_samples),
        }
