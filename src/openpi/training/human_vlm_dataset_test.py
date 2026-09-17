import hashlib
import os

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import json

import openpi.models.tokenizer as tokenizer_module
from openpi.models.tokenizer import HumanVLMDiscreteTokenizer
from openpi.models.tokenizer import TokenizedVLMView
import openpi.training.human_vlm_dataset as human_vlm_dataset


def test_frame_cache_fallback_resolves_and_is_memoized(monkeypatch, tmp_path):
    home = tmp_path / "home"
    primary = tmp_path / "primary"
    fallback = tmp_path / "fallback"
    primary.mkdir()
    video_path = home / "repo" / "video.mp4"
    key = hashlib.sha1(os.path.realpath(str(video_path)).encode()).hexdigest()[:16]
    cache_dir = fallback / key
    cache_dir.mkdir(parents=True)
    (cache_dir / "meta.json").write_text(
        json.dumps({"frame_index_rule": "dataset_frame_index", "dataset_fps": 30.0})
    )
    Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8)).save(cache_dir / "000016.jpg")
    monkeypatch.setenv("FRAME_CACHE_DIR", str(primary))
    monkeypatch.setenv("FRAME_CACHE_FALLBACK_DIRS", str(fallback))

    dataset = object.__new__(human_vlm_dataset.HumanVLMPairedOfflineDataset)
    dataset.fake_images = False
    dataset.lerobot_home = home
    dataset._frame_cache_meta = {}
    dataset._frame_cache_dir_by_video = {}
    image = dataset._load_cached_frame("repo", "video.mp4", 16)

    assert image.shape == (human_vlm_dataset.IMAGE_SIZE, human_vlm_dataset.IMAGE_SIZE, 3)
    assert dataset._frame_cache_dir_by_video[video_path] == cache_dir


class _FakeHumanVLMTokenizer:
    subtask_calls = []
    fast_calls = []
    condition_calls = []
    fast_action_arrays = []
    subtask_states = []
    fast_states = []

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def tokenize_subtask(self, episode_instruction, state, subtask):
        assert episode_instruction != subtask
        assert episode_instruction
        assert subtask
        self.subtask_calls.append((episode_instruction, subtask))
        self.subtask_states.append(np.asarray(state, dtype=np.float32).copy())
        return TokenizedVLMView(
            tokens=np.asarray([1, 2, 3, 0], dtype=np.int32),
            token_mask=np.asarray([True, True, True, False]),
            ar_mask=np.asarray([0, 0, 1, 0], dtype=np.int32),
            loss_mask=np.asarray([False, False, True, False]),
            truncated=False,
        )

    def tokenize_action_condition(self, subtask, state):
        assert subtask
        self.condition_calls.append(subtask)
        return TokenizedVLMView(
            tokens=np.asarray([7, 8, 0, 0], dtype=np.int32),
            token_mask=np.asarray([True, True, False, False]),
            ar_mask=np.asarray([0, 0, 0, 0], dtype=np.int32),
            loss_mask=np.asarray([False, False, False, False]),
            truncated=False,
        )

    def tokenize_fast_action(self, subtask, state, actions):
        assert subtask
        assert actions.shape[1] == 21
        self.fast_calls.append((subtask, actions.shape[0]))
        self.fast_action_arrays.append(np.asarray(actions, dtype=np.float32).copy())
        self.fast_states.append(np.asarray(state, dtype=np.float32).copy())
        return TokenizedVLMView(
            tokens=np.asarray([4, 5, 6, 0], dtype=np.int32),
            token_mask=np.asarray([True, True, True, False]),
            ar_mask=np.asarray([0, 1, 1, 0], dtype=np.int32),
            loss_mask=np.asarray([False, True, True, False]),
            truncated=False,
        )


def _pose_rows():
    extrinsics = []
    for _ in range(4):
        mat = np.eye(4, dtype=np.float32)[:3]
        extrinsics.extend(mat.reshape(-1).tolist())
    return extrinsics


def _row(
    task="pick the cup",
    episode_index=0,
    *,
    repo_id="EgoDex_lerobot_active",
    action_mask=None,
    state=None,
    state_pose_valid=None,
):
    if action_mask is None:
        action_mask = np.ones((50, 21), dtype=np.bool_)
    if state is None:
        state = np.zeros(21, dtype=np.float32)
    if state_pose_valid is None:
        state_pose_valid = np.ones(3, dtype=np.bool_)
    return {
        "repo_id": repo_id,
        "episode_index": episode_index,
        "frame_index": 16,
        "video_relpath": "videos/chunk-000/file-000.mp4",
        "task": task,
        "action": np.zeros((50, 21), dtype=np.float32).reshape(-1).tolist(),
        "action_loss_mask": action_mask.reshape(-1).tolist(),
        "observation.state": np.asarray(state, dtype=np.float32).tolist(),
        "observation.state_pose_valid": np.asarray(state_pose_valid, dtype=np.bool_).tolist(),
        "und_frame_indices": [0, 0, 0, 16],
        "camera_token_history_mask": [False, False, False, True],
        "observation.camera_extrinsics": _pose_rows(),
        "observation.camera_fov": np.ones((4, 2), dtype=np.float32).reshape(-1).tolist(),
        "observation.camera_fov_valid": [False, False, False, True],
        "observation.camera_image_hw": np.asarray([[1080, 1920]] * 4, dtype=np.float32).reshape(-1).tolist(),
    }


def _write_pack(tmp_path, rows, sidecar_rows, *, pack_name="EgoDex_stride3_v5_intrinsics_canonical_20260714", repo_id="EgoDex_lerobot_active"):
    pack_root = tmp_path / pack_name
    pack_dir = pack_root / repo_id
    pack_dir.mkdir(parents=True)
    pack_path = pack_dir / "part-000.parquet"
    pq.write_table(pa.Table.from_pylist(rows), pack_path)
    pq.write_table(pa.Table.from_pylist(sidecar_rows), pack_root / "episode_annotations.parquet")
    return pack_path


def _write_norm_stats(tmp_path, *, state_q01=None, state_q99=None, action_q01=None, action_q99=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    if state_q01 is None:
        state_q01 = np.zeros(21, dtype=np.float32)
    if state_q99 is None:
        state_q99 = np.ones(21, dtype=np.float32) * 10.0
    if action_q01 is None:
        action_q01 = np.zeros(21, dtype=np.float32)
    if action_q99 is None:
        action_q99 = np.ones(21, dtype=np.float32)
    path = tmp_path / "norm_stats.json"
    path.write_text(
        json.dumps(
            {
                "action_layout": human_vlm_dataset.MODEL_ACTION_LAYOUT,
                "state_q01": np.asarray(state_q01, dtype=np.float32).tolist(),
                "state_q99": np.asarray(state_q99, dtype=np.float32).tolist(),
                "action_q01": np.asarray(action_q01, dtype=np.float32).tolist(),
                "action_q99": np.asarray(action_q99, dtype=np.float32).tolist(),
            }
        )
    )
    return path


def test_human_source_norm_map_and_action_weight(monkeypatch, tmp_path):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    pack_path = _write_pack(
        tmp_path,
        [_row(state=np.full(21, 5.0, dtype=np.float32))],
        [{"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "make breakfast"}],
    )
    shared = _write_norm_stats(tmp_path / "shared", state_q99=np.full(21, 20.0))
    egodex = _write_norm_stats(tmp_path / "egodex", state_q99=np.full(21, 10.0))
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        norm_stats_path=str(shared),
        source_norm_stats_paths=f"EgoDex={egodex}",
        action_loss_weight=0.2,
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=False,
        fake_images=True,
    )

    item = dataset[0]
    assert np.allclose(item["state"][:21], 0.0)
    assert item["action_loss_weight"].item() == pytest.approx(0.2)


def test_selection_manifest_applies_episode_range_and_anchor_stride(monkeypatch, tmp_path):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    pack_path = _write_pack(
        tmp_path,
        [_row(task=f"task {index}") for index in range(5)],
        [{"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "instruction"}],
    )
    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        json.dumps(
            {
                "path": str(pack_path),
                "source": "EgoDex",
                "row_group": 0,
                "row_start": 1,
                "num_rows": 4,
                "anchor_stride": 2,
            }
        )
        + "\n"
    )
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        norm_stats_path=str(_write_norm_stats(tmp_path / "norm")),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=False,
        fake_images=True,
        selection_manifest_path=selection,
    )

    assert dataset.num_anchor_rows == 2
    assert dataset._locate_anchor(0) == (0, 0, 1)
    assert dataset._locate_anchor(1) == (0, 0, 3)


def test_selection_manifest_rejects_source_reweighting(monkeypatch, tmp_path):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    pack_path = _write_pack(
        tmp_path,
        [_row()],
        [{"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "instruction"}],
    )
    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        json.dumps(
            {
                "path": str(pack_path),
                "source": "EgoDex",
                "row_group": 0,
                "row_start": 0,
                "num_rows": 1,
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="source_weights must be unset"):
        human_vlm_dataset.HumanVLMPairedOfflineDataset(
            pack_paths=str(pack_path),
            source_weights="EgoDex=1",
            norm_stats_path=str(_write_norm_stats(tmp_path / "norm")),
            max_token_len=4,
            enable_subtask_view=True,
            enable_fast_view=False,
            fake_images=True,
            selection_manifest_path=selection,
        )


def _index_for(dataset, source, view, offset=0):
    """Flat index of the `offset`-th sample of a given (source, view).

    View routing is cycle-dependent, so (source, view) is no longer a
    contiguous block: the index must be resolved by asking the router.
    """
    found = 0
    for cycle in range(dataset.view_cycle_span):
        for slot in range(dataset.num_anchor_rows):
            src, v, _anchor = dataset.route_view(slot, cycle)
            if src == source and v == view:
                if found == offset:
                    return cycle * dataset.num_anchor_rows + slot
                found += 1
    raise AssertionError(f"no sample for {source}/{view} at offset {offset}")


def _routed_pairs(dataset):
    """Every (source, view_name) the index space can actually produce."""
    return {
        (src, human_vlm_dataset._view_name(v))
        for cycle in range(dataset.view_cycle_span)
        for slot in range(dataset.num_anchor_rows)
        for src, v, _a in [dataset.route_view(slot, cycle)]
    }


def test_as_paths_expands_manifest_json_and_text_lists(tmp_path):
    p0 = tmp_path / "a.parquet"
    p1 = tmp_path / "b.parquet"
    sidecar = tmp_path / "episode_annotations.parquet"
    p0.write_text("")
    p1.write_text("")
    sidecar.write_text("")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"parquets": [str(p0), str(sidecar), str(p1)]}))
    relative_manifest = tmp_path / "relative_manifest.json"
    relative_manifest.write_text(json.dumps({"parquets": [p0.name, sidecar.name, p1.name]}))
    text_list = tmp_path / "manifest.txt"
    text_list.write_text(f"# comment\n{sidecar}\n{p1}\n\n")

    assert human_vlm_dataset._as_paths(manifest) == [p0, p1]
    assert human_vlm_dataset._as_paths(relative_manifest) == [p0, p1]
    assert human_vlm_dataset._as_paths(text_list) == [p1]
    assert human_vlm_dataset._as_paths(str(p0)) == [p0]
    assert human_vlm_dataset._as_paths(str(tmp_path / "*.parquet")) == [p0, p1]


def test_episode_split_is_episode_disjoint_and_bucket_deterministic():
    keys = [("EgoDex_lerobot_active", i) for i in range(2000)]
    train = {
        key
        for key in keys
        if human_vlm_dataset.episode_in_split(key[0], key[1], human_vlm_dataset.SPLIT_TRAIN, 200)
    }
    val = {
        key
        for key in keys
        if human_vlm_dataset.episode_in_split(key[0], key[1], human_vlm_dataset.SPLIT_VAL, 200)
    }
    all_keys = set(keys)

    assert train.isdisjoint(val)
    assert train | val == all_keys
    assert all(
        human_vlm_dataset.episode_split_bucket(repo_id, episode_index)
        == human_vlm_dataset.episode_split_bucket(repo_id, episode_index)
        for repo_id, episode_index in keys
    )


def test_val_basis_points_reuses_the_same_bucket_contract():
    keys = [("EgoLive_active_shards/shard-000", i) for i in range(5000)]
    val_100 = {
        key
        for key in keys
        if human_vlm_dataset.episode_in_split(key[0], key[1], human_vlm_dataset.SPLIT_VAL, 100)
    }
    val_200 = {
        key
        for key in keys
        if human_vlm_dataset.episode_in_split(key[0], key[1], human_vlm_dataset.SPLIT_VAL, 200)
    }
    by_bucket = {
        key
        for key in keys
        if human_vlm_dataset.episode_split_bucket(key[0], key[1]) < 100
    }

    assert val_100 == by_bucket
    assert val_100 <= val_200


def test_human_vlm_dataset_enforces_episode_split_on_real_rows(monkeypatch, tmp_path):
    _FakeHumanVLMTokenizer.subtask_calls = []
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    repo_id = "EgoDex_lerobot_active"

    def first_episode_for(split):
        for episode_index in range(10_000):
            if human_vlm_dataset.episode_in_split(repo_id, episode_index, split, 200):
                return episode_index
        raise AssertionError(f"could not find {split} episode")

    val_episode = first_episode_for(human_vlm_dataset.SPLIT_VAL)
    train_episode = first_episode_for(human_vlm_dataset.SPLIT_TRAIN)
    pack_path = _write_pack(
        tmp_path,
        [
            _row(task="validation-only subtask", episode_index=val_episode, repo_id=repo_id),
            _row(task="train-only subtask", episode_index=train_episode, repo_id=repo_id),
        ],
        [
            {"repo_id": repo_id, "episode_index": val_episode, "episode_instruction": "validation episode"},
            {"repo_id": repo_id, "episode_index": train_episode, "episode_instruction": "train episode"},
        ],
    )
    norm_stats = _write_norm_stats(tmp_path)

    train_dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        norm_stats_path=str(norm_stats),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=False,
        fake_images=True,
        split=human_vlm_dataset.SPLIT_TRAIN,
    )
    train_item = train_dataset[0]
    assert train_item["vlm_view_type"].item() == human_vlm_dataset.VIEW_SUBTASK
    assert _FakeHumanVLMTokenizer.subtask_calls[-1] == ("train episode", "train-only subtask")
    assert train_dataset._build_item_unchecked(0, human_vlm_dataset.VIEW_SUBTASK)["_bad_reason"] == "split_mismatch"  # noqa: SLF001

    val_dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        norm_stats_path=str(norm_stats),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=False,
        fake_images=True,
        split=human_vlm_dataset.SPLIT_VAL,
    )
    val_item = val_dataset[0]
    assert val_item["vlm_view_type"].item() == human_vlm_dataset.VIEW_SUBTASK
    assert _FakeHumanVLMTokenizer.subtask_calls[-1] == ("validation episode", "validation-only subtask")


def test_human_vlm_dataset_joins_sidecar_and_preserves_pack_task(monkeypatch, tmp_path):
    _FakeHumanVLMTokenizer.subtask_calls = []
    _FakeHumanVLMTokenizer.subtask_states = []
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    pack_path = _write_pack(
        tmp_path,
        [_row(task="pick the cup")],
        [
            {
                "repo_id": "EgoDex_lerobot_active",
                "episode_index": 0,
                "episode_instruction": "make breakfast",
                "subtask_segments_json": "[]",
            }
        ],
    )
    norm_stats = _write_norm_stats(tmp_path)

    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        norm_stats_path=str(norm_stats),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=False,
        fake_images=True,
    )
    item = dataset[0]

    assert item["vlm_view_type"].item() == human_vlm_dataset.VIEW_SUBTASK
    assert _FakeHumanVLMTokenizer.subtask_calls == [("make breakfast", "pick the cup")]
    assert item["front_history_images"].shape == (4, 224, 224, 3)
    assert item["action_dim_mask"].shape == (50, 32)
    assert item["action_dim_mask"][:, :21].all()
    assert not item["action_dim_mask"][:, 21:].any()


def test_human_vlm_dataset_rejects_duplicate_sidecar_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    pack_path = _write_pack(
        tmp_path,
        [_row(task="pick the cup")],
        [
            {"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "a"},
            {"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "b"},
        ],
    )
    with pytest.raises(ValueError, match="duplicate"):
        human_vlm_dataset.HumanVLMPairedOfflineDataset(
            pack_paths=str(pack_path),
            norm_stats_path=str(_write_norm_stats(tmp_path)),
            max_token_len=4,
            enable_subtask_view=True,
            enable_fast_view=False,
            fake_images=True,
        )


def test_human_vlm_dataset_filters_exact_duplicate_subtask_view(monkeypatch, tmp_path):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    pack_path = _write_pack(
        tmp_path,
        [_row(task="pick the cup")],
        [
            {
                "repo_id": "EgoDex_lerobot_active",
                "episode_index": 0,
                "episode_instruction": "pick the cup",
                "subtask_segments_json": "[]",
            }
        ],
    )
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=False,
        fake_images=True,
        max_filter_attempts=1,
    )
    with pytest.raises(RuntimeError, match="exceeded max resample"):
        dataset[0]


def test_human_vlm_dataset_normalizes_state_before_tokenization(monkeypatch, tmp_path):
    _FakeHumanVLMTokenizer.subtask_states = []
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    raw_state = np.linspace(-5, 15, 21, dtype=np.float32)
    state_q01 = np.zeros(21, dtype=np.float32)
    state_q99 = np.ones(21, dtype=np.float32) * 10.0
    state_q01[3] = 2.0
    state_q99[3] = 2.0  # degenerate dimension should become 0, not NaN.
    pack_path = _write_pack(
        tmp_path,
        [_row(task="pick the cup", state=raw_state)],
        [{"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "make breakfast"}],
    )
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        norm_stats_path=str(_write_norm_stats(tmp_path, state_q01=state_q01, state_q99=state_q99)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=False,
        fake_images=True,
    )
    item = dataset[0]
    state_seen = _FakeHumanVLMTokenizer.subtask_states[-1]

    assert np.isfinite(state_seen).all()
    assert np.all(state_seen[:21] >= -1.0)
    assert np.all(state_seen[:21] <= 1.0)
    assert state_seen[3] == 0.0
    assert not state_seen[21:].any()
    assert np.allclose(item["state"], state_seen)


def test_human_vlm_retry_preserves_subtask_view_when_both_views_enabled(monkeypatch, tmp_path):
    _FakeHumanVLMTokenizer.subtask_calls = []
    _FakeHumanVLMTokenizer.fast_calls = []
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    pack_path = _write_pack(
        tmp_path,
        [_row(task="same task", episode_index=0), _row(task="valid subtask", episode_index=1)],
        [
            {"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "same task"},
            {"repo_id": "EgoDex_lerobot_active", "episode_index": 1, "episode_instruction": "make dinner"},
        ],
    )
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=True,
        fake_images=True,
    )
    # Cycle-dependent routing means index 0 is not guaranteed to be the
    # subtask view, so resolve the index through the router.
    item = dataset[_index_for(dataset, "EgoDex", human_vlm_dataset.VIEW_SUBTASK)]

    assert item["vlm_view_type"].item() == human_vlm_dataset.VIEW_SUBTASK
    assert _FakeHumanVLMTokenizer.subtask_calls[-1] == ("make dinner", "valid subtask")
    assert not _FakeHumanVLMTokenizer.fast_calls


def test_human_vlm_retry_preserves_fast_view_when_both_views_enabled(monkeypatch, tmp_path):
    _FakeHumanVLMTokenizer.subtask_calls = []
    _FakeHumanVLMTokenizer.fast_calls = []
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    bad_mask = np.zeros((50, 21), dtype=np.bool_)
    pack_path = _write_pack(
        tmp_path,
        [_row(task="bad fast", episode_index=0, action_mask=bad_mask), _row(task="good fast", episode_index=1)],
        [
            {"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "do task"},
            {"repo_id": "EgoDex_lerobot_active", "episode_index": 1, "episode_instruction": "do task"},
        ],
    )
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=True,
        fake_images=True,
    )
    item = dataset[_index_for(dataset, "EgoDex", human_vlm_dataset.VIEW_FAST)]

    assert item["vlm_view_type"].item() == human_vlm_dataset.VIEW_FAST
    assert _FakeHumanVLMTokenizer.fast_calls[-1] == ("good fast", 50)
    assert not _FakeHumanVLMTokenizer.subtask_calls


@pytest.mark.parametrize("view", [human_vlm_dataset.VIEW_SUBTASK, human_vlm_dataset.VIEW_FAST])
def test_invalid_state_pose_blocks_are_zeroed_before_tokenization(monkeypatch, tmp_path, view):
    _FakeHumanVLMTokenizer.subtask_states = []
    _FakeHumanVLMTokenizer.fast_states = []
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    # Packed state/valid order is [head, left, right]. The left pose is invalid
    # and contains an obvious placeholder that must not survive normalization.
    packed_state = np.concatenate(
        [
            np.arange(1, 8, dtype=np.float32),
            np.full(7, 999.0, dtype=np.float32),
            np.arange(21, 28, dtype=np.float32),
        ]
    )
    pack_path = _write_pack(
        tmp_path,
        [_row(state=packed_state, state_pose_valid=[True, False, True])],
        [{"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "make breakfast"}],
    )
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        norm_stats_path=str(_write_norm_stats(tmp_path, state_q99=np.full(21, 1000.0))),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=True,
        fake_images=True,
    )
    item = dataset[_index_for(dataset, "EgoDex", view)]
    tokenizer_state = (
        _FakeHumanVLMTokenizer.subtask_states[-1]
        if view == human_vlm_dataset.VIEW_SUBTASK
        else _FakeHumanVLMTokenizer.fast_states[-1]
    )

    assert item["state_pose_valid"].tolist() == [False, True, True]
    assert np.all(item["state"][:7] == 0.0)
    assert np.all(tokenizer_state[:7] == 0.0)
    assert not np.all(item["state"][7:21] == 0.0)


def test_human_vlm_retry_preserves_source(monkeypatch, tmp_path):
    _FakeHumanVLMTokenizer.subtask_calls = []
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    egodex_pack = _write_pack(
        tmp_path,
        [_row(task="egodex valid", episode_index=0, repo_id="EgoDex_lerobot_active")],
        [{"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "egodex episode"}],
    )
    egolive_pack = _write_pack(
        tmp_path,
        [
            _row(task="egolive invalid", episode_index=0, repo_id="EgoLive_active_shards/shard-000"),
            _row(task="egolive valid", episode_index=1, repo_id="EgoLive_active_shards/shard-000"),
        ],
        [
            {"repo_id": "EgoLive_active_shards/shard-000", "episode_index": 0, "episode_instruction": "egolive invalid"},
            {"repo_id": "EgoLive_active_shards/shard-000", "episode_index": 1, "episode_instruction": "egolive episode"},
        ],
        pack_name="EgoLive_checked_stride3_fov_actionvalid_highmem_20260717",
        repo_id="shard-000",
    )
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=f"{egodex_pack},{egolive_pack}",
        source_weights="EgoDex=1,EgoLive=1",
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=False,
        fake_images=True,
    )
    item = dataset[_index_for(dataset, "EgoLive", human_vlm_dataset.VIEW_SUBTASK)]

    assert item["vlm_view_type"].item() == human_vlm_dataset.VIEW_SUBTASK
    assert _FakeHumanVLMTokenizer.subtask_calls[-1] == ("egolive episode", "egolive valid")


def test_human_vlm_retry_exhaustion_reports_source_view_and_reasons(monkeypatch, tmp_path):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    pack_path = _write_pack(
        tmp_path,
        [_row(task="same task", episode_index=0)],
        [{"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "same task"}],
    )
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=False,
        fake_images=True,
        max_filter_attempts=2,
    )
    with pytest.raises(RuntimeError, match="source=EgoDex view=high_level.*degenerate_episode_instruction_equals_task"):
        dataset[0]


def _tiny_sentencepiece_model(tmp_path):
    sentencepiece = pytest.importorskip("sentencepiece")
    corpus = tmp_path / "spm_corpus.txt"
    corpus.write_text(
        "\n".join(
            [
                "Task: make breakfast, State: " + " ".join(str(i) for i in range(256)) + ";\nSubtask: grasp mug",
                "Task: grasp mug, State: " + " ".join(str(i) for i in range(256)) + ";\nAction: |",
                "open drawer place cup close lid pour water",
            ]
            * 20
        )
    )
    prefix = tmp_path / "tiny_pg"
    sentencepiece.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=256,
        model_type="bpe",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        hard_vocab_limit=False,
    )
    return prefix.with_suffix(".model")


class _TinyFastProcessor:
    def __call__(self, actions):
        del actions
        return [np.asarray([4, 5, 6], dtype=np.int32)]

    def decode(self, tokens, *, time_horizon, action_dim):
        del tokens
        return [np.zeros((time_horizon, action_dim), dtype=np.float32)]


def test_real_human_vlm_tokenizer_subtask_boundaries_and_masks(monkeypatch, tmp_path):
    model_path = _tiny_sentencepiece_model(tmp_path)
    monkeypatch.setattr(tokenizer_module.download, "maybe_download", lambda *args, **kwargs: model_path)
    tokenizer = HumanVLMDiscreteTokenizer(max_len=160, fast_tokenizer_path=None)
    state = np.linspace(-1, 1, 32, dtype=np.float32)
    high_level = "make breakfast"
    subtask = "grasp mug"

    tokenized = tokenizer.tokenize_subtask(high_level, state, subtask)
    expected_prefix = f"Task: {high_level}, State: {tokenizer_module._discretize_state_to_string(state)};\nSubtask: "
    prefix_tokens = tokenizer._paligemma_tokenizer.encode(expected_prefix, add_bos=True)
    response_tokens = tokenizer._paligemma_tokenizer.encode(subtask, add_eos=True)
    prefix_text = tokenizer._paligemma_tokenizer.decode(prefix_tokens)
    response_text = tokenizer._paligemma_tokenizer.decode(response_tokens)

    assert high_level in prefix_text
    assert "Subtask:" in prefix_text
    assert subtask not in prefix_text
    assert subtask in response_text
    assert tokenized.loss_mask[: len(prefix_tokens)].sum() == 0
    assert tokenized.loss_mask[len(prefix_tokens) : len(prefix_tokens) + len(response_tokens)].all()
    assert tokenized.loss_mask[len(prefix_tokens) + len(response_tokens) :].sum() == 0
    assert tokenized.token_mask[len(prefix_tokens) + len(response_tokens) :].sum() == 0
    assert tokenized.tokens[len(prefix_tokens) + len(response_tokens) - 1] == tokenizer.eos_id
    assert not tokenized.truncated

    truncated = HumanVLMDiscreteTokenizer(max_len=len(prefix_tokens) - 1, fast_tokenizer_path=None)
    truncated._paligemma_tokenizer = tokenizer._paligemma_tokenizer
    assert truncated.tokenize_subtask(high_level, state, subtask).truncated


def test_real_human_vlm_tokenizer_fast_boundaries_and_masks(monkeypatch, tmp_path):
    model_path = _tiny_sentencepiece_model(tmp_path)
    monkeypatch.setattr(tokenizer_module.download, "maybe_download", lambda *args, **kwargs: model_path)
    monkeypatch.setattr(tokenizer_module.AutoProcessor, "from_pretrained", lambda *args, **kwargs: _TinyFastProcessor())
    tokenizer = HumanVLMDiscreteTokenizer(max_len=160, fast_tokenizer_path="fake-fast")
    state = np.zeros(32, dtype=np.float32)
    actions = np.zeros((10, 21), dtype=np.float32)
    subtask = "grasp mug"

    tokenized = tokenizer.tokenize_fast_action(subtask, state, actions)
    expected_prefix = f"Task: {subtask}, State: {tokenizer_module._discretize_state_to_string(state)};\nAction: "
    prefix_tokens = tokenizer._paligemma_tokenizer.encode(expected_prefix, add_bos=True)
    response_len = int(tokenized.loss_mask.sum())
    prefix_text = tokenizer._paligemma_tokenizer.decode(prefix_tokens)

    assert subtask in prefix_text
    assert "Action:" in prefix_text
    assert tokenized.loss_mask[: len(prefix_tokens)].sum() == 0
    assert tokenized.loss_mask[len(prefix_tokens) : len(prefix_tokens) + response_len].all()
    assert tokenized.tokens[len(prefix_tokens) + response_len - 1] == tokenizer.eos_id
    assert not tokenized.truncated

    truncated = HumanVLMDiscreteTokenizer(max_len=len(prefix_tokens), fast_tokenizer_path="fake-fast")
    truncated._paligemma_tokenizer = tokenizer._paligemma_tokenizer
    truncated._fast_tokenizer = tokenizer._fast_tokenizer
    assert truncated.tokenize_fast_action(subtask, state, actions).truncated


def test_parquet_handles_are_lazy_and_bounded(monkeypatch, tmp_path):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    pack_path = _write_pack(
        tmp_path,
        [_row(task="pick the cup")],
        [{"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "make breakfast"}],
    )
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=False,
        fake_images=True,
    )
    # No handles are opened at construction time.
    assert len(dataset._parquet_handles) == 0
    assert dataset._row_group_sizes and dataset._row_group_sizes[0]

    dataset[0]
    assert 1 <= len(dataset._parquet_handles) <= dataset._parquet_handle_limit

    # A forked worker (different pid) must not reuse inherited descriptors.
    dataset._parquet_handles_pid = -1
    dataset._row_group_cache.clear()
    dataset[0]
    assert dataset._parquet_handles_pid == __import__("os").getpid()
    assert 1 <= len(dataset._parquet_handles) <= dataset._parquet_handle_limit


def test_parquet_handle_limit_evicts_oldest(monkeypatch, tmp_path):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    paths = []
    for i in range(3):
        paths.append(
            _write_pack(
                tmp_path / f"p{i}",
                [_row(task=f"task {i}")],
                [{"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "cook"}],
            )
        )
    monkeypatch.setenv("HUMAN_VLM_PARQUET_HANDLE_LIMIT", "4")
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=",".join(str(p) for p in paths),
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=False,
        fake_images=True,
    )
    dataset._parquet_handle_limit = 2
    for file_idx in range(3):
        dataset._parquet_file(file_idx)
    assert len(dataset._parquet_handles) == 2
    assert 0 not in dataset._parquet_handles


def test_observation_from_dict_accepts_uint8_front_history_images():
    """The offline packs emit uint8 history frames; Observation is Float-typed."""
    model_module = pytest.importorskip("openpi.models.model")
    torch = pytest.importorskip("torch")

    batch = {
        "image": {"left_wrist_0_rgb": torch.zeros(2, 4, 4, 3, dtype=torch.uint8)},
        "image_mask": {"left_wrist_0_rgb": torch.zeros(2, dtype=torch.bool)},
        "state": torch.zeros(2, 32),
        "state_pose_valid": torch.tensor([[False, True, True], [True, True, True]]),
        "front_history_images": torch.full((2, 4, 4, 4, 3), 255, dtype=torch.uint8),
        "front_history_masks": torch.ones(2, 4, dtype=torch.bool),
    }
    obs = model_module.Observation.from_dict(batch)

    assert obs.front_history_images.dtype == torch.float32
    # uint8 255 maps to +1.0 in the [-1, 1] convention used for wrist images.
    assert torch.allclose(obs.front_history_images, torch.ones_like(obs.front_history_images))
    # Channels-last layout is preserved for the camera-token path.
    assert obs.front_history_images.shape == (2, 4, 4, 4, 3)
    assert torch.equal(obs.state_pose_valid, batch["state_pose_valid"])


def test_observation_accepts_per_timestep_action_dim_mask():
    """Human packs emit action_dim_mask as [B, H, D], not [B, D]."""
    model_module = pytest.importorskip("openpi.models.model")
    torch = pytest.importorskip("torch")

    batch = {
        "image": {"left_wrist_0_rgb": torch.zeros(2, 3, 4, 4)},
        "image_mask": {"left_wrist_0_rgb": torch.zeros(2, dtype=torch.bool)},
        "state": torch.zeros(2, 32),
        "action_dim_mask": torch.ones(2, 50, 32, dtype=torch.bool),
        "action_time_valid_mask": torch.ones(2, 50, dtype=torch.bool),
    }
    obs = model_module.Observation.from_dict(batch)
    assert obs.action_dim_mask.shape == (2, 50, 32)

    flat = dict(batch)
    flat["action_dim_mask"] = torch.ones(2, 32, dtype=torch.bool)
    assert model_module.Observation.from_dict(flat).action_dim_mask.shape == (2, 32)


def test_episode_instruction_never_falls_back_to_pack_columns(monkeypatch, tmp_path):
    """episode_instruction must come from the sidecar only, with no fallback.

    A pack row may carry its own episode_instruction / episode_task_description
    columns. Using them when the sidecar join fails would silently swap in a
    different annotation semantics and hide the broken join.
    """
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    row = _row(task="grasp the mug")
    row["episode_instruction"] = "make breakfast"
    row["episode_task_description"] = "prepare a meal"
    pack_path = _write_pack(
        tmp_path,
        [row],
        # Sidecar joins on a different episode_index, so this row has no
        # sidecar-provided episode instruction.
        [{"repo_id": "EgoDex_lerobot_active", "episode_index": 999, "episode_instruction": "make breakfast"}],
    )
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=False,
        fake_images=True,
        max_filter_attempts=2,
    )
    with pytest.raises(RuntimeError, match="missing_episode_instruction"):
        dataset[0]


def test_degenerate_and_missing_instructions_use_distinct_reasons(monkeypatch, tmp_path):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    pack_path = _write_pack(
        tmp_path,
        [_row(task="same text")],
        [{"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "Same Text"}],
    )
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=False,
        fake_images=True,
        max_filter_attempts=2,
    )
    with pytest.raises(RuntimeError, match="degenerate_episode_instruction_equals_task"):
        dataset[0]


def test_parse_source_views_and_intersection_with_global_views():
    parsed = human_vlm_dataset.parse_source_views("EgoDex=subtask+fast,VITRA=fast")
    assert parsed == {
        "EgoDex": {human_vlm_dataset.VIEW_SUBTASK: 0.5, human_vlm_dataset.VIEW_FAST: 0.5},
        "VITRA": {human_vlm_dataset.VIEW_FAST: 1.0},
    }
    weighted = human_vlm_dataset.parse_source_views("EgoDex=subtask:3+fast:1")
    assert weighted["EgoDex"] == {human_vlm_dataset.VIEW_SUBTASK: 0.75, human_vlm_dataset.VIEW_FAST: 0.25}
    assert human_vlm_dataset.parse_source_views(None) == {}
    with pytest.raises(ValueError, match="unknown view"):
        human_vlm_dataset.parse_source_views("EgoDex=nosuchview")
    with pytest.raises(ValueError, match="source_views entry"):
        human_vlm_dataset.parse_source_views("EgoDex")


def _two_source_packs(tmp_path, n_rows=4):
    """Two sources with several anchors each.

    More than one anchor per source matters: views split a source's anchor
    budget, so a single-anchor source cannot express a two-view split.
    """
    egodex = _write_pack(
        tmp_path / "a",
        [_row(task=f"egodex sub {i}", episode_index=i, repo_id="EgoDex_lerobot_active") for i in range(n_rows)],
        [
            {"repo_id": "EgoDex_lerobot_active", "episode_index": i, "episode_instruction": f"high level dex {i}"}
            for i in range(n_rows)
        ],
    )
    vitra = _write_pack(
        tmp_path / "b",
        [_row(task=f"vitra sub {i}", episode_index=i, repo_id="shard-00") for i in range(n_rows)],
        [
            {"repo_id": "shard-00", "episode_index": i, "episode_instruction": f"vitra sub {i}"}
            for i in range(n_rows)
        ],
        pack_name="VITRA_stride3_fovfix_trainpath_20260724",
        repo_id="shard-00",
    )
    return egodex, vitra


def test_source_views_restrict_which_views_a_source_emits(monkeypatch, tmp_path):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    egodex, vitra = _two_source_packs(tmp_path)
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=f"{egodex},{vitra}",
        source_weights="EgoDex=1,VITRA=1",
        # VITRA has no genuine episode-level instruction, so it may only do FAST.
        source_views="EgoDex=subtask+fast,VITRA=fast",
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=True,
        fake_images=True,
    )
    assert set(dataset._views_by_source["EgoDex"]) == {human_vlm_dataset.VIEW_SUBTASK, human_vlm_dataset.VIEW_FAST}
    assert dataset._views_by_source["VITRA"] == [human_vlm_dataset.VIEW_FAST]

    seen = _routed_pairs(dataset)
    assert ("VITRA", "high_level") not in seen
    assert ("VITRA", "low_level") in seen
    assert ("EgoDex", "high_level") in seen and ("EgoDex", "low_level") in seen


def test_source_with_only_globally_disabled_view_is_excluded(monkeypatch, tmp_path):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    egodex, vitra = _two_source_packs(tmp_path)
    # FAST off globally: VITRA (fast-only) must drop out rather than emit
    # subtask samples it cannot satisfy.
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=f"{egodex},{vitra}",
        source_weights="EgoDex=1,VITRA=1",
        source_views="EgoDex=subtask,VITRA=fast",
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=False,
        fake_images=True,
    )
    assert dataset._views_by_source["VITRA"] == []
    assert "VITRA" not in dataset._view_schedule_by_source
    assert set(dataset._view_schedule_by_source) == {"EgoDex"}


def test_index_space_covers_anchors_times_span(monkeypatch, tmp_path):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    egodex, vitra = _two_source_packs(tmp_path)
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=f"{egodex},{vitra}",
        source_weights="EgoDex=1,VITRA=1",
        source_views="EgoDex=subtask+fast,VITRA=fast",
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=True,
        fake_images=True,
    )
    # Views SPLIT each source's anchor budget rather than multiplying it, so
    # the totals equal the anchor counts regardless of how many views a
    # source has.
    anchors = dataset._source_cum_rows["EgoDex"][-1] + dataset._source_cum_rows["VITRA"][-1]
    # Cycle-dependent routing multiplies the index space by the schedule span,
    # which is applied to EVERY source and so leaves the mixture untouched.
    assert dataset.num_frames == anchors * dataset.view_cycle_span
    assert sum(dataset.index_space_counts().values()) == dataset.num_frames


def test_physical_once_routes_each_anchor_once_without_low_only_duplication(monkeypatch, tmp_path):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    egodex, vitra = _two_source_packs(tmp_path)
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=f"{egodex},{vitra}",
        source_weights="EgoDex=1,VITRA=1",
        source_views="EgoDex=subtask+fast,VITRA=fast",
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=True,
        fake_images=True,
        view_cycle_mode="physical_once",
    )
    anchors_by_source = {source: rows[-1] for source, rows in dataset._source_cum_rows.items()}
    counts = dataset.index_space_counts()
    assert dataset.view_cycle_span == 1
    assert dataset.num_frames == sum(anchors_by_source.values())
    assert sum(n for (source, _view), n in counts.items() if source == "EgoDex") == anchors_by_source["EgoDex"]
    assert counts[("VITRA", human_vlm_dataset.VIEW_FAST)] == anchors_by_source["VITRA"]


def test_physical_once_phase_complements_multiview_source(monkeypatch, tmp_path):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    egodex, vitra = _two_source_packs(tmp_path)
    common = {
        "pack_paths": f"{egodex},{vitra}",
        "source_weights": "EgoDex=1,VITRA=1",
        "source_views": "EgoDex=subtask+fast,VITRA=fast",
        "norm_stats_path": str(_write_norm_stats(tmp_path)),
        "max_token_len": 4,
        "enable_subtask_view": True,
        "enable_fast_view": True,
        "fake_images": True,
        "view_cycle_mode": "physical_once",
    }
    phase0 = human_vlm_dataset.HumanVLMPairedOfflineDataset(view_route_phase=0, **common)
    phase1 = human_vlm_dataset.HumanVLMPairedOfflineDataset(view_route_phase=1, **common)
    for slot in range(phase0.num_anchor_rows):
        source0, view0, _ = phase0.route_view(slot, 0)
        source1, view1, _ = phase1.route_view(slot, 0)
        assert source0 == source1
        if source0 == "EgoDex":
            assert view0 != view1
        else:
            assert view0 == view1 == human_vlm_dataset.VIEW_FAST


def test_two_view_source_does_not_double_its_share_of_the_mixture(monkeypatch, tmp_path):
    """Views split a source's anchor budget; they must not multiply it.

    With `anchors x views`, a two-view source silently got twice the sampling
    weight its configured source_weight asked for.
    """
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    egodex, vitra = _two_source_packs(tmp_path)
    common = {
        "pack_paths": f"{egodex},{vitra}",
        "source_weights": "EgoDex=1,VITRA=1",
        "norm_stats_path": str(_write_norm_stats(tmp_path)),
        "max_token_len": 4,
        "enable_subtask_view": True,
        "enable_fast_view": True,
        "fake_images": True,
    }
    both = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        source_views="EgoDex=subtask+fast,VITRA=fast", **common
    )

    def share(dataset, source):
        counts = dataset.index_space_counts()
        total = sum(counts.values())
        return sum(n for (s, _v), n in counts.items() if s == source) / total

    # EgoDex has two views and VITRA one, yet their shares stay equal because
    # source_weights (1:1) alone decides the split.
    assert share(both, "EgoDex") == share(both, "VITRA")

    single = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        source_views="EgoDex=subtask,VITRA=fast", **common
    )
    assert share(both, "EgoDex") == share(single, "EgoDex")


def test_view_weights_split_a_source_budget_proportionally(monkeypatch, tmp_path):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    rows = [_row(task=f"t{i}", episode_index=i) for i in range(20)]
    sidecar = [
        {"repo_id": "EgoDex_lerobot_active", "episode_index": i, "episode_instruction": f"high {i}"}
        for i in range(20)
    ]
    pack = _write_pack(tmp_path, rows, sidecar)
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack),
        source_views="EgoDex=subtask:3+fast:1",
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=True,
        fake_images=True,
    )
    counts = dataset.index_space_counts()
    sizes = {human_vlm_dataset._view_name(v): n for (_s, v), n in counts.items()}
    assert sizes["high_level"] + sizes["low_level"] == dataset.num_frames
    # 3:1 -> schedule span 4, so 20 anchors x 4 cycles = 80 samples, 60/20.
    assert sizes["high_level"] / dataset.num_frames == 0.75
    assert sizes["low_level"] / dataset.num_frames == 0.25


def test_runtime_stats_report_source_and_view_ratios_separately(monkeypatch, tmp_path):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    egodex, vitra = _two_source_packs(tmp_path)
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=f"{egodex},{vitra}",
        source_weights="EgoDex=1,VITRA=1",
        source_views="EgoDex=subtask,VITRA=fast",
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=True,
        fake_images=True,
    )
    for idx in range(len(dataset)):
        dataset[idx]
    stats = dataset.runtime_stats()
    assert set(stats["realized_source_ratio"]) == {"EgoDex", "VITRA"}
    assert set(stats["realized_view_ratio"]) == {"high_level", "low_level"}
    assert abs(sum(stats["realized_source_ratio"].values()) - 1.0) < 1e-9
    assert abs(sum(stats["realized_view_ratio"].values()) - 1.0) < 1e-9
    assert stats["configured_view_weights_by_source"]["VITRA"] == {"low_level": 1.0}


def test_norm_stats_without_action_layout_fail_closed(monkeypatch, tmp_path):
    """Stats predating the canonical reordering must be rejected, not used.

    They are per-dimension, so applying head-pose quantiles to left-wrist
    values would silently mis-normalize every sample.
    """
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    pack_path = _write_pack(
        tmp_path,
        [_row(task="pick the cup")],
        [{"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "make breakfast"}],
    )
    stale = tmp_path / "stale_stats.json"
    stale.write_text(
        json.dumps(
            {
                "state_q01": np.zeros(21).tolist(),
                "state_q99": np.ones(21).tolist(),
                "action_q01": np.zeros(21).tolist(),
                "action_q99": np.ones(21).tolist(),
            }
        )
    )
    with pytest.raises(ValueError, match="action_layout"):
        human_vlm_dataset.HumanVLMPairedOfflineDataset(
            pack_paths=str(pack_path),
            norm_stats_path=str(stale),
            max_token_len=4,
            enable_subtask_view=True,
            enable_fast_view=False,
            fake_images=True,
        )


def test_dataset_emits_canonical_left_right_head_actions(monkeypatch, tmp_path):
    """End-to-end: a packed [head, left, right] row leaves as [left, right, head]."""
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    packed = np.zeros((50, 21), dtype=np.float32)
    packed[:, 0:7] = 1.0    # head
    packed[:, 7:14] = 2.0   # left
    packed[:, 14:21] = 3.0  # right
    row = _row(task="pick the cup")
    row["action"] = packed.reshape(-1).tolist()
    pack_path = _write_pack(
        tmp_path,
        [row],
        [{"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "make breakfast"}],
    )
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=False,
        fake_images=True,
    )
    actions = np.asarray(dataset[0]["actions"])
    # Only the translation triplet is compared: the quaternion part of each
    # block is unit-normalized by the pose contract, so it no longer carries
    # the raw fill value.
    assert np.allclose(actions[:, 0:3], 2.0)    # left first
    assert np.allclose(actions[:, 7:10], 3.0)   # then right
    assert np.allclose(actions[:, 14:17], 1.0)  # head last
    assert np.allclose(actions[:, 21:], 0.0)    # padding untouched


def _low_level_dataset(monkeypatch, tmp_path, **kwargs):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    pack_path = _write_pack(
        tmp_path,
        [_row(task="grasp the mug")],
        [{"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "make breakfast"}],
    )
    return human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        source_views="EgoDex=low_level",
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=True,
        fake_images=True,
        **kwargs,
    )


def test_low_level_sample_emits_condition_without_fast_targets(monkeypatch, tmp_path):
    """The Action Expert's conditioning must not contain the GT FAST tokens.

    If the answer sits in the prefix the expert attends to, the flow loss is
    trivially satisfiable and the model learns nothing about actions.
    """
    dataset = _low_level_dataset(monkeypatch, tmp_path)
    item = dataset[0]

    assert bool(item["has_flow_target"])
    fast_targets = set(np.asarray(item["tokenized_prompt"])[np.asarray(item["token_loss_mask"])].tolist())
    condition = np.asarray(item["flow_condition_tokens"])
    condition_valid = set(condition[np.asarray(item["flow_condition_mask"])].tolist())
    assert fast_targets, "the fast view should supervise at least one token"
    assert not (fast_targets & condition_valid), "FAST target ids leaked into the flow conditioning"
    # Conditioning supervises nothing by itself.
    assert not np.asarray(item["flow_condition_ar_mask"]).any()


def test_high_level_sample_has_no_flow_target(monkeypatch, tmp_path):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    pack_path = _write_pack(
        tmp_path,
        [_row(task="grasp the mug")],
        [{"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "make breakfast"}],
    )
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        source_views="EgoDex=high_level",
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=False,
        fake_images=True,
    )
    item = dataset[0]
    assert not bool(item["has_flow_target"])
    assert item["vlm_view_type"].item() == human_vlm_dataset.VIEW_HIGH_LEVEL


def test_flow_actions_are_normalized_horizon_length_and_zero_padded(monkeypatch, tmp_path):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    raw = np.zeros((50, 21), dtype=np.float32)
    raw[:, :] = 5.0  # inside the [0, 10] norm range of the test stats
    row = _row(task="grasp the mug")
    row["action"] = raw.reshape(-1).tolist()
    pack_path = _write_pack(
        tmp_path,
        [row],
        [{"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "make breakfast"}],
    )
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        source_views="EgoDex=low_level",
        norm_stats_path=str(_write_norm_stats(tmp_path, action_q01=np.zeros(21), action_q99=np.ones(21) * 10.0)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=True,
        fast_horizon=10,
        fake_images=True,
    )
    item = dataset[0]
    flow = np.asarray(item["flow_actions"])
    mask = np.asarray(item["flow_action_mask"])

    assert flow.shape == (10, human_vlm_dataset.MODEL_ACTION_DIM)
    assert np.all(flow >= -1.0) and np.all(flow <= 1.0)
    # Translation dims: 5.0 is the midpoint of the [0, 10] norm range.
    for block_start in (0, 7, 14):
        assert np.allclose(flow[:, block_start : block_start + 3], 0.0, atol=1e-6)
    assert np.allclose(flow[:, 21:], 0.0)  # padding is zero
    assert mask[:, :21].all() and not mask[:, 21:].any()  # padded dims masked false


def test_fast_and_flow_consume_the_same_canonical_trajectory(monkeypatch, tmp_path):
    """Both heads must supervise the identical h-step canonical chunk."""
    dataset = _low_level_dataset(monkeypatch, tmp_path, fast_horizon=10)
    item = dataset[0]
    fast_actions_seen = _FakeHumanVLMTokenizer.fast_calls[-1]
    flow = np.asarray(item["flow_actions"])
    # The fake tokenizer records (subtask, horizon); the horizon must match the
    # flow target's time dimension.
    assert fast_actions_seen[1] == flow.shape[0] == 10


def test_partial_fast_horizon_uses_valid_prefix_and_masks_flow_tail(monkeypatch, tmp_path):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    _FakeHumanVLMTokenizer.fast_calls = []
    mask = np.ones((50, 21), dtype=np.bool_)
    mask[12:, :] = False
    pack_path = _write_pack(
        tmp_path,
        [_row(task="grasp the mug", action_mask=mask)],
        [{"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "make breakfast"}],
    )
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        source_views="EgoDex=low_level",
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=True,
        fast_horizon=30,
        allow_partial_fast_horizon=True,
        fake_images=True,
    )

    item = dataset[0]
    assert _FakeHumanVLMTokenizer.fast_calls[-1][1] == 12
    assert item["flow_actions"].shape[0] == 30
    assert np.asarray(item["flow_action_mask"])[:12, :21].all()
    assert not np.asarray(item["flow_action_mask"])[12:].any()


def test_sample_carries_source_id_for_cross_worker_aggregation(monkeypatch, tmp_path):
    dataset = _low_level_dataset(monkeypatch, tmp_path)
    item = dataset[0]
    assert int(item["source_id"]) == human_vlm_dataset.SOURCE_IDS["EgoDex"]


def test_egoprostandard_pack_enables_optional_wrist_images(monkeypatch, tmp_path):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    row = _row(repo_id="LightwheelAI_EgoPro/EgoProStandard")
    row.update(
        {
            "left_wrist_video_relpath": "/virtual/episode.mcap::left_wrist",
            "right_wrist_video_relpath": "/virtual/episode.mcap::right_wrist",
            "left_wrist_frame_index": 16,
            "right_wrist_frame_index": 16,
        }
    )
    pack_path = _write_pack(
        tmp_path,
        [row],
        [
            {
                "repo_id": "LightwheelAI_EgoPro/EgoProStandard",
                "episode_index": 0,
                "episode_instruction": "make breakfast",
            }
        ],
        pack_name="EgoProStandard_pilot",
        repo_id="shard-000",
    )
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        norm_stats_path=str(_write_norm_stats(tmp_path / "norm")),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=False,
        fake_images=True,
    )

    item = dataset[0]
    assert int(item["source_id"]) == human_vlm_dataset.SOURCE_IDS["EgoProStandard"]
    assert bool(item["image_mask"]["left_wrist_0_rgb"])
    assert bool(item["image_mask"]["right_wrist_0_rgb"])


def _identity_extrinsics_rows(history_mask):
    """Extrinsics where the earliest valid slot is identity (the ref frame)."""
    ex = np.zeros((4, 3, 4), dtype=np.float32)
    for slot in range(4):
        ex[slot, :3, :3] = np.eye(3)
        if history_mask[slot]:
            ex[slot, :3, 3] = [0.1 * slot, 0.0, 0.0]
    first_valid = int(np.argmax(history_mask))
    ex[first_valid, :3, 3] = 0.0  # identity at the reference frame
    return ex.reshape(-1).tolist()


def test_reference_frame_invariant_holds_with_leading_history_padding(monkeypatch, tmp_path):
    """History padding must not move the reference frame.

    The packs express camera_extrinsics, state and future actions in the
    earliest VALID history frame. With leading padding such as
    [False, False, True, True] the reference is slot 2, not slot 0.
    """
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    history_mask = [False, False, True, True]
    row = _row(task="pick the cup")
    row["camera_token_history_mask"] = history_mask
    row["observation.camera_extrinsics"] = _identity_extrinsics_rows(history_mask)
    pack_path = _write_pack(
        tmp_path,
        [row],
        [{"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "make breakfast"}],
    )
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=False,
        fake_images=True,
    )
    item = dataset[0]

    valid = np.asarray(item["camera_pose_valid"])
    extrinsics = np.asarray(item["camera_extrinsics"])
    assert valid.tolist() == history_mask
    first_valid = int(np.argmax(valid))
    assert first_valid == 2, "the reference is the earliest VALID slot, not slot 0"
    # The reference slot is the identity transform, i.e. the frame everything
    # else is expressed in.
    assert np.allclose(extrinsics[first_valid][:3, :3], np.eye(3), atol=1e-6)
    assert np.allclose(extrinsics[first_valid][:3, 3], 0.0, atol=1e-6)
    # Invalid leading slots stay masked out and must not become the reference.
    assert not valid[:first_valid].any()


def test_actions_are_sign_unwrapped_against_the_state(monkeypatch, tmp_path):
    """Emitted actions must be temporally sign-continuous, anchored on state."""
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    packed_state = np.zeros(21, dtype=np.float32)
    packed_actions = np.zeros((50, 21), dtype=np.float32)
    for block in range(3):
        s = block * 7
        packed_state[s + 6] = 1.0  # w = 1
        # Alternate the written sign every frame; the rotation never changes.
        for t in range(50):
            packed_actions[t, s + 6] = 1.0 if t % 2 == 0 else -1.0

    row = _row(task="pick the cup")
    row["observation.state"] = packed_state.tolist()
    row["action"] = packed_actions.reshape(-1).tolist()
    pack_path = _write_pack(
        tmp_path,
        [row],
        [{"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "make breakfast"}],
    )
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=False,
        fake_images=True,
    )
    actions = np.asarray(dataset[0]["actions"])
    for block in range(3):
        q = actions[:, block * 7 + 3 : block * 7 + 7]
        dots = np.sum(q[1:] * q[:-1], axis=-1)
        assert (dots >= 0).all(), f"block {block} still contains a temporal sign flip"


def test_fast_and_flow_share_the_identical_mask_aware_unwrapped_trajectory(monkeypatch, tmp_path):
    """Both heads must regress the same h10 chunk, including its signs.

    The fake tokenizer records the exact array handed to tokenize_fast_action;
    it must match the emitted flow target on every valid dimension.
    """
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    _FakeHumanVLMTokenizer.fast_action_arrays = []

    rng = np.random.default_rng(0)
    packed_actions = rng.normal(size=(50, 21)).astype(np.float32)
    packed_state = rng.normal(size=21).astype(np.float32)
    # The low-level view requires a fully valid h10 chunk, so the mask hole
    # sits after the horizon; it still exercises the mask-aware chain over the
    # full 50-frame trajectory the dataset emits.
    mask = np.ones((50, 21), dtype=bool)
    mask[30, :] = False

    row = _row(task="grasp the mug")
    row["action"] = packed_actions.reshape(-1).tolist()
    row["observation.state"] = packed_state.tolist()
    row["action_loss_mask"] = mask.reshape(-1).tolist()
    pack_path = _write_pack(
        tmp_path,
        [row],
        [{"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "make breakfast"}],
    )
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        source_views="EgoDex=low_level",
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=8,
        enable_subtask_view=True,
        enable_fast_view=True,
        fast_horizon=10,
        fake_images=True,
    )
    item = dataset[0]

    fast_input = _FakeHumanVLMTokenizer.fast_action_arrays[-1]
    flow = np.asarray(item["flow_actions"])[:, :21]
    flow_mask = np.asarray(item["flow_action_mask"])[:, :21]
    assert fast_input.shape == flow.shape
    assert np.allclose(fast_input[flow_mask], flow[flow_mask], atol=1e-6)


def test_emitted_actions_have_zero_flip_rate_over_valid_frames(monkeypatch, tmp_path):
    monkeypatch.setattr(human_vlm_dataset, "HumanVLMDiscreteTokenizer", _FakeHumanVLMTokenizer)
    packed_state = np.zeros(21, dtype=np.float32)
    packed_actions = np.zeros((50, 21), dtype=np.float32)
    for block in range(3):
        s = block * 7
        packed_state[s + 6] = 1.0
        for t in range(50):
            packed_actions[t, s + 6] = 1.0 if t % 2 == 0 else -1.0
    mask = np.ones((50, 21), dtype=bool)
    mask[5, :] = False  # a hole that must not become a sign reference

    row = _row(task="grasp the mug")
    row["action"] = packed_actions.reshape(-1).tolist()
    row["observation.state"] = packed_state.tolist()
    row["action_loss_mask"] = mask.reshape(-1).tolist()
    pack_path = _write_pack(
        tmp_path,
        [row],
        [{"repo_id": "EgoDex_lerobot_active", "episode_index": 0, "episode_instruction": "make breakfast"}],
    )
    dataset = human_vlm_dataset.HumanVLMPairedOfflineDataset(
        pack_paths=str(pack_path),
        norm_stats_path=str(_write_norm_stats(tmp_path)),
        max_token_len=4,
        enable_subtask_view=True,
        enable_fast_view=False,
        fake_images=True,
    )
    actions = np.asarray(dataset[0]["actions"])
    valid_idx = [t for t in range(50) if mask[t].all()]
    for block in range(3):
        q = actions[valid_idx][:, block * 7 + 3 : block * 7 + 7]
        dots = np.sum(q[1:] * q[:-1], axis=-1)
        assert (dots >= 0).all(), f"block {block} flips across valid frames"


def test_observation_accepts_flow_fields_alongside_history_images():
    """Flow and history-image dims must not share jaxtyping symbols.

    front_history_images binds the frame height; if the flow horizon reused
    that symbol, jaxtyping would demand horizon == image height and reject
    every real batch.
    """
    model_module = pytest.importorskip("openpi.models.model")
    torch = pytest.importorskip("torch")

    batch = {
        "image": {"left_wrist_0_rgb": torch.zeros(2, 3, 224, 224)},
        "image_mask": {"left_wrist_0_rgb": torch.zeros(2, dtype=torch.bool)},
        "state": torch.zeros(2, 32),
        # 4 history frames at 224x224x3 ...
        "front_history_images": torch.zeros(2, 4, 224, 224, 3),
        "front_history_masks": torch.ones(2, 4, dtype=torch.bool),
        # ... alongside a 10-step flow target.
        "flow_actions": torch.zeros(2, 10, 32),
        "flow_action_mask": torch.ones(2, 10, 32, dtype=torch.bool),
        "flow_time_valid_mask": torch.ones(2, 10, dtype=torch.bool),
        "has_flow_target": torch.ones(2, dtype=torch.bool),
        "source_id": torch.zeros(2, dtype=torch.long),
    }
    obs = model_module.Observation.from_dict(batch)

    assert obs.flow_actions.shape == (2, 10, 32)
    assert obs.front_history_images.shape == (2, 4, 224, 224, 3)
    assert obs.flow_time_valid_mask.shape == (2, 10)
    assert int(obs.source_id[0]) == 0


def test_joint_objective_can_slice_a_real_frozen_observation():
    """Batch splitting must work on the real Observation, not just a stub.

    Observation is a frozen dataclass: assigning to a field raises
    FrozenInstanceError. The tiny test double is a plain object, so a
    setattr-based implementation passes there and fails on real data.
    """
    model_module = pytest.importorskip("openpi.models.model")
    pi0_pytorch = pytest.importorskip("openpi.models_pytorch.pi0_pytorch")
    torch = pytest.importorskip("torch")

    batch = {
        "image": {"left_wrist_0_rgb": torch.zeros(4, 3, 8, 8)},
        "image_mask": {"left_wrist_0_rgb": torch.zeros(4, dtype=torch.bool)},
        "state": torch.zeros(4, 32),
        "vlm_view_type": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "source_id": torch.tensor([0, 2, 1, 2], dtype=torch.long),
        "flow_actions": torch.zeros(4, 10, 32),
        "flow_action_mask": torch.ones(4, 10, 32, dtype=torch.bool),
        "flow_time_valid_mask": torch.ones(4, 10, dtype=torch.bool),
    }
    obs = model_module.Observation.from_dict(batch)
    assert obs.__dataclass_params__.frozen, "guard: Observation should stay frozen"

    low_idx = torch.tensor([1, 3])
    selected = pi0_pytorch.PI0Pytorch._select_observation(obs, low_idx)

    assert selected.state.shape == (2, 32)
    assert selected.vlm_view_type.tolist() == [1, 1]
    assert selected.source_id.tolist() == [2, 2]
    assert selected.flow_actions.shape == (2, 10, 32)
    assert selected.images["left_wrist_0_rgb"].shape == (2, 3, 8, 8)
    # The original is untouched.
    assert obs.state.shape == (4, 32)

    replaced = pi0_pytorch.PI0Pytorch._observation_replace(obs, state=torch.ones(4, 32))
    assert replaced.state.sum() == 4 * 32
    assert obs.state.sum() == 0
