"""Regression tests for the episode-disjoint validation split."""

import pytest

hvd = pytest.importorskip("openpi.training.human_vlm_dataset")

TRAIN, VAL, ALL = hvd.SPLIT_TRAIN, hvd.SPLIT_VAL, hvd.SPLIT_ALL


def test_bucket_is_deterministic_across_calls():
    a = hvd.episode_split_bucket("EgoDex_lerobot_active", 17)
    assert a == hvd.episode_split_bucket("EgoDex_lerobot_active", 17)
    assert 0 <= a < 10000


def test_repo_id_participates_in_the_key():
    """Same episode_index in different repos must not collide by construction."""
    assert hvd.episode_split_bucket("A", 5) != hvd.episode_split_bucket("B", 5)


def test_train_and_val_are_disjoint_and_exhaustive():
    val_bp = 200  # 2%
    eps = [("EgoDex_lerobot_active", i) for i in range(5000)]
    train = {e for e in eps if hvd.episode_in_split(*e, TRAIN, val_bp)}
    val = {e for e in eps if hvd.episode_in_split(*e, VAL, val_bp)}
    assert train & val == set(), "an episode leaked into both splits"
    assert train | val == set(eps)


def test_val_fraction_is_close_to_requested():
    eps = [("EgoDex_lerobot_active", i) for i in range(20000)]
    for bp, tol in ((100, 0.004), (200, 0.005), (500, 0.008)):
        frac = sum(hvd.episode_in_split(*e, VAL, bp) for e in eps) / len(eps)
        assert abs(frac - bp / 10000) < tol, f"{bp}bp -> {frac:.4f}"


def test_all_split_keeps_everything():
    eps = [("EgoLive", i) for i in range(500)]
    assert all(hvd.episode_in_split(*e, ALL, 500) for e in eps)


def test_split_is_stable_when_val_fraction_changes():
    """Growing the val set must only ADD episodes, never reshuffle them.

    Otherwise a later change to val_bp silently moves previously-trained
    episodes into validation and invalidates every earlier comparison.
    """
    eps = [("VITRA", i) for i in range(4000)]
    small = {e for e in eps if hvd.episode_in_split(*e, VAL, 100)}
    large = {e for e in eps if hvd.episode_in_split(*e, VAL, 300)}
    assert small <= large


def test_every_anchor_of_an_episode_lands_on_one_side():
    """The property an anchor-level split would violate."""
    sides = {
        hvd.episode_in_split("EgoDex_lerobot_active", 42, VAL, 200) for _ in range(100)
    }
    assert len(sides) == 1
