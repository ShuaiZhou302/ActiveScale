import openpi.training.pytorch_checkpoints as ckpt


def test_final_step_saves_once_without_off_by_one():
    """The last checkpoint is at num_train_steps, not num_train_steps - 1.

    `global_step` is post-increment, so the old `== num_train_steps - 1` rule
    fired on the second-to-last step and wrote a redundant extra checkpoint
    (observed as steps 10, 19 and 20 in a 20-step run with save_interval 10).
    """
    saved = [s for s in range(1, 21) if ckpt.should_save(s, save_interval=10, num_train_steps=20)]
    assert saved == [10, 20]


def test_should_save_covers_final_step_not_on_interval():
    saved = [s for s in range(1, 26) if ckpt.should_save(s, save_interval=10, num_train_steps=25)]
    assert saved == [10, 20, 25]


def test_should_save_ignores_step_zero():
    assert not ckpt.should_save(0, save_interval=10, num_train_steps=20)


def test_partition_keeps_latest_two_and_period_multiples():
    steps = [10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000]
    resumable, archival, delete = ckpt.partition_checkpoints(steps, keep_last=2, keep_period=50_000)
    assert resumable == [60_000, 70_000]
    assert archival == [50_000]
    assert delete == [10_000, 20_000, 30_000, 40_000]


def test_partition_never_deletes_a_newest_checkpoint_that_is_also_periodic():
    resumable, archival, delete = ckpt.partition_checkpoints(
        [40_000, 50_000], keep_last=2, keep_period=50_000
    )
    assert resumable == [40_000, 50_000]
    assert archival == []
    assert delete == []


def test_partition_without_keep_period_keeps_only_latest():
    resumable, archival, delete = ckpt.partition_checkpoints([1, 2, 3, 4], keep_last=2, keep_period=None)
    assert resumable == [3, 4]
    assert archival == []
    assert delete == [1, 2]


def _make_ckpt(root, step, *, with_optimizer=True):
    d = root / str(step)
    d.mkdir(parents=True)
    (d / "model.safetensors").write_text("model")
    (d / "metadata.pt").write_text("meta")
    if with_optimizer:
        (d / "optimizer.pt").write_text("optimizer")
    return d


def test_prune_deletes_old_and_strips_archival_optimizer(tmp_path):
    for step in (10_000, 20_000, 50_000, 60_000, 70_000):
        _make_ckpt(tmp_path, step)

    report = ckpt.prune_checkpoints(tmp_path, keep_last=2, keep_period=50_000)

    assert report["resumable"] == [60_000, 70_000]
    assert report["archival"] == [50_000]
    assert report["deleted"] == [10_000, 20_000]
    assert report["optimizer_stripped"] == [50_000]

    assert not (tmp_path / "10000").exists()
    assert not (tmp_path / "20000").exists()
    # Archival checkpoint keeps weights but drops the large optimizer shard.
    assert (tmp_path / "50000" / "model.safetensors").is_file()
    assert not (tmp_path / "50000" / "optimizer.pt").exists()
    # The newest two stay fully resumable.
    for step in (60_000, 70_000):
        assert (tmp_path / str(step) / "optimizer.pt").is_file()


def test_prune_is_idempotent(tmp_path):
    for step in (10, 20, 30):
        _make_ckpt(tmp_path, step)
    first = ckpt.prune_checkpoints(tmp_path, keep_last=2, keep_period=None)
    second = ckpt.prune_checkpoints(tmp_path, keep_last=2, keep_period=None)
    assert first["deleted"] == [10]
    assert second["deleted"] == []
    assert ckpt.existing_checkpoint_steps(tmp_path) == [20, 30]


def test_prune_ignores_tmp_and_non_numeric_dirs(tmp_path):
    _make_ckpt(tmp_path, 100)
    (tmp_path / "tmp_200").mkdir()
    (tmp_path / "assets").mkdir()
    report = ckpt.prune_checkpoints(tmp_path, keep_last=1, keep_period=None)
    assert report["deleted"] == []
    assert (tmp_path / "tmp_200").is_dir()
    assert (tmp_path / "assets").is_dir()


def test_resume_uses_latest_checkpoint_which_retention_always_keeps(tmp_path):
    """Retention must never strip the checkpoint a resume would pick up."""
    for step in (10, 20, 30, 40):
        _make_ckpt(tmp_path, step)
    ckpt.prune_checkpoints(tmp_path, keep_last=2, keep_period=None)
    steps = ckpt.existing_checkpoint_steps(tmp_path)
    latest = max(steps)
    assert latest == 40
    assert (tmp_path / str(latest) / "optimizer.pt").is_file()
    assert (tmp_path / str(latest) / "model.safetensors").is_file()
