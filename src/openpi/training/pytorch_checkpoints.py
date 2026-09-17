"""Checkpoint scheduling and retention for the PyTorch training loop.

The policy is deliberately expressed as pure functions so it can be tested
without a model, a GPU or any disk state.

Retention keeps:
  * the newest `keep_last` checkpoints (these can resume training), and
  * every checkpoint whose step is divisible by `keep_period` (archival).

Archival checkpoints that are no longer among the newest `keep_last` do not
need optimizer state to be useful, and the optimizer shard is by far the
largest file (13.5 GiB of a 21 GiB pi0.5 checkpoint), so it is dropped.
"""

from __future__ import annotations

import logging
from pathlib import Path
import shutil

LOGGER = logging.getLogger(__name__)

OPTIMIZER_FILENAME = "optimizer.pt"


def should_save(global_step: int, *, save_interval: int, num_train_steps: int) -> bool:
    """Whether a checkpoint is due after completing `global_step` steps.

    `global_step` is post-increment: it equals the number of optimizer steps
    completed so far, so the final step is `num_train_steps` (not
    `num_train_steps - 1`, which would checkpoint the second-to-last step and
    write a redundant extra checkpoint).
    """
    if global_step <= 0:
        return False
    if save_interval > 0 and global_step % save_interval == 0:
        return True
    return global_step == num_train_steps


def partition_checkpoints(
    steps: list[int], *, keep_last: int = 2, keep_period: int | None = None
) -> tuple[list[int], list[int], list[int]]:
    """Split existing checkpoint steps into (resumable, archival, delete).

    resumable: newest `keep_last` steps, kept whole.
    archival:  kept only because step % keep_period == 0; optimizer dropped.
    delete:    everything else.
    """
    ordered = sorted(set(int(s) for s in steps))
    keep_last = max(0, int(keep_last))
    newest = set(ordered[-keep_last:]) if keep_last else set()
    periodic = (
        {s for s in ordered if keep_period and keep_period > 0 and s % keep_period == 0}
        if keep_period
        else set()
    )
    resumable = [s for s in ordered if s in newest]
    archival = [s for s in ordered if s in periodic and s not in newest]
    delete = [s for s in ordered if s not in newest and s not in periodic]
    return resumable, archival, delete


def existing_checkpoint_steps(checkpoint_dir: Path) -> list[int]:
    if not checkpoint_dir.is_dir():
        return []
    return sorted(
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    )


def prune_checkpoints(
    checkpoint_dir: Path, *, keep_last: int = 2, keep_period: int | None = None, drop_archival_optimizer: bool = True
) -> dict[str, list[int]]:
    """Apply the retention policy to `checkpoint_dir`. Returns what it did."""
    checkpoint_dir = Path(checkpoint_dir)
    steps = existing_checkpoint_steps(checkpoint_dir)
    resumable, archival, delete = partition_checkpoints(steps, keep_last=keep_last, keep_period=keep_period)

    for step in delete:
        target = checkpoint_dir / str(step)
        shutil.rmtree(target, ignore_errors=True)
        LOGGER.info("Pruned checkpoint %s", target)

    stripped: list[int] = []
    if drop_archival_optimizer:
        for step in archival:
            optimizer_path = checkpoint_dir / str(step) / OPTIMIZER_FILENAME
            if optimizer_path.is_file():
                optimizer_path.unlink()
                stripped.append(step)
                LOGGER.info("Dropped optimizer state from archival checkpoint %s", optimizer_path.parent)

    return {"resumable": resumable, "archival": archival, "deleted": delete, "optimizer_stripped": stripped}
