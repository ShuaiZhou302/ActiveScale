"""Per-source / per-view metric aggregation, including the DDP reduction."""

import importlib.util
import os
import pathlib
import queue as queue_mod
import socket
import sys
import tempfile

import pytest

torch = pytest.importorskip("torch")

_SPEC = importlib.util.spec_from_file_location(
    "train_pytorch_metrics",
    pathlib.Path(__file__).resolve().parents[3] / "scripts" / "train_pytorch.py",
)


def _load_trainer():
    """Import the trainer module, skipping if its heavy deps are absent."""
    module = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = module
    try:
        _SPEC.loader.exec_module(module)
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"trainer deps unavailable: {exc}")
    return module


class _Obs:
    def __init__(self, source_ids, view_types):
        self.source_id = torch.tensor(source_ids, dtype=torch.long)
        self.vlm_view_type = torch.tensor(view_types, dtype=torch.long)


def test_counts_split_by_source_view_and_pair():
    trainer = _load_trainer()
    # EgoDex high, EgoDex low, EgoLive high, VITRA low
    counts = trainer.batch_source_view_counts(_Obs([0, 0, 1, 2], [0, 1, 0, 1]))
    assert counts["count/source/EgoDex"] == 2.0
    assert counts["count/source/EgoLive"] == 1.0
    assert counts["count/source/VITRA"] == 1.0
    assert counts["count/view/high_level"] == 2.0
    assert counts["count/view/low_level"] == 2.0
    assert counts["count/source_view/EgoDex/high_level"] == 1.0
    assert counts["count/source_view/VITRA/low_level"] == 1.0


def test_counts_are_empty_without_view_ids():
    trainer = _load_trainer()

    class _Bare:
        pass

    assert trainer.batch_source_view_counts(_Bare()) == {}


def test_missing_source_id_falls_back_to_unknown():
    trainer = _load_trainer()

    class _NoSource:
        source_id = None
        vlm_view_type = torch.tensor([0, 1], dtype=torch.long)

    counts = trainer.batch_source_view_counts(_NoSource())
    assert counts["count/source/unknown"] == 2.0


def test_realized_ratios_are_computed_per_group():
    trainer = _load_trainer()
    counts = {
        "count/source/EgoDex": 3.0,
        "count/source/EgoLive": 5.0,
        "count/source/VITRA": 2.0,
        "count/view/high_level": 6.0,
        "count/view/low_level": 4.0,
    }
    out = trainer.with_realized_ratios(counts)
    assert out["ratio/source/EgoDex"] == pytest.approx(0.3)
    assert out["ratio/source/EgoLive"] == pytest.approx(0.5)
    assert out["ratio/source/VITRA"] == pytest.approx(0.2)
    assert out["ratio/view/high_level"] == pytest.approx(0.6)
    assert out["ratio/view/low_level"] == pytest.approx(0.4)
    # Source and view ratios normalize independently.
    assert sum(v for k, v in out.items() if k.startswith("ratio/source/")) == pytest.approx(1.0)
    assert sum(v for k, v in out.items() if k.startswith("ratio/view/")) == pytest.approx(1.0)


def test_all_reduce_is_a_noop_without_distributed():
    trainer = _load_trainer()
    counts = {"count/source/EgoDex": 1.0}
    assert trainer.all_reduce_metrics(counts, torch.device("cpu")) == counts


def test_realized_ratios_tolerate_empty_counts():
    trainer = _load_trainer()
    assert trainer.with_realized_ratios({}) == {}


def test_sampler_epoch_resume_uses_micro_batches():
    trainer = _load_trainer()
    # `global_step` is optimizer steps, but len(loader) is finite micro-batches.
    assert trainer.sampler_epoch_from_global_step(global_step=4, loader_len=10, accum_steps=2) == 0
    assert trainer.sampler_epoch_from_global_step(global_step=5, loader_len=10, accum_steps=2) == 1
    assert trainer.sampler_epoch_from_global_step(global_step=9, loader_len=10, accum_steps=2) == 1
    assert trainer.sampler_epoch_from_global_step(global_step=10, loader_len=10, accum_steps=2) == 2


def test_sampler_epoch_resume_tolerates_unknown_loader_length():
    trainer = _load_trainer()
    assert trainer.sampler_epoch_from_global_step(global_step=7, loader_len=0, accum_steps=4) == 7


def test_sampler_offset_resume_uses_micro_batches():
    trainer = _load_trainer()
    assert trainer.sampler_offset_from_global_step(global_step=4, loader_len=10, accum_steps=2) == 8
    assert trainer.sampler_offset_from_global_step(global_step=5, loader_len=10, accum_steps=2) == 0
    assert trainer.sampler_offset_from_global_step(global_step=7, loader_len=10, accum_steps=2) == 4


def test_sampler_offset_resume_tolerates_unknown_loader_length():
    trainer = _load_trainer()
    assert trainer.sampler_offset_from_global_step(global_step=7, loader_len=0, accum_steps=4) == 0


def test_window_average_unions_keys_across_steps():
    """A branch absent from the window's FIRST step must still be reported.

    Under the joint objective each step reports only the branches its batch
    contained. Keying the average off `infos[0]` therefore dropped every
    metric of whichever branch that step happened to lack -- which is how a
    whole logging window came back with no fast_/flow_ keys at all.
    """
    trainer = _load_trainer()
    infos = [
        {"loss": 1.0, "learning_rate": 5e-5, "high_loss_subtask_ce": 0.8, "high_level_examples": 4.0},
        {"loss": 2.0, "learning_rate": 5e-5, "fast_loss_fast_ce": 4.0, "flow_loss_action": 0.2},
    ]
    out = trainer.average_extra_metrics(infos)

    assert "fast_loss_fast_ce" in out, "low-level branch vanished because step 0 lacked it"
    assert "flow_loss_action" in out
    assert "high_loss_subtask_ce" in out
    # Steps missing a key count as 0; the paired *_example_count is the denominator.
    assert out["fast_loss_fast_ce"] == pytest.approx(2.0)
    assert out["high_loss_subtask_ce"] == pytest.approx(0.4)
    # Core keys are averaged separately and must not leak in.
    assert "loss" not in out
    assert "learning_rate" not in out


def test_window_average_is_order_independent():
    trainer = _load_trainer()
    a = {"loss": 1.0, "learning_rate": 5e-5, "high_loss_subtask_ce": 0.8}
    b = {"loss": 2.0, "learning_rate": 5e-5, "fast_loss_fast_ce": 4.0}
    assert trainer.average_extra_metrics([a, b]) == trainer.average_extra_metrics([b, a])


def test_merge_micro_metrics_sums_counts_and_averages_means():
    trainer = _load_trainer()
    out = trainer.merge_micro_metrics(
        [
            {
                "loss_fast": 2.0,
                "fast_loss_sum": 8.0,
                "fast_example_count": 4.0,
                "flow_flow_mse_sum": 12.0,
                "flow_flow_element_count": 6.0,
                "count/source/EgoDex": 4.0,
                "token_truncation_count": 1.0,
            },
            {
                "loss_fast": 4.0,
                "fast_loss_sum": 10.0,
                "fast_example_count": 5.0,
                "flow_flow_mse_sum": 20.0,
                "flow_flow_element_count": 10.0,
                "count/source/EgoDex": 5.0,
                "token_truncation_count": 0.0,
            },
        ]
    )

    assert out["loss_fast"] == pytest.approx(3.0)
    assert out["fast_loss_sum"] == pytest.approx(18.0)
    assert out["fast_example_count"] == pytest.approx(9.0)
    assert out["flow_flow_mse_sum"] == pytest.approx(32.0)
    assert out["flow_flow_element_count"] == pytest.approx(16.0)
    assert out["count/source/EgoDex"] == pytest.approx(9.0)
    assert out["token_truncation_count"] == pytest.approx(1.0)


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ddp_metric_worker(rank: int, world_size: int, init_method: str, result_queue):
    trainer = _load_trainer()
    initialized = False
    try:
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        initialized = True
        if rank == 0:
            infos = [
                {
                    "subtask_loss_sum": 12.0,
                    "subtask_example_count": 3.0,
                    "camera_trans_sum": 3.0,
                    "camera_trans_count": 1.0,
                    "camera_rot_sum": 4.0,
                    "camera_rot_count": 2.0,
                    "camera_fov_sum": 6.0,
                    "camera_fov_count": 3.0,
                }
            ]
        else:
            infos = [
                {
                    "fast_loss_sum": 10.0,
                    "fast_example_count": 2.0,
                    "flow_flow_mse_sum": 18.0,
                    "flow_flow_element_count": 6.0,
                    "camera_trans_sum": 7.0,
                    "camera_trans_count": 3.0,
                    "camera_rot_sum": 8.0,
                    "camera_rot_count": 2.0,
                    "camera_fov_sum": 9.0,
                    "camera_fov_count": 3.0,
                }
            ]
        reduced = trainer.all_reduce_metrics(trainer.accumulate_sum_count(infos), torch.device("cpu"))
        means = trainer.global_conditional_means(reduced)
        result_queue.put({"rank": rank, "means": means})
    except Exception as exc:  # pragma: no cover - diagnostic path
        result_queue.put({"rank": rank, "error": repr(exc)})
    finally:
        if initialized:
            torch.distributed.destroy_process_group()


def test_two_rank_global_means_match_manual_high_low_split():
    trainer = _load_trainer()
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")
    mp = pytest.importorskip("torch.multiprocessing")
    # Fork avoids re-importing the whole OpenPI/JAX stack in every child on
    # Linux, which made this tiny gloo regression look like a hang on HPC3.
    ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
    result_queue = ctx.Queue()
    port = _free_local_port()
    init_method = f"tcp://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as _tmpdir:
        procs = [
            ctx.Process(target=_ddp_metric_worker, args=(rank, 2, init_method, result_queue))
            for rank in range(2)
        ]
        for proc in procs:
            proc.start()
        try:
            results = [result_queue.get(timeout=60) for _ in procs]
        except queue_mod.Empty:  # pragma: no cover - defensive cleanup path
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()
            pytest.fail("timed out waiting for 2-rank DDP metric workers")
        for proc in procs:
            proc.join(timeout=30)
            assert proc.exitcode == 0

    for out in results:
        if "error" in out:
            pytest.fail(f"rank {out['rank']} failed: {out['error']}")
        out = out["means"]
        assert out["global/subtask_ce"] == pytest.approx(4.0)
        assert out["global/subtask_ce_count"] == pytest.approx(3.0)
        assert out["global/fast_ce"] == pytest.approx(5.0)
        assert out["global/fast_ce_count"] == pytest.approx(2.0)
        assert out["global/flow_mse"] == pytest.approx(3.0)
        assert out["global/flow_mse_count"] == pytest.approx(6.0)
        assert out["global/camera_trans"] == pytest.approx(2.5)
        assert out["global/camera_trans_count"] == pytest.approx(4.0)
        assert out["global/camera_rot"] == pytest.approx(3.0)
        assert out["global/camera_rot_count"] == pytest.approx(4.0)
        assert out["global/camera_fov"] == pytest.approx(2.5)
        assert out["global/camera_fov_count"] == pytest.approx(6.0)


# ------------------------------------------------ global conditional means

def test_global_mean_is_not_a_mean_of_rank_means():
    """The exact defect: a rank with 1 example must not outweigh one with 63."""
    trainer = _load_trainer()
    rank0 = {"subtask_loss_sum": 10.0, "subtask_example_count": 1.0}   # mean 10.0
    rank1 = {"subtask_loss_sum": 63.0, "subtask_example_count": 63.0}  # mean 1.0
    acc = {}
    for r in (rank0, rank1):
        for k, v in trainer.accumulate_sum_count([r]).items():
            acc[k] = acc.get(k, 0.0) + v
    out = trainer.global_conditional_means(acc)
    # Correct global mean is 73/64 = 1.1406, NOT (10.0 + 1.0)/2 = 5.5.
    assert out["global/subtask_ce"] == pytest.approx(73.0 / 64.0)
    assert out["global/subtask_ce_count"] == pytest.approx(64.0)


def test_absent_branch_is_omitted_not_reported_as_zero():
    """A step with no high-level examples must not dilute high-level loss."""
    trainer = _load_trainer()
    infos = [
        {"subtask_loss_sum": 4.0, "subtask_example_count": 2.0},
        {"fast_loss_sum": 9.0, "fast_example_count": 3.0},  # no subtask at all
    ]
    out = trainer.global_conditional_means(trainer.accumulate_sum_count(infos))
    assert out["global/subtask_ce"] == pytest.approx(2.0), "diluted by the absent step"
    assert out["global/fast_ce"] == pytest.approx(3.0)


def test_branch_absent_everywhere_produces_no_key():
    trainer = _load_trainer()
    out = trainer.global_conditional_means(
        trainer.accumulate_sum_count([{"fast_loss_sum": 1.0, "fast_example_count": 1.0}])
    )
    assert "global/subtask_ce" not in out
    assert "global/flow_mse" not in out


def test_prefixed_branch_metrics_are_collected():
    """The joint objective emits fast_*/flow_*-prefixed copies."""
    trainer = _load_trainer()
    infos = [{"flow_flow_mse_sum": 8.0, "flow_flow_element_count": 4.0}]
    out = trainer.global_conditional_means(trainer.accumulate_sum_count(infos))
    assert out["global/flow_mse"] == pytest.approx(2.0)


def test_camera_pairs_use_their_own_denominators():
    trainer = _load_trainer()
    infos = [
        {
            "camera_trans_sum": 6.0, "camera_trans_count": 3.0,
            "camera_rot_sum": 8.0, "camera_rot_count": 4.0,
            "camera_fov_sum": 5.0, "camera_fov_count": 10.0,
        }
    ]
    out = trainer.global_conditional_means(trainer.accumulate_sum_count(infos))
    assert out["global/camera_trans"] == pytest.approx(2.0)
    assert out["global/camera_rot"] == pytest.approx(2.0)
    assert out["global/camera_fov"] == pytest.approx(0.5)
