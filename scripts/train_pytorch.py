"""PyTorch training entry point with DDP and FSDP support."""

import contextlib
import dataclasses
import gc
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm

try:
    import wandb
except ImportError:  # wandb is only required when wandb_enabled=true.
    wandb = None

import openpi.models.pi0_config
from openpi.models_pytorch.camera_head import load_state_dict_with_camera_whitelist
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data
import openpi.training.pytorch_checkpoints as _pytorch_checkpoints
import openpi.training.runtime_snapshot as _runtime_snapshot

# Source/view ids travel inside the batch because DataLoader workers and DDP
# ranks each see a different slice: worker-local counters cannot be reduced.
_SOURCE_NAMES = {
    0: "EgoDex",
    1: "EgoLive",
    2: "VITRA",
    3: "EgoVerse",
    4: "GenRobot",
    5: "Piper",
    6: "AgiBotWorld-Beta",
    7: "RoboCOIN",
    99: "unknown",
}
_VIEW_NAMES = {0: "high_level", 1: "low_level"}
_OBJECT_PROCESS_GROUP = None


def _object_process_group():
    """Return the CPU group used for variable-size Python metadata."""
    return _OBJECT_PROCESS_GROUP


def batch_source_view_counts(observation) -> dict[str, float]:
    """Count examples per source, per view and per source/view in one batch."""
    counts: dict[str, float] = {}
    source_id = getattr(observation, "source_id", None)
    view_type = getattr(observation, "vlm_view_type", None)
    if view_type is None:
        return counts
    view_type = view_type.reshape(-1)
    sources = source_id.reshape(-1) if source_id is not None else torch.full_like(view_type, 99)
    for s, v in zip(sources.tolist(), view_type.tolist(), strict=True):
        sname = _SOURCE_NAMES.get(int(s), f"source_{int(s)}")
        vname = _VIEW_NAMES.get(int(v), f"view_{int(v)}")
        counts[f"count/source/{sname}"] = counts.get(f"count/source/{sname}", 0.0) + 1.0
        counts[f"count/view/{vname}"] = counts.get(f"count/view/{vname}", 0.0) + 1.0
        key = f"count/source_view/{sname}/{vname}"
        counts[key] = counts.get(key, 0.0) + 1.0
    return counts


def all_reduce_metrics(metrics: dict[str, float], device) -> dict[str, float]:
    """Sum metric values across DDP ranks so ratios reflect the global batch.

    Ranks can hold different source/view mixes, so a rank-local ratio is not
    the ratio the model actually trained on. Keys are unioned across ranks
    first; otherwise a rank missing a key would desynchronize the collective.
    """
    if not torch.distributed.is_initialized():
        return metrics
    object_group = _object_process_group()
    world_size = torch.distributed.get_world_size(group=object_group)
    gathered: list[list[str]] = [None] * world_size  # type: ignore[list-item]
    torch.distributed.all_gather_object(gathered, sorted(metrics), group=object_group)
    keys = sorted({k for part in gathered for k in part})
    if not keys:
        return metrics
    values = torch.tensor([metrics.get(k, 0.0) for k in keys], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    return dict(zip(keys, values.tolist(), strict=True))


def reduce_sampling_counts(counts: dict[str, int]) -> dict[str, int]:
    """Merge dynamic robot sampling counters across DDP ranks."""
    if not torch.distributed.is_initialized():
        return dict(counts)
    object_group = _object_process_group()
    gathered: list[dict[str, int] | None] = [None] * torch.distributed.get_world_size(group=object_group)
    torch.distributed.all_gather_object(gathered, counts, group=object_group)
    merged: dict[str, int] = {}
    for rank_counts in gathered:
        for key, value in (rank_counts or {}).items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged


def update_sampling_report(
    path: Path,
    cumulative: dict[str, int],
    window: dict[str, int],
    *,
    global_step: int,
) -> dict[str, int]:
    """Atomically persist cumulative sampler counts consumed by training."""
    for key, value in window.items():
        cumulative[key] = cumulative.get(key, 0) + int(value)
    payload = {
        "schema_version": 1,
        "semantics": "sampler indices handed to the training DataLoader after resume skipping",
        "global_step": int(global_step),
        "counts": dict(sorted(cumulative.items())),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp_path.replace(path)
    return cumulative


def with_realized_ratios(counts: dict[str, float]) -> dict[str, float]:
    """Add realized source and view ratios alongside the raw counts."""
    out = dict(counts)
    for group in ("source", "view"):
        prefix = f"count/{group}/"
        total = sum(v for k, v in counts.items() if k.startswith(prefix))
        if total > 0:
            for k, v in counts.items():
                if k.startswith(prefix):
                    out[f"ratio/{group}/{k[len(prefix) :]}"] = v / total
    return out


# (numerator, denominator) -> reported global conditional mean.
# Averaging rank-local averages is wrong twice over: it weights a rank holding
# one valid example the same as a rank holding sixty-four, and a step where a
# branch was absent contributes a zero that dilutes the branch it never saw.
GLOBAL_MEAN_PAIRS = {
    "global/subtask_ce": ("subtask_loss_sum", "subtask_example_count"),
    "global/fast_ce": ("fast_loss_sum", "fast_example_count"),
    "global/fast_ce_weighted": ("fast_weighted_loss_sum", "fast_example_count"),
    "global/flow_mse": ("flow_mse_sum", "flow_element_count"),
    "global/flow_mse_weighted": ("flow_weighted_mse_sum", "flow_element_count"),
    "global/camera_trans": ("camera_trans_sum", "camera_trans_count"),
    "global/camera_rot": ("camera_rot_sum", "camera_rot_count"),
    "global/camera_fov": ("camera_fov_sum", "camera_fov_count"),
}
# The joint objective prefixes branch metrics, so accept either spelling.
_METRIC_PREFIXES = ("", "high_", "fast_", "flow_", "low_")


def accumulate_sum_count(infos: list[dict[str, float]]) -> dict[str, float]:
    """Sum every numerator/denominator over the window, without dividing.

    Division happens once, after the cross-rank all-reduce.
    """
    out: dict[str, float] = {}
    metric_keys = {key for pair in GLOBAL_MEAN_PAIRS.values() for key in pair}
    for key in metric_keys:
        total = 0.0
        for info in infos:
            for prefix in _METRIC_PREFIXES:
                value = info.get(f"{prefix}{key}")
                if value is not None:
                    total += float(value)
        out[f"acc/{key}"] = total
    return out


def global_conditional_means(accumulated: dict[str, float]) -> dict[str, float]:
    """Form sum/count AFTER the all-reduce. Absent branches stay absent."""
    out: dict[str, float] = {}
    for label, (num_key, den_key) in GLOBAL_MEAN_PAIRS.items():
        den = accumulated.get(f"acc/{den_key}", 0.0)
        if den > 0:
            out[label] = accumulated[f"acc/{num_key}"] / den
            out[f"{label}_count"] = den
    return out


def average_extra_metrics(infos: list[dict[str, float]]) -> dict[str, float]:
    """Average the non-core metrics over a logging window.

    The key set is the union over the whole window, not just `infos[0]`: under
    the joint objective a step reports only the branches its batch actually
    contained, so keying off the first step silently drops every metric of
    whichever branch that step happened to lack. Missing steps count as 0 --
    the paired `*_example_count` keys carry the denominator.
    """
    core = {"loss", "learning_rate", "grad_norm", "grad_norm_pre_clip"}
    keys = {k for info in infos for k in info} - core
    return {k: sum(info.get(k, 0.0) for info in infos) / len(infos) for k in keys}


def is_additive_metric(key: str) -> bool:
    """Return true for counters/numerators that must be summed over GA/log windows."""
    if key.startswith("count/"):
        return True
    additive_suffixes = (
        "_count",
        "_sum",
        "_element_count",
        "_n_valid",
    )
    if key.endswith(additive_suffixes):
        return True
    if "truncation" in key:
        return True
    return key in {
        "high_level_examples",
        "low_level_examples",
        "valid_target_token_count",
    }


def merge_micro_metrics(micro_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Merge micro-batch metrics into one optimizer-step payload.

    Gradient accumulation should sum additive quantities (counts, numerators,
    element counts, truncation counts) but average losses and other mean-like
    diagnostics. Averaging every key diluted sample counts by GA and made
    global conditional means wrong.
    """
    if not micro_metrics:
        return {}
    out: dict[str, float] = {}
    for metrics in micro_metrics:
        for key, value in metrics.items():
            out[key] = out.get(key, 0.0) + float(value)
    denom = float(len(micro_metrics))
    for key in list(out):
        if not is_additive_metric(key):
            out[key] /= denom
    return out


def sampler_epoch_from_global_step(global_step: int, loader_len: int, accum_steps: int) -> int:
    """Estimate the finite DataLoader pass that contains an optimizer step.

    `global_step` counts optimizer steps, while `len(loader)` counts the number
    of micro-batches in one finite pass through the wrapped torch DataLoader.
    Gradient accumulation therefore consumes `accum_steps` loader batches per
    optimizer step. This value is used only to seed/reseed distributed samplers
    at startup or resume; the infinite loader still advances the epoch whenever
    its internal finite DataLoader is exhausted.
    """
    accum_steps = max(1, int(accum_steps))
    if loader_len <= 0:
        return int(global_step)
    return (int(global_step) * accum_steps) // int(loader_len)


def sampler_offset_from_global_step(global_step: int, loader_len: int, accum_steps: int) -> int:
    """Return local micro-batch offset inside the sampler epoch at resume."""
    accum_steps = max(1, int(accum_steps))
    if loader_len <= 0:
        return 0
    return (int(global_step) * accum_steps) % int(loader_len)


def _profile_timing_enabled() -> bool:
    return os.environ.get("PROFILE_STEP_TIMING", "0").strip().lower() in {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _gradient_clip_foreach() -> bool | None:
    raw = os.environ.get("PI05_GRAD_CLIP_FOREACH", "auto").strip().lower()
    if raw in {"", "auto"}:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError("PI05_GRAD_CLIP_FOREACH must be auto, 0, or 1")


def _adamw_foreach() -> bool | None:
    raw = os.environ.get("PI05_ADAMW_FOREACH", "auto").strip().lower()
    if raw in {"", "auto"}:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError("PI05_ADAMW_FOREACH must be auto, 0, or 1")


def _largest_gradient_tensors(model: torch.nn.Module, limit: int = 10) -> list[tuple[str, float, float]]:
    """Return the largest post-clip parameter gradients for anomaly diagnosis."""
    rows = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        rows.append((name, float(torch.linalg.vector_norm(gradient)), float(gradient.abs().max())))
    return sorted(rows, key=lambda row: row[1], reverse=True)[:limit]


def _batch_numeric_ranges(observation, actions: torch.Tensor | None) -> list[tuple[str, float, int]]:
    """Summarize floating-point training inputs when a gradient anomaly occurs."""
    rows: list[tuple[str, float, int]] = []

    def add(name: str, value) -> None:
        if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.numel() == 0:
            return
        finite = torch.isfinite(value)
        finite_values = value[finite]
        max_abs = float(finite_values.detach().float().abs().max()) if finite_values.numel() else float("inf")
        rows.append((name, max_abs, int(value.numel() - finite_values.numel())))

    add("actions", actions)
    for name, value in vars(observation).items():
        if name == "images":
            continue
        if isinstance(value, dict):
            for child_name, child_value in value.items():
                add(f"observation.{name}.{child_name}", child_value)
        else:
            add(f"observation.{name}", value)
    return sorted(rows, key=lambda row: (row[2] > 0, row[1]), reverse=True)


def _sync_for_profile(device) -> None:
    if _profile_timing_enabled() and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        if wandb is not None:
            wandb.init(mode="disabled")
        return
    if wandb is None:
        raise ImportError("wandb_enabled=true requires the wandb package; install it or pass --wandb_enabled false")

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    global _OBJECT_PROCESS_GROUP

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # Bind rank -> device before NCCL initialization so barriers and
        # object collectives never have to guess the CUDA device mapping.
        torch.cuda.set_device(device)
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        init_kwargs = {"backend": backend, "init_method": "env://"}
        if backend == "nccl":
            init_kwargs["device_id"] = device
        torch.distributed.init_process_group(**init_kwargs)

        # Python dictionaries have variable keys and sizes. Keep those tiny
        # control-plane collectives on CPU/Gloo; gradients and numeric metric
        # reductions remain on the default NCCL group.
        if backend == "nccl":
            _OBJECT_PROCESS_GROUP = torch.distributed.new_group(backend="gloo")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    return use_ddp, local_rank, device


def cleanup_ddp():
    global _OBJECT_PROCESS_GROUP

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        if _OBJECT_PROCESS_GROUP is not None:
            torch.distributed.destroy_process_group(_OBJECT_PROCESS_GROUP)
            _OBJECT_PROCESS_GROUP = None
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if _pytorch_checkpoints.should_save(
        global_step, save_interval=config.save_interval, num_train_steps=config.num_train_steps
    ):
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        pruned = _pytorch_checkpoints.prune_checkpoints(
            config.checkpoint_dir,
            keep_last=config.keep_last_checkpoints,
            keep_period=config.keep_period,
        )
        if pruned["deleted"] or pruned["optimizer_stripped"]:
            logging.info(
                f"Checkpoint retention: kept resumable={pruned['resumable']} archival={pruned['archival']} "
                f"deleted={pruned['deleted']} optimizer_stripped={pruned['optimizer_stripped']}"
            )

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        if is_main:
            shutil.rmtree(config.checkpoint_dir)
            logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        if is_main:
            exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")
    if use_ddp:
        dist.barrier()

    runtime_snapshot_path = config.checkpoint_dir / "resolved_config.json"
    if is_main:
        runtime_snapshot = _runtime_snapshot.build_runtime_snapshot(
            config,
            repo_root=Path(__file__).resolve().parents[1],
            world_size=dist.get_world_size() if use_ddp else 1,
        )
        if resuming and runtime_snapshot_path.is_file():
            previous_snapshot = json.loads(runtime_snapshot_path.read_text())
            if previous_snapshot.get("contract_sha256") != runtime_snapshot.get("contract_sha256"):
                raise RuntimeError(
                    "Resume runtime contract differs from resolved_config.json: "
                    f"previous={previous_snapshot.get('contract_sha256')} "
                    f"current={runtime_snapshot.get('contract_sha256')}"
                )
        _runtime_snapshot.write_runtime_snapshot(runtime_snapshot_path, runtime_snapshot)
        logging.info("Wrote resolved runtime config: %s", runtime_snapshot_path)
    if use_ddp:
        dist.barrier()

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
            training_objective=getattr(config.model, "training_objective", "flow_camera"),
            enable_front_camera_tokens=getattr(config.model, "enable_front_camera_tokens", False),
            front_camera_history_offsets=getattr(config.model, "front_camera_history_offsets", (-48, -32, -16, 0)),
            front_camera_pose_loss_weight=getattr(config.model, "front_camera_pose_loss_weight", 0.2),
            front_camera_pose_embed_dim=getattr(config.model, "front_camera_pose_embed_dim", 2048),
            front_camera_pose_trunk_depth=getattr(config.model, "front_camera_pose_trunk_depth", 4),
            front_camera_pose_num_heads=getattr(config.model, "front_camera_pose_num_heads", 16),
            front_camera_pose_mlp_ratio=getattr(config.model, "front_camera_pose_mlp_ratio", 4),
            front_camera_pose_num_iterations=getattr(config.model, "front_camera_pose_num_iterations", 4),
            front_camera_pose_causal_attn=getattr(config.model, "front_camera_pose_causal_attn", True),
            front_camera_pose_loss_gamma=getattr(config.model, "front_camera_pose_loss_gamma", 0.6),
            front_camera_pose_loss_weight_trans=getattr(config.model, "front_camera_pose_loss_weight_trans", 1.0),
            front_camera_pose_loss_weight_rot=getattr(config.model, "front_camera_pose_loss_weight_rot", 1.0),
            front_camera_pose_loss_weight_focal=getattr(config.model, "front_camera_pose_loss_weight_focal", 0.5),
            front_camera_pose_loss_normalize_trans=getattr(
                config.model, "front_camera_pose_loss_normalize_trans", False
            ),
            front_camera_pose_loss_d_bar_floor=getattr(config.model, "front_camera_pose_loss_d_bar_floor", 0.01),
            front_camera_use_temporal_embeddings=getattr(config.model, "front_camera_use_temporal_embeddings", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    objective = getattr(model_cfg, "training_objective", "flow_camera")
    # human_joint_camera trains the continuous action path, so the Action
    # Expert, action projections and time MLP must stay trainable. Only the
    # VLM-only objective freezes them.
    if objective == "vlm_discrete_camera":
        model.freeze_for_vlm_discrete_camera()
        if is_main:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            logging.info(
                "Configured vlm_discrete_camera objective: trainable_params=%d frozen_params=%d",
                trainable,
                frozen,
            )

    disable_gc = _env_flag("PI05_DISABLE_GRADIENT_CHECKPOINTING", False)
    if hasattr(model, "gradient_checkpointing_enable") and not disable_gc:
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    elif disable_gc:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing disabled by PI05_DISABLE_GRADIENT_CHECKPOINTING=1")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        ddp_find_unused = _env_flag("PI05_DDP_FIND_UNUSED_PARAMETERS", True)
        ddp_static_graph = _env_flag("PI05_DDP_STATIC_GRAPH", world_size >= 8 and objective != "human_joint_camera")
        if is_main:
            logging.info(
                "DDP settings: find_unused_parameters=%s static_graph=%s",
                ddp_find_unused,
                ddp_static_graph,
            )
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=ddp_find_unused,
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            # static_graph assumes the same parameters are used every
            # iteration. human_joint_camera routes high-level and low-level
            # examples down different subgraphs, so which parameters
            # participate changes from batch to batch and the assumption
            # breaks. Keep it off for that objective regardless of world size.
            static_graph=ddp_static_graph,
        )

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        state_dict = safetensors.torch.load_file(model_path, device="cpu")
        report = load_state_dict_with_camera_whitelist(
            model_to_load,
            state_dict,
            allow_camera_missing=getattr(model_cfg, "enable_front_camera_tokens", False),
        )
        del state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}: {report}")

    if is_main:
        model_for_audit = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        parameter_groups: dict[str, dict[str, int]] = {}
        for name, parameter in model_for_audit.named_parameters():
            group = ".".join(name.split(".")[:2])
            counters = parameter_groups.setdefault(
                group, {"trainable_tensors": 0, "trainable_parameters": 0, "frozen_tensors": 0, "frozen_parameters": 0}
            )
            prefix = "trainable" if parameter.requires_grad else "frozen"
            counters[f"{prefix}_tensors"] += 1
            counters[f"{prefix}_parameters"] += parameter.numel()
        load_report = dataclasses.asdict(report) if config.pytorch_weight_path is not None else None
        _runtime_snapshot.update_runtime_snapshot(
            runtime_snapshot_path,
            {"model_initialization": {"checkpoint_load_report": load_report, "parameter_groups": parameter_groups}},
        )

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    optim_params = [param for param in model.parameters() if param.requires_grad]
    optim = torch.optim.AdamW(
        optim_params,
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
        foreach=_adamw_foreach(),
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(f"Training objective: {getattr(model_cfg, 'training_objective', 'flow_camera')}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    # Gradient accumulation: `global_step`, the LR schedule, logging and
    # checkpointing all count OPTIMIZER steps. Each optimizer step consumes
    # `gradient_accumulation_steps` micro-batches.
    accum_steps = max(1, int(getattr(config, "gradient_accumulation_steps", 1)))
    micro_index = 0
    micro_losses: list[float] = []
    micro_metrics: list[dict[str, float]] = []
    micro_timings: list[dict[str, float]] = []
    rank_counts: dict[str, float] = {}
    sampling_report_path = config.checkpoint_dir / "robot_sampling_observed.json"
    sampling_cumulative: dict[str, int] = {}
    if resuming and is_main and sampling_report_path.is_file():
        sampling_cumulative = {
            key: int(value) for key, value in json.loads(sampling_report_path.read_text()).get("counts", {}).items()
        }
    profile_step_timing = _profile_timing_enabled()
    debug_optimizer_phase = _env_flag("PI05_DEBUG_OPTIMIZER_PHASE", False)
    gradient_clip_foreach = _gradient_clip_foreach()
    batch_fetch_start = time.perf_counter()
    if is_main and accum_steps > 1:
        logging.info(
            f"Gradient accumulation: {accum_steps} micro-batches of {effective_batch_size} per GPU per optimizer "
            f"step (effective global batch {config.batch_size * accum_steps})"
        )
    if is_main and profile_step_timing:
        logging.info("PROFILE_STEP_TIMING=1: synchronizing CUDA for data/forward/backward/optimizer timing")

    while global_step < config.num_train_steps:
        # Seed the distributed sampler at the resumed position. `len(loader)`
        # is finite micro-batches per sampler epoch; `global_step` is optimizer
        # steps, so gradient accumulation must be included. The offset prevents
        # replaying the beginning of that finite pass after a checkpoint resume.
        if use_ddp and hasattr(loader, "set_epoch"):
            loader_len = len(loader)
            sampler_epoch = sampler_epoch_from_global_step(global_step, loader_len, accum_steps)
            sampler_offset = sampler_offset_from_global_step(global_step, loader_len, accum_steps)
            if is_main and sampler_offset:
                logging.info(
                    "Resume sampler position: global_step=%d accum_steps=%d loader_len=%d epoch=%d skip_batches=%d",
                    global_step,
                    accum_steps,
                    loader_len,
                    sampler_epoch,
                    sampler_offset,
                )
            loader.set_epoch(sampler_epoch, skip_batches=sampler_offset)

        for observation, actions in loader:
            if profile_step_timing:
                _sync_for_profile(device)
            data_fetch_s = time.perf_counter() - batch_fetch_start
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # The unified data loader returns (observation, actions) tuple
            data_move_start = time.perf_counter()
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901
            if profile_step_timing:
                _sync_for_profile(device)
            data_move_s = time.perf_counter() - data_move_start

            if micro_index == 0:
                # Update LR once per optimizer step
                for pg in optim.param_groups:
                    pg["lr"] = lr_schedule(global_step)

            is_final_micro = micro_index == accum_steps - 1
            # Skip DDP gradient all-reduce on non-final micro-batches.
            sync_context = (
                model.no_sync()
                if (use_ddp and not is_final_micro and hasattr(model, "no_sync"))
                else contextlib.nullcontext()
            )
            with sync_context:
                forward_start = time.perf_counter()
                losses = model(observation, actions)
                extra_metrics = {}
                if hasattr(losses, "loss"):
                    extra_metrics = {
                        key: float(value.detach().float().cpu())
                        for key, value in getattr(losses, "metrics", {}).items()
                        if isinstance(value, torch.Tensor)
                    }
                    losses = losses.loss
                # Ensure losses is a tensor and handle different return types
                if isinstance(losses, list | tuple):
                    losses = torch.stack(losses)
                elif not isinstance(losses, torch.Tensor):
                    losses = torch.tensor(losses, device=device, dtype=torch.float32)

                loss = losses.mean() / accum_steps
                if profile_step_timing:
                    _sync_for_profile(device)
                forward_s = time.perf_counter() - forward_start

                # Backward pass
                backward_start = time.perf_counter()
                loss.backward()
                if profile_step_timing:
                    _sync_for_profile(device)
                backward_s = time.perf_counter() - backward_start

            micro_losses.append(float(loss.detach()) * accum_steps)
            extra_metrics.update(batch_source_view_counts(observation))
            micro_metrics.append(extra_metrics)
            if profile_step_timing:
                micro_timings.append(
                    {
                        "timing/data_fetch_s": data_fetch_s,
                        "timing/data_move_s": data_move_s,
                        "timing/data_s": data_fetch_s + data_move_s,
                        "timing/forward_s": forward_s,
                        "timing/backward_s": backward_s,
                    }
                )
            micro_index += 1
            if not is_final_micro:
                batch_fetch_start = time.perf_counter()
                continue
            micro_index = 0

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            optimizer_start = time.perf_counter()
            if debug_optimizer_phase:
                logging.info(
                    "rank=%d optimizer phase: starting gradient clipping (foreach=%s)",
                    local_rank,
                    gradient_clip_foreach,
                )
            pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.optimizer.clip_gradient_norm,
                foreach=gradient_clip_foreach,
                error_if_nonfinite=True,
            )
            pre_clip_grad_norm_value = float(pre_clip_grad_norm)
            if is_main and pre_clip_grad_norm_value >= float(os.environ.get("PI05_GRAD_AUDIT_THRESHOLD", "10000")):
                logging.warning(
                    "Large pre-clip gradient norm at step %d: %.6e; top post-clip tensors follow",
                    global_step,
                    pre_clip_grad_norm_value,
                )
                for name, tensor_norm, max_abs in _largest_gradient_tensors(model):
                    logging.warning("gradient tensor=%s norm=%.6e max_abs=%.6e", name, tensor_norm, max_abs)
                for name, max_abs, nonfinite in _batch_numeric_ranges(observation, actions):
                    logging.warning("batch tensor=%s max_abs=%.6e nonfinite=%d", name, max_abs, nonfinite)
                for name, value in sorted(merge_micro_metrics(micro_metrics).items()):
                    if "loss" in name:
                        logging.warning("loss component=%s value=%.6e", name, float(value))
            if debug_optimizer_phase and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            if debug_optimizer_phase:
                logging.info("rank=%d optimizer phase: gradient clipping complete; starting AdamW step", local_rank)

            # Optimizer step
            optim.step()
            if debug_optimizer_phase and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            if debug_optimizer_phase:
                logging.info("rank=%d optimizer phase: AdamW step complete", local_rank)
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None
            if profile_step_timing:
                _sync_for_profile(device)
            optimizer_s = time.perf_counter() - optimizer_start

            step_loss = sum(micro_losses) / len(micro_losses)
            step_metrics = merge_micro_metrics(micro_metrics)
            if profile_step_timing:
                for timings in micro_timings:
                    for key, value in timings.items():
                        step_metrics[key] = step_metrics.get(key, 0.0) + value
                step_metrics["timing/optimizer_s"] = optimizer_s
                step_metrics["timing/step_profile_s"] = sum(
                    step_metrics.get(key, 0.0)
                    for key in (
                        "timing/data_s",
                        "timing/forward_s",
                        "timing/backward_s",
                        "timing/optimizer_s",
                    )
                )
            micro_losses = []
            micro_metrics = []
            micro_timings = []

            # Every rank accumulates its own slice of the mixture; the reduce
            # below turns these into the global counts the model trained on.
            for key, value in step_metrics.items():
                if key.startswith("count/"):
                    rank_counts[key] = rank_counts.get(key, 0.0) + value

            # Every rank keeps its own logging-window numerators/counts so the
            # later all-reduce reflects the real global batch, not rank 0.
            infos.append(
                {
                    "loss": step_loss,
                    "learning_rate": optim.param_groups[0]["lr"],
                    # Retain the old key for existing dashboards while naming
                    # its pre-clipping semantics explicitly.
                    "grad_norm": pre_clip_grad_norm_value,
                    "grad_norm_pre_clip": pre_clip_grad_norm_value,
                    **step_metrics,
                }
            )

            if global_step % config.log_interval == 0:
                # Collectives must run on every rank; calling them inside the
                # is_main branch would deadlock the others.
                if debug_optimizer_phase:
                    logging.info("rank=%d metrics phase: starting global source counts", local_rank)
                global_counts = with_realized_ratios(all_reduce_metrics(rank_counts, device))
                if debug_optimizer_phase:
                    logging.info("rank=%d metrics phase: global source counts complete", local_rank)
                rank_counts = {}
                # Branch losses: all-reduce numerator and denominator, then
                # divide ONCE. Also a collective, so it stays outside is_main.
                if debug_optimizer_phase:
                    logging.info("rank=%d metrics phase: starting global branch means", local_rank)
                global_means = global_conditional_means(all_reduce_metrics(accumulate_sum_count(infos), device))
                if debug_optimizer_phase:
                    logging.info("rank=%d metrics phase: global branch means complete", local_rank)
                    logging.info("rank=%d metrics phase: starting sampling snapshot", local_rank)
                sampling_window = reduce_sampling_counts(loader.sampling_snapshot(reset=True))
                if debug_optimizer_phase:
                    logging.info("rank=%d metrics phase: sampling snapshot complete", local_rank)
            else:
                global_counts = {}
                global_means = {}
                sampling_window = {}

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
                avg_extra_metrics = average_extra_metrics(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm_pre_clip={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )
                # Console-log the reduced mixture too: routing these only into
                # the wandb payload makes them invisible whenever wandb is off,
                # which is exactly when a smoke run needs them most.
                if global_counts:
                    ratios = {k: round(v, 4) for k, v in sorted(global_counts.items()) if k.startswith("ratio/")}
                    counts = {k: int(v) for k, v in sorted(global_counts.items()) if k.startswith("count/")}
                    logging.info(f"step={global_step} mixture_counts={counts}")
                    logging.info(f"step={global_step} mixture_ratios={ratios}")
                per_view = {
                    k: round(v, 4)
                    for k, v in sorted(avg_extra_metrics.items())
                    if k.startswith(("loss_", "high_", "low_", "fast_", "flow_"))
                }
                if per_view:
                    logging.info(f"step={global_step} objective_losses={per_view}")
                if global_means:
                    logging.info(
                        f"step={global_step} global_means={ {k: round(v, 6) for k, v in sorted(global_means.items())} }"
                    )
                if sampling_window:
                    sampling_cumulative = update_sampling_report(
                        sampling_report_path,
                        sampling_cumulative,
                        sampling_window,
                        global_step=global_step,
                    )
                    task_counts = {
                        key.removeprefix("task/"): value
                        for key, value in sorted(sampling_window.items())
                        if key.startswith("task/")
                    }
                    logging.info("step=%d robot_sampling_task_counts=%s", global_step, task_counts)
                timing_metrics = {
                    k: round(v, 4) for k, v in sorted(avg_extra_metrics.items()) if k.startswith("timing/")
                }
                if timing_metrics:
                    logging.info(f"step={global_step} timing={timing_metrics}")

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                        **avg_extra_metrics,
                        **global_counts,
                        **global_means,
                    }
                    sample_total = sampling_window.get("total", 0)
                    if sample_total:
                        for key, value in sampling_window.items():
                            if key.startswith("task/"):
                                task_id = key.removeprefix("task/")
                                log_payload[f"robot_sampling/count/task_{task_id}"] = value
                                log_payload[f"robot_sampling/ratio/task_{task_id}"] = value / sample_total
                        log_payload["robot_sampling/count/total"] = sample_total
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()

            # Reset stats collection on every rank. `infos` feeds a collective
            # all-reduce above, so non-main ranks must not carry old windows
            # into the next logging boundary.
            if global_step % config.log_interval == 0:
                infos = []

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{step_loss:.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )
            batch_fetch_start = time.perf_counter()

    # Close progress bar
    if pbar is not None:
        pbar.close()

    final_sampling_window = reduce_sampling_counts(loader.sampling_snapshot(reset=True))
    if is_main and final_sampling_window:
        sampling_cumulative = update_sampling_report(
            sampling_report_path,
            sampling_cumulative,
            final_sampling_window,
            global_step=global_step,
        )
        logging.info("Final robot sampling audit written to %s", sampling_report_path)

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
