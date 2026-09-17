from collections import Counter
from collections.abc import Iterator, Sequence
import itertools
import json
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch

try:
    # lerobot is only required for LeRobot dataset configs; the offline human
    # VLM path must stay importable without it.
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
except ImportError:
    lerobot_dataset = None

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def set_epoch(self, epoch: int, skip_batches: int = 0) -> None:
        """Set sampler epoch and optional resume offset.

        `skip_batches` is used when resuming in the middle of a finite
        DataLoader pass: the next iterator drops that many local micro-batches
        before yielding. This avoids replaying the beginning of the sampler
        epoch after a checkpoint resume.
        """
        raise NotImplementedError("Subclasses of DataLoader should implement set_epoch.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of DataLoader should implement __len__.")

    def sampling_snapshot(self, *, reset: bool = False) -> dict[str, int]:
        raise NotImplementedError("This data loader does not expose sampling statistics.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)

    def __getattr__(self, name: str):
        # Preserve feature detection: hasattr(wrapper, "sample_identity")
        # should only be true when the underlying dataset really supports it.
        if name == "sample_identity" and hasattr(self._dataset, name):
            return getattr(self._dataset, name)
        raise AttributeError(name)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    if data_config.offline_human_vlm_pack_paths is not None:
        from openpi.training.human_vlm_dataset import HumanVLMPairedOfflineDataset

        if (
            data_config.offline_human_vlm_enable_fast_view
            and data_config.offline_human_vlm_fast_horizon != action_horizon
        ):
            raise ValueError(
                "human FAST/flow horizon must match model action_horizon: "
                f"{data_config.offline_human_vlm_fast_horizon} != {action_horizon}"
            )
        human = HumanVLMPairedOfflineDataset(
            pack_paths=data_config.offline_human_vlm_pack_paths,
            source_weights=data_config.offline_human_vlm_source_weights,
            max_token_len=model_config.max_token_len,
            fast_tokenizer_path=data_config.offline_human_vlm_fast_tokenizer_path,
            enable_subtask_view=data_config.offline_human_vlm_enable_subtask_view,
            enable_fast_view=data_config.offline_human_vlm_enable_fast_view,
            source_views=data_config.offline_human_vlm_source_views,
            fast_horizon=data_config.offline_human_vlm_fast_horizon,
            allow_partial_fast_horizon=data_config.offline_human_vlm_allow_partial_fast_horizon,
            norm_stats_path=data_config.offline_human_vlm_norm_stats_path,
            source_norm_stats_paths=data_config.offline_human_vlm_source_norm_stats_paths,
            action_norm_stats_path=data_config.offline_human_vlm_action_norm_stats_path,
            action_loss_weight=data_config.offline_human_vlm_action_loss_weight,
            fake_images=data_config.offline_human_vlm_fake_images,
            row_group_cache_size=data_config.offline_human_vlm_row_group_cache_size,
            max_filter_attempts=data_config.offline_human_vlm_max_filter_attempts,
            split=data_config.offline_human_vlm_split,
            val_basis_points=data_config.offline_human_vlm_val_basis_points,
            selection_manifest_path=data_config.offline_human_vlm_selection_manifest_path,
            view_cycle_mode=data_config.offline_human_vlm_view_cycle_mode,
            view_route_phase=data_config.offline_human_vlm_view_route_phase,
        )
        if not data_config.human_piper_cotrain:
            return human

        from openpi.training.human_piper_cotrain_dataset import HumanPiperCotrainDataset
        from openpi.training.human_piper_cotrain_dataset import PiperJointObjectiveAdapter
        from openpi.training.piper_camera_token_dataset import PiperCameraTokenLeRobotDataset
        from openpi.training.public_robot_cotrain_dataset import PublicRobotJointObjectiveAdapter
        from openpi.training.public_robot_cotrain_dataset import PublicRobotLeRobotDataset

        roots = tuple(data_config.cotrain_piper_roots)
        tasks = tuple(data_config.cotrain_piper_task_indices)
        caches = tuple(data_config.cotrain_piper_frame_cache_dirs)
        include_paths = tuple(data_config.cotrain_piper_included_episodes_paths)
        exclusions = tuple(data_config.cotrain_piper_excluded_anchors)
        if not roots or not (len(roots) == len(tasks) == len(caches)):
            raise ValueError("Human+Piper cotrain requires equally sized non-empty roots/tasks/cache lists")
        if exclusions and len(exclusions) != len(roots):
            raise ValueError("cotrain_piper_excluded_anchors must be empty or match cotrain_piper_roots")
        if include_paths and len(include_paths) != len(roots):
            raise ValueError(
                "cotrain_piper_included_episodes_paths must be empty or match cotrain_piper_roots"
            )
        if data_config.cotrain_piper_norm_stats_path is None:
            raise ValueError("Human+Piper cotrain requires cotrain_piper_norm_stats_path")
        piper_adapters = []
        for index, (root, task_ids, cache) in enumerate(zip(roots, tasks, caches, strict=True)):
            included_episodes = None
            if include_paths and include_paths[index]:
                include_path = pathlib.Path(os.path.expandvars(os.path.expanduser(include_paths[index])))
                payload = json.loads(include_path.read_text())
                if not isinstance(payload, list) or not all(isinstance(item, int) for item in payload):
                    raise ValueError(f"Piper episode allowlist must be a JSON integer list: {include_path}")
                included_episodes = payload
            base = PiperCameraTokenLeRobotDataset(
                root,
                action_horizon=action_horizon,
                anchor_stride=data_config.cotrain_piper_anchor_stride,
                task_indices=task_ids,
                frame_cache_dir=cache,
                history_offsets=model_config.front_camera_history_offsets,
                included_episodes=included_episodes,
                excluded_anchors=exclusions[index] if exclusions else (),
            )
            piper_adapters.append(
                PiperJointObjectiveAdapter(
                    base,
                    tokenizer=human.tokenizer,
                    norm_stats_path=data_config.cotrain_piper_norm_stats_path,
                    action_loss_weight=data_config.cotrain_piper_action_loss_weight,
                )
            )
        public_sources = tuple(data_config.cotrain_public_robot_sources)
        public_manifests = tuple(data_config.cotrain_public_robot_manifest_paths)
        public_roots = tuple(data_config.cotrain_public_robot_roots)
        public_caches = tuple(data_config.cotrain_public_robot_frame_cache_dirs)
        if public_sources and not (
            len(public_sources) == len(public_manifests) == len(public_roots) == len(public_caches)
        ):
            raise ValueError("Public Robot source/manifest/root/cache lists must have equal lengths")
        if public_sources and data_config.cotrain_public_robot_norm_stats_dir is None:
            raise ValueError("Public Robot cotrain requires cotrain_public_robot_norm_stats_dir")
        public_adapters = []
        for source, manifest, root, cache in zip(
            public_sources, public_manifests, public_roots, public_caches, strict=True
        ):
            base = PublicRobotLeRobotDataset(
                source=source,
                manifest_path=manifest,
                root=root,
                action_horizon=action_horizon,
                anchor_stride=data_config.cotrain_public_robot_anchor_stride,
                frame_cache_dir=cache,
                history_offsets=model_config.front_camera_history_offsets,
                fake_images=data_config.cotrain_public_robot_fake_images,
            )
            public_adapters.append(
                PublicRobotJointObjectiveAdapter(
                    base,
                    tokenizer=human.tokenizer,
                    norm_stats_dir=data_config.cotrain_public_robot_norm_stats_dir,
                    action_loss_weight=data_config.cotrain_public_robot_action_loss_weight,
                    allow_partial_fast_horizon=data_config.offline_human_vlm_allow_partial_fast_horizon,
                )
            )
        front_history_enabled = bool(
            model_config.enable_front_history_images or model_config.enable_front_camera_tokens
        )
        current_front_history_slot = None
        if not front_history_enabled:
            try:
                current_front_history_slot = tuple(model_config.front_camera_history_offsets).index(0)
            except ValueError as exc:
                raise ValueError("Original pi0.5 cotrain requires history offset 0 for its current front image") from exc
        return HumanPiperCotrainDataset(
            human,
            piper_adapters,
            public_robot_datasets=public_adapters,
            human_slots=data_config.cotrain_human_slots,
            piper_slots=data_config.cotrain_piper_slots,
            epoch_size_multiple=data_config.cotrain_epoch_size_multiple,
            current_front_history_slot=current_front_history_slot,
        )

    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    if data_config.piper_camera_token_dataset:
        from openpi.training.piper_camera_token_dataset import PiperCameraTokenLeRobotDataset

        root = os.environ.get("HF_LEROBOT_HOME")
        if root is None:
            raise ValueError("HF_LEROBOT_HOME must point to the Piper LeRobot root parent.")
        return PiperCameraTokenLeRobotDataset(
            os.path.join(root, repo_id),
            action_horizon=action_horizon,
            task_indices=data_config.piper_task_indices,
            frame_cache_dir=data_config.piper_frame_cache_dir,
            history_offsets=model_config.front_camera_history_offsets,
            state_gripper_indices=data_config.piper_state_gripper_indices,
            excluded_anchors=data_config.piper_excluded_anchors,
        )

    if lerobot_dataset is None:
        raise ImportError(f"lerobot is required for LeRobot dataset repo_id={repo_id!r} but is not installed")
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    if not getattr(dataset, "is_human_vlm_offline_pack", False):
        dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires the optional RLDS dependencies.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class _ResumeOffsetSampler(torch.utils.data.Sampler):
    """Sampler wrapper that can skip local batches once after a resume."""

    def __init__(self, sampler: torch.utils.data.Sampler, batch_size: int):
        self._sampler = sampler
        self._batch_size = int(batch_size)
        self._skip_batches = 0

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self._sampler, "set_epoch"):
            self._sampler.set_epoch(epoch)

    def set_skip_batches(self, skip_batches: int) -> None:
        self._skip_batches = max(0, int(skip_batches))

    def __iter__(self):
        iterator = iter(self._sampler)
        skip_samples = self._skip_batches * self._batch_size
        self._skip_batches = 0
        if skip_samples:
            iterator = itertools.islice(iterator, skip_samples, None)
        return iterator

    def __len__(self) -> int:
        return len(self._sampler)


class _SamplingAuditSampler(torch.utils.data.Sampler):
    """Count robot sampler indices actually handed to the DataLoader."""

    def __init__(self, sampler: torch.utils.data.Sampler, dataset: Dataset):
        self._sampler = sampler
        self._dataset = dataset
        self._counts: Counter[str] = Counter()

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self._sampler, "set_epoch"):
            self._sampler.set_epoch(epoch)

    def set_skip_batches(self, skip_batches: int) -> None:
        if hasattr(self._sampler, "set_skip_batches"):
            self._sampler.set_skip_batches(skip_batches)

    def __iter__(self):
        for index in self._sampler:
            identity = self._dataset.sample_identity(int(index))
            task = int(identity["task_index"])
            episode = int(identity["episode_index"])
            progress_bin = int(identity["episode_progress_bin"])
            prompt = str(identity["prompt"])
            self._counts["total"] += 1
            self._counts[f"task/{task}"] += 1
            self._counts[f"prompt/{task}/{prompt}"] += 1
            self._counts[f"episode/{task}/{episode}"] += 1
            self._counts[f"task_progress/{task}/{progress_bin}"] += 1
            yield index

    def __len__(self) -> int:
        return len(self._sampler)

    def snapshot(self, *, reset: bool = False) -> dict[str, int]:
        result = dict(self._counts)
        if reset:
            self._counts.clear()
        return result


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        self._epoch = 0
        self._skip_batches_on_next_iter = 0

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        if sampler is None and framework == "pytorch" and hasattr(dataset, "sample_identity"):
            sampler = (
                torch.utils.data.RandomSampler(dataset, generator=generator)
                if shuffle
                else torch.utils.data.SequentialSampler(dataset)
            )
        if sampler is not None and not isinstance(sampler, _ResumeOffsetSampler):
            sampler = _ResumeOffsetSampler(sampler, local_batch_size)
        if sampler is not None and hasattr(dataset, "sample_identity"):
            sampler = _SamplingAuditSampler(sampler, dataset)
        self._sampler = sampler
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def set_epoch(self, epoch: int, skip_batches: int = 0) -> None:
        self._epoch = int(epoch)
        self._skip_batches_on_next_iter = max(0, int(skip_batches))

    def __len__(self) -> int:
        if self._num_batches is not None:
            return self._num_batches
        return len(self._data_loader)

    def __iter__(self):
        num_items = 0
        while True:
            # This wrapper is deliberately infinite for training. The outer
            # loop therefore does not re-enter at epoch boundaries, so the
            # distributed sampler must be reseeded right where the underlying
            # torch DataLoader starts a new pass.
            if hasattr(self._sampler, "set_epoch"):
                self._sampler.set_epoch(self._epoch)
            if hasattr(self._sampler, "set_skip_batches"):
                self._sampler.set_skip_batches(self._skip_batches_on_next_iter)
            data_iter = iter(self._data_loader)
            if self._skip_batches_on_next_iter and not hasattr(self._sampler, "set_skip_batches"):
                skipped = 0
                while skipped < self._skip_batches_on_next_iter:
                    try:
                        next(data_iter)
                    except StopIteration:
                        break
                    skipped += 1
            self._skip_batches_on_next_iter = 0
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)
            self._epoch += 1

    def sampling_snapshot(self, *, reset: bool = False) -> dict[str, int]:
        if hasattr(self._sampler, "snapshot"):
            return self._sampler.snapshot(reset=reset)
        return {}


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)

    def set_epoch(self, epoch: int, skip_batches: int = 0) -> None:
        del epoch
        del skip_batches

    def __len__(self) -> int:
        if self._num_batches is not None:
            return self._num_batches
        return len(self._dataset)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def set_epoch(self, epoch: int, skip_batches: int = 0) -> None:
        if hasattr(self._data_loader, "set_epoch"):
            self._data_loader.set_epoch(epoch, skip_batches=skip_batches)

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]

    def __len__(self) -> int:
        return len(self._data_loader)

    def sampling_snapshot(self, *, reset: bool = False) -> dict[str, int]:
        if hasattr(self._data_loader, "sampling_snapshot"):
            return self._data_loader.sampling_snapshot(reset=reset)
        return {}
