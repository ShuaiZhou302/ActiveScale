import dataclasses

import jax
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


class _RecordingSampler(torch.utils.data.Sampler[int]):
    def __init__(self, size: int):
        self._size = size
        self.epochs: list[int] = []

    def __iter__(self):
        return iter(range(self._size))

    def __len__(self):
        return self._size

    def set_epoch(self, epoch: int):
        self.epochs.append(epoch)


class _IndexDataset(torch.utils.data.Dataset):
    def __init__(self, size: int):
        self._size = size

    def __getitem__(self, index: int):
        return {"value": index}

    def __len__(self):
        return self._size


def test_torch_data_loader_reseeds_sampler_on_each_internal_epoch():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)
    sampler = _RecordingSampler(len(dataset))

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=2, sampler=sampler, num_batches=5)
    batches = list(loader)

    assert len(batches) == 5
    # With batch_size=2 and four samples, five batches span three internal
    # passes over the torch DataLoader. This used to stay empty because the
    # public DataLoaderImpl did not expose set_epoch and TorchDataLoader never
    # called the sampler at restart.
    assert sampler.epochs == [0, 1, 2]


def test_torch_data_loader_resume_skip_batches_once():
    dataset = _IndexDataset(10)
    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=2, framework="pytorch")

    loader.set_epoch(0, skip_batches=3)
    data_iter = iter(loader)

    assert next(data_iter)["value"].tolist() == [6, 7]
    assert next(data_iter)["value"].tolist() == [8, 9]
    assert next(data_iter)["value"].tolist() == [0, 1]


def test_torch_data_loader_resume_skip_batches_with_sampler():
    dataset = _IndexDataset(10)
    sampler = _RecordingSampler(len(dataset))
    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=2, sampler=sampler, framework="pytorch")

    loader.set_epoch(7, skip_batches=3)
    data_iter = iter(loader)

    first_batch = next(data_iter)
    assert sampler.epochs == [7]
    assert first_batch["value"].tolist() == [6, 7]
    assert next(data_iter)["value"].tolist() == [8, 9]
    assert next(data_iter)["value"].tolist() == [0, 1]


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
