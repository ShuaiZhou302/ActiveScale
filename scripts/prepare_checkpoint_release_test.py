from __future__ import annotations

import json
import pathlib

import pytest
from safetensors.torch import save_file
import torch

from scripts.prepare_checkpoint_release import prepare_release


def _release_inputs(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_file({"weight": torch.ones(1)}, checkpoint / "model.safetensors")

    config = tmp_path / "config.json"
    config.write_text("{}")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "activescale_commit": "test",
                "training_config": "test_config",
                "checkpoint_step": 10,
            }
        )
    )
    return checkpoint, config, metadata


def test_initialization_release_does_not_require_assets(tmp_path: pathlib.Path) -> None:
    checkpoint, config, metadata = _release_inputs(tmp_path)
    output = tmp_path / "release"
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("# Tested model card\n")

    prepare_release(checkpoint, output, config, metadata, "initialization", model_card)

    assert (output / "model.safetensors").is_file()
    assert not (output / "assets").exists()
    assert not (output / "optimizer.pt").exists()
    assert not (output / "metadata.pt").exists()
    assert (output / "README.md").read_text() == "# Tested model card\n"
    assert (output / "LICENSE").is_file()
    assert (output / "LICENSE_GEMMA.txt").is_file()
    assert not (output / "NOTICE").exists()


def test_inference_release_requires_assets(tmp_path: pathlib.Path) -> None:
    checkpoint, config, metadata = _release_inputs(tmp_path)

    with pytest.raises(FileNotFoundError, match="normalization assets"):
        prepare_release(checkpoint, tmp_path / "release", config, metadata, "inference")
