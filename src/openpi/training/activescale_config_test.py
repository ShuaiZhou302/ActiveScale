import pathlib
import sys

import pytest
import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training import activescale_config
from openpi.training import config
from openpi.training import optimizer


def test_public_training_cli_builds(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(sys, "argv", ["train_pytorch.py", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        config.cli()
    assert exit_info.value.code == 0
    assert "activescale_pi05_human_robot_h50" in capsys.readouterr().out


def test_public_midtrain_recipe_matches_model_contract():
    train = config.get_config("activescale_pi05_human_robot_h50")
    assert train.batch_size == 256
    assert train.gradient_accumulation_steps == 1
    assert train.num_train_steps == 1
    assert train.model.action_horizon == 50
    assert train.model.front_camera_history_offsets == (-48, -32, -16, 0)
    assert not train.model.enable_front_history_images
    assert train.model.enable_front_camera_tokens
    assert train.model.front_camera_pose_loss_weight == 0.2
    assert train.data.human_source_views is not None
    assert train.data.human_source_norm_stats_paths is not None

    resolved = train.data.create(pathlib.Path("assets"), train.model)
    assert resolved.offline_human_vlm_action_loss_weight == 0.2
    assert resolved.cotrain_piper_action_loss_weight == 1.0
    assert resolved.cotrain_public_robot_action_loss_weight == 1.0
    assert resolved.cotrain_piper_anchor_stride == 6
    assert resolved.cotrain_public_robot_anchor_stride == 6

    with torch.device("meta"):
        model = PI0Pytorch(train.model)
    assert model.enable_front_history_images


def test_public_configs_do_not_embed_lab_paths():
    for name in (
        "activescale_pi05_human_robot_h50_smoke",
        "activescale_pi05_human_robot_h50",
        "activescale_pi05_piper_posttrain_h50",
        "activescale_pi05_piper_bag_policy",
        "activescale_pi05_piper_drawer_policy",
        "activescale_pi05_piper_pot_policy",
        "activescale_pi05_piper_box_policy",
        "activescale_pi05_piper_under_policy",
    ):
        rendered = repr(config.get_config(name))
        assert "/data/user/" not in rendered
        assert "/home/agilex/" not in rendered
        assert "10.120." not in rendered


def test_public_posttrain_recipe_matches_deployed_checkpoint_contract():
    train = config.get_config("activescale_pi05_piper_posttrain_h50")
    assert train.batch_size == 32
    assert train.num_workers == 0
    assert train.model.training_objective == "flow_camera"
    assert train.model.action_horizon == 50
    assert train.model.max_token_len == 512
    assert train.model.enable_front_camera_tokens
    assert train.model.front_camera_pose_loss_weight == 0.2
    assert train.lr_schedule.peak_lr == 5e-5
    assert train.lr_schedule.decay_lr == 5e-6


@pytest.mark.parametrize(
    ("task", "task_indices", "asset_id"),
    (
        ("bag", (0,), "cobot-avp-teleop_lerobot_task0_bag_camera_clean_v2"),
        ("drawer", tuple(range(1, 11)), "cobot-avp-teleop_lerobot_task1_10_drawer_camera_new_20260906"),
        ("pot", tuple(range(11, 16)), "cobot-avp-teleop_lerobot_task11_15_pot_camera"),
        ("box", tuple(range(16, 26)), "cobot-avp-teleop_lerobot_task16_25_box_camera_new_20260906"),
        (
            "under",
            tuple(range(26, 35)),
            "cobot-avp-teleop_lerobot_task26_34_under_reannotated_camera",
        ),
    ),
)
def test_deployed_piper_policy_configs_match_checkpoint_contract(task, task_indices, asset_id):
    train = config.get_config(f"activescale_pi05_piper_{task}_policy")
    assert train.model.training_objective == "flow_camera"
    assert train.model.action_horizon == 50
    assert train.model.max_token_len == 512
    assert train.model.enable_front_camera_tokens
    assert not train.model.enable_front_history_images
    assert train.data.task_indices == task_indices
    assert train.data.assets.asset_id == asset_id
    assert train.lr_schedule == optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    )
    assert train.policy_metadata == {"task_family": task, "history_contract": "mask"}


def test_piper_selection_contract_is_configurable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ACTIVESCALE_PIPER_ROOTS", "/piper/shared,/piper/under,/piper/recollection")
    monkeypatch.setenv("ACTIVESCALE_PIPER_CACHES", "/cache/shared,/cache/under,/cache/recollection")
    monkeypatch.setenv("ACTIVESCALE_PIPER_TASK_INDICES", "0-25;26-34;0-25")
    monkeypatch.setenv("ACTIVESCALE_PIPER_INCLUDED_EPISODES", "-,-,/assets/recollection.json")
    monkeypatch.setenv("ACTIVESCALE_PIPER_EXCLUDED_ANCHORS", "[[[66,545]],[],[]]")

    data = activescale_config._cotrain_data()
    assert data.piper_task_indices == (tuple(range(26)), tuple(range(26, 35)), tuple(range(26)))
    assert data.piper_included_episodes_paths == ("", "", "/assets/recollection.json")
    assert data.piper_excluded_anchors == (((66, 545),), (), ())


def test_piper_selection_contract_rejects_misaligned_groups(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ACTIVESCALE_PIPER_ROOTS", "/piper/a,/piper/b")
    monkeypatch.setenv("ACTIVESCALE_PIPER_CACHES", "/cache/a,/cache/b")
    monkeypatch.setenv("ACTIVESCALE_PIPER_TASK_INDICES", "0-25")

    with pytest.raises(ValueError, match="one semicolon-separated group per Piper root"):
        activescale_config._cotrain_data()
