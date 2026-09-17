from contextlib import contextmanager
import copy
import math
import re
import types

import torch
from torch import nn

from openpi.models_pytorch.camera_head import CameraHead
from openpi.models_pytorch.camera_head import activate_pose
from openpi.models_pytorch.camera_head import apply_action_dim_mask_for_flow
from openpi.models_pytorch.camera_head import apply_action_mask_for_denoise_update
from openpi.models_pytorch.camera_head import build_front_camera_token_layout
from openpi.models_pytorch.camera_head import camera_pose_loss
from openpi.models_pytorch.camera_head import canonicalize_gt_quaternion_xyzw
from openpi.models_pytorch.camera_head import combine_action_dim_time_masks
from openpi.models_pytorch.camera_head import load_state_dict_with_camera_whitelist
from openpi.models_pytorch.camera_head import make_front_camera_prefix_att_masks
from openpi.models_pytorch.camera_head import make_front_history_prefix_att_masks
from openpi.models_pytorch.camera_head import matrix_to_quaternion_xyzw
from openpi.models_pytorch.camera_head import ref_from_camera_matrix_to_pose7

try:
    import pytest
except ImportError:
    class _PytestFallback:
        @contextmanager
        def raises(self, exc_type, match=None):
            try:
                yield
            except exc_type as exc:
                if match is not None and re.search(match, str(exc)) is None:
                    raise AssertionError(f"Exception message {str(exc)!r} does not match {match!r}") from exc
                return
            raise AssertionError(f"Expected {exc_type.__name__}")

    pytest = _PytestFallback()


def _make_att_2d_masks(pad_masks, att_masks):
    cumsum = torch.cumsum(att_masks, dim=1)
    return (cumsum[:, None, :] <= cumsum[:, :, None]) & (pad_masks[:, None, :] & pad_masks[:, :, None])


def _quat_xyzw_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = quat.unbind(-1)
    two_s = 2.0 / (quat * quat).sum(dim=-1)
    return torch.stack(
        [
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def _axis_angle_to_matrix(axis: torch.Tensor, angle: float) -> torch.Tensor:
    axis = axis.to(torch.float32)
    axis = axis / axis.norm().clamp_min(1e-8)
    kx, ky, kz = axis
    k = torch.tensor([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    eye = torch.eye(3)
    return eye + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def test_front_camera_token_layout_first_rest_and_indices():
    valid = torch.tensor(
        [
            [True, True, True, True],
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [True, False, True, True],
            [False, True, False, True],
        ]
    )
    layout = build_front_camera_token_layout(valid, image_patch_count=3)

    expected_type_ids = torch.tensor(
        [
            [0, 1, 1, 1],
            [-1, 0, 1, 1],
            [-1, -1, 0, 1],
            [-1, -1, -1, 0],
            [0, -1, 1, 1],
            [-1, 0, -1, 1],
        ]
    )
    assert torch.equal(layout.valid_mask, valid)
    assert torch.equal(layout.type_ids, expected_type_ids)
    assert torch.equal(layout.token_indices[0], torch.tensor([3, 7, 11, 15]))


def test_front_camera_attention_mask_has_no_future_or_context_leakage():
    valid = torch.ones(1, 4, dtype=torch.bool)
    lang_mask = torch.ones(1, 2, dtype=torch.bool)
    pad, ar, layout = make_front_camera_prefix_att_masks(
        valid,
        front_patch_count=2,
        wrist_patch_counts=(1, 1),
        wrist_valid_masks=(torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool)),
        lang_mask=lang_mask,
    )
    att = _make_att_2d_masks(pad, ar)[0]
    c0, c1, c2, c3 = layout.token_indices[0].tolist()

    assert not att[c0, 3]  # C0 cannot see front slot 1.
    assert not att[c0, 12]  # C0 cannot see left wrist.
    assert not att[c0, 14]  # C0 cannot see language/state.
    assert att[c1, c0]  # Later camera tokens can use earlier camera tokens.
    assert att[c1, 3] and att[c1, 4]  # C1 can see its own front patches.
    assert not att[c1, 6]  # C1 cannot see front slot 2.
    assert att[14, c0] and att[14, c3]  # Later context can see camera tokens.
    assert att[12, 13] and att[13, 12]  # Left/right wrist share one post-camera bidirectional block.
    assert att[12, 14] and att[14, 12]  # Language/state share that same block too.

    suffix_pad = torch.ones(1, 3, dtype=torch.bool)
    suffix_ar = torch.tensor([[True, False, False]])
    full_att = _make_att_2d_masks(torch.cat([pad, suffix_pad], dim=1), torch.cat([ar, suffix_ar], dim=1))[0]
    first_suffix = pad.shape[1]
    assert full_att[first_suffix, c0] and full_att[first_suffix, c3]


def test_front_history_attention_mask_has_temporal_blocks_without_separator_tokens():
    valid = torch.ones(1, 4, dtype=torch.bool)
    pad, ar = make_front_history_prefix_att_masks(
        valid,
        front_patch_count=2,
        wrist_patch_counts=(1,),
        wrist_valid_masks=(torch.ones(1, dtype=torch.bool),),
        lang_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    assert pad.shape == ar.shape == (1, 11)
    att = _make_att_2d_masks(pad, ar)[0]
    assert not att[0, 2]
    assert att[2, 0]
    assert not att[7, 8]
    assert att[8, 10]


def test_front_camera_attention_accepts_prefix_lm_language_mask():
    valid = torch.ones(1, 4, dtype=torch.bool)
    lang_mask = torch.ones(1, 4, dtype=torch.bool)
    # Two textual prompt tokens followed by two response tokens.
    lang_ar = torch.tensor([[0, 0, 1, 1]], dtype=torch.bool)
    pad, ar, layout = make_front_camera_prefix_att_masks(
        valid,
        front_patch_count=1,
        wrist_patch_counts=(1, 1),
        wrist_valid_masks=(torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool)),
        lang_mask=lang_mask,
        lang_att_mask=lang_ar,
    )
    att = _make_att_2d_masks(pad, ar)[0]
    c0, c1, _c2, c3 = layout.token_indices[0].tolist()
    text_start = pad.shape[1] - 4
    prompt0, prompt1, resp0, resp1 = range(text_start, text_start + 4)

    assert not att[c0, prompt0]
    assert not att[c1, resp0]
    assert not att[prompt0, resp0]
    assert att[prompt0, c3]
    assert att[prompt0, prompt1] and att[prompt1, prompt0]
    assert att[resp0, prompt0] and att[resp0, c3]
    assert not att[resp0, resp1]
    assert att[resp1, resp0] and att[resp1, prompt1]


def test_invalid_camera_slots_are_padded_out_of_attention():
    valid = torch.tensor([[False, False, True, True]])
    pad, ar, layout = make_front_camera_prefix_att_masks(
        valid,
        front_patch_count=2,
        wrist_patch_counts=(1, 1),
        wrist_valid_masks=(torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool)),
        lang_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    att = _make_att_2d_masks(pad, ar)[0]
    c2, c3 = layout.token_indices[0, 2:].tolist()
    assert not pad[0, layout.token_indices[0, 0]]
    assert not pad[0, layout.token_indices[0, 1]]
    assert att[c2, c2]
    assert not att[c2, layout.token_indices[0, 0]]
    assert not att[c2, layout.token_indices[0, 1]]
    assert not att[c2, c3]


def test_pi0pytorch_attention_mask_gives_padded_queries_self_edge():
    pi0_pytorch = pytest.importorskip("openpi.models_pytorch.pi0_pytorch")

    pad = torch.tensor([[False, True, False, True]], dtype=torch.bool)
    ar = torch.tensor([[False, False, True, False]], dtype=torch.bool)
    att = pi0_pytorch.make_att_2d_masks(pad, ar)[0]

    assert att[0, 0]  # padded query has a safe finite self edge.
    assert att[2, 2]  # another padded query has a safe finite self edge.
    assert not att[1, 0]  # valid tokens still cannot attend padded keys.
    assert not att[3, 2]


def test_ref_from_camera_pose7_uses_stored_direction_without_inversion():
    ref_from_camera = torch.eye(4).reshape(1, 1, 4, 4)
    ref_from_camera[..., 0, 3] = -1.0
    pose = ref_from_camera_matrix_to_pose7(ref_from_camera)
    assert torch.allclose(pose[0, 0, :3], torch.tensor([-1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(pose[0, 0, 3:7], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-6)


def test_ref_from_camera_pose7_matches_action_pose_and_inverse_is_wrong():
    ref_from_camera = torch.eye(4).reshape(1, 1, 4, 4)
    ref_from_camera[0, 0, :3, :3] = _axis_angle_to_matrix(torch.tensor([0.2, -0.4, 1.0]), 0.35)
    ref_from_camera[0, 0, :3, 3] = torch.tensor([0.21, -0.13, 0.34])

    action_camera_pose = ref_from_camera_matrix_to_pose7(ref_from_camera)[0, 0]
    direct_pose = ref_from_camera_matrix_to_pose7(ref_from_camera)[0, 0]
    inverse_pose = ref_from_camera_matrix_to_pose7(torch.linalg.inv(ref_from_camera))[0, 0]

    assert torch.allclose(direct_pose, action_camera_pose, atol=1e-6)
    assert (inverse_pose - action_camera_pose).abs().max() > 0.05


def test_ref_from_camera_rotation_quaternion_xyzw_canonical():
    angle = -math.pi / 2
    rot_z = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0, 0.0],
            [math.sin(angle), math.cos(angle), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ).reshape(1, 1, 4, 4)
    pose = ref_from_camera_matrix_to_pose7(rot_z)
    expected = torch.tensor([0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5)])
    assert torch.allclose(pose[0, 0, 3:7], expected, atol=1e-6)


def test_first_valid_camera_slot_is_identity_target():
    stored_ref_from_camera = torch.eye(4).repeat(1, 4, 1, 1)
    stored_ref_from_camera[:, 0, 0, 3] = 123.0  # invalid slot ignored by mask.
    stored_ref_from_camera[:, 1, 0, 3] = 0.0
    stored_ref_from_camera[:, 2, 0, 3] = -0.5
    valid = torch.tensor([[False, True, True, True]])
    pose = ref_from_camera_matrix_to_pose7(stored_ref_from_camera)
    assert torch.allclose(pose[valid][0], torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))


def test_camera_pose_loss_separates_pose_and_fov_validity_across_stages():
    pred = torch.zeros(1, 4, 9, requires_grad=True)
    pred.data[..., 6] = 1.0
    target = pred.detach().clone()
    target[..., 0] = 0.1
    pose_valid = torch.tensor([[True, True, False, False]])
    fov_valid = torch.tensor([[True, False, False, False]])

    loss, metrics = camera_pose_loss([pred, pred * 1.1], target, pose_valid_mask=pose_valid, fov_valid_mask=fov_valid)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["camera_pose_n_valid"].item() == 2
    assert metrics["camera_fov_n_valid"].item() == 1


def test_activate_pose_keeps_predicted_quaternion_raw_and_relu_fov():
    raw = torch.tensor([[[1.0, 2.0, 3.0, 2.0, 0.0, 0.0, 2.0, -1.0, 0.5]]])
    activated = activate_pose(raw)
    assert torch.allclose(activated[..., :7], raw[..., :7])
    assert torch.allclose(activated[..., 7:9], torch.tensor([[[0.0, 0.5]]]))


def test_gt_quaternion_is_canonicalized_but_pred_loss_is_raw_l1():
    pred_small = torch.zeros(1, 1, 9, requires_grad=True)
    pred_large = torch.zeros(1, 1, 9, requires_grad=True)
    pred_small.data[..., 6] = 0.5
    pred_large.data[..., 6] = 2.0
    target = torch.zeros(1, 1, 9)
    target[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, -1.0])
    valid = torch.ones(1, 1, dtype=torch.bool)

    canonical = canonicalize_gt_quaternion_xyzw(target[..., 3:7])
    assert torch.allclose(canonical, torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]))
    loss_small, _ = camera_pose_loss(pred_small, target, pose_valid_mask=valid, rot_weight=1.0, trans_weight=0.0, fov_weight=0.0)
    loss_large, _ = camera_pose_loss(pred_large, target, pose_valid_mask=valid, rot_weight=1.0, trans_weight=0.0, fov_weight=0.0)
    assert loss_small.item() != loss_large.item()


def test_matrix_to_quaternion_handles_hard_rotations_with_positive_qw():
    torch.manual_seed(7)
    cases = [
        torch.eye(3),
        _axis_angle_to_matrix(torch.tensor([0.0, 0.0, 1.0]), math.pi / 2),
        _axis_angle_to_matrix(torch.tensor([0.0, 0.0, 1.0]), -math.pi / 2),
        _axis_angle_to_matrix(torch.tensor([1.0, -1.0, 0.0]), math.pi),
        _axis_angle_to_matrix(torch.tensor([1.0, 2.0, 3.0]), math.pi),
        _axis_angle_to_matrix(torch.tensor([0.5, -2.0, 1.0]), math.radians(179.9)),
    ]
    for _ in range(5):
        axis = torch.randn(3)
        angle = float(torch.rand(()) * 2 * math.pi - math.pi)
        cases.append(_axis_angle_to_matrix(axis, angle))

    for rot in cases:
        quat = matrix_to_quaternion_xyzw(rot)
        recon = _quat_xyzw_to_matrix(quat)
        assert torch.isfinite(quat).all()
        assert quat[-1] >= -1e-6
        assert torch.allclose(torch.linalg.norm(quat), torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(recon, rot, atol=2e-5)

    plus_90_z = matrix_to_quaternion_xyzw(_axis_angle_to_matrix(torch.tensor([0.0, 0.0, 1.0]), math.pi / 2))
    assert torch.allclose(plus_90_z, torch.tensor([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]), atol=1e-6)


def test_camera_head_returns_staged_outputs_and_masks_invalid_slots():
    head = CameraHead(dim_in=16, trunk_depth=1, num_heads=4, mlp_ratio=2, num_iterations=4)
    valid = torch.tensor([[True, True, False, False], [False, False, False, False]])
    out_list = head(torch.randn(2, 4, 16), valid_mask=valid)

    assert len(out_list) == 4
    assert out_list[-1].shape == (2, 4, 9)
    assert torch.allclose(out_list[-1][~valid], torch.zeros_like(out_list[-1][~valid]))
    assert torch.all(out_list[-1][valid][..., 7:9] >= 0)


def test_camera_head_all_valid_mask_matches_unmasked_path():
    torch.manual_seed(1)
    head = CameraHead(dim_in=16, trunk_depth=1, num_heads=4, mlp_ratio=2, num_iterations=2)
    head.eval()
    features = torch.randn(2, 4, 16)
    with torch.no_grad():
        unmasked = head(features)[-1]
        all_valid = head(features, valid_mask=torch.ones(2, 4, dtype=torch.bool))[-1]
    assert torch.allclose(unmasked, all_valid, atol=1e-6)


def test_camera_head_iterative_residual_detaches_previous_prediction():
    torch.manual_seed(2)
    head = CameraHead(dim_in=16, trunk_depth=1, num_heads=4, mlp_ratio=2, num_iterations=2)
    features = torch.randn(1, 4, 16, requires_grad=True)
    raw_deltas = []

    def capture_raw_delta(_module, _inputs, output):
        output.retain_grad()
        raw_deltas.append(output)

    handle = head.pose_branch.register_forward_hook(capture_raw_delta)
    out_list = head(features)
    handle.remove()
    loss = out_list[-1].sum()
    loss.backward()
    assert len(raw_deltas) == 2
    assert raw_deltas[0].grad is None
    assert raw_deltas[1].grad is not None
    assert torch.isfinite(raw_deltas[1].grad).all()


def test_camera_head_invalid_slots_do_not_change_valid_outputs():
    torch.manual_seed(0)
    head = CameraHead(dim_in=16, trunk_depth=1, num_heads=4, mlp_ratio=2, num_iterations=2)
    head.eval()
    valid = torch.tensor([[False, False, True, True]])
    features = torch.randn(1, 4, 16)
    features_changed = features.clone()
    features_changed[:, :2] = 1e6

    with torch.no_grad():
        out = head(features, valid_mask=valid)[-1]
        out_changed = head(features_changed, valid_mask=valid)[-1]
    assert torch.allclose(out[:, 2:], out_changed[:, 2:], atol=1e-5)


def test_camera_head_leading_invalid_slots_have_finite_gradients():
    torch.manual_seed(3)
    head = CameraHead(dim_in=16, trunk_depth=1, num_heads=4, mlp_ratio=2, num_iterations=2)
    valid = torch.tensor([[False, False, True, True]])
    features = torch.randn(1, 4, 16, requires_grad=True)

    out = head(features, valid_mask=valid)[-1]
    assert torch.allclose(out[:, :2], torch.zeros_like(out[:, :2]))
    loss = out[:, 2:].sum()
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_action_dim_mask_applies_to_noise_input_target_and_loss_space():
    actions = torch.ones(1, 2, 5)
    noise = torch.full_like(actions, 3.0)
    mask = torch.tensor([[[True, False, True, False, False], [False, True, True, False, False]]])
    time = torch.tensor([0.25])

    noise_m, actions_m, x_t, u_t = apply_action_dim_mask_for_flow(actions, noise, mask, time)
    assert torch.equal(noise_m != 0, mask)
    assert torch.equal(actions_m != 0, mask)
    assert torch.equal(x_t != 0, mask)
    assert torch.equal(u_t != 0, mask)


def test_action_time_mask_combines_with_dim_mask_for_human_and_robot_actions():
    actions = torch.ones(2, 50, 32)
    action_dim_mask = torch.zeros(2, 32, dtype=torch.bool)
    action_dim_mask[0, :21] = True
    action_dim_mask[1, :23] = True
    action_time_valid_mask = torch.ones(2, 50, dtype=torch.bool)
    action_time_valid_mask[:, -10:] = False

    mask = combine_action_dim_time_masks(
        actions,
        action_dim_mask=action_dim_mask,
        action_time_valid_mask=action_time_valid_mask,
    )
    assert mask[0, :40, :21].all()
    assert not mask[0, :, 21:].any()
    assert mask[1, :40, :23].all()
    assert not mask[1, :, 23:].any()
    assert not mask[:, -10:, :].any()


def test_human_action_loss_mask_pads_to_32_and_combines_with_temporal_validity():
    actions = torch.ones(1, 50, 32)
    human_action_loss_mask = torch.ones(1, 50, 21, dtype=torch.bool)
    human_action_loss_mask[:, 7:11, 7:14] = False  # left wrist locally invalid.
    human_action_loss_mask[:, 13:17, 14:21] = False  # right wrist locally invalid.
    action_time_valid_mask = torch.ones(1, 50, dtype=torch.bool)
    action_time_valid_mask[:, -6:] = False

    mask = combine_action_dim_time_masks(
        actions,
        action_dim_mask=human_action_loss_mask,
        action_time_valid_mask=action_time_valid_mask,
    )

    assert mask.shape == actions.shape
    assert mask[:, :44, :7].all()
    assert not mask[:, 7:11, 7:14].any()
    assert not mask[:, 13:17, 14:21].any()
    assert not mask[..., 21:].any()
    assert not mask[:, -6:, :].any()

    noise = torch.ones_like(actions) * 3.0
    noise_m, actions_m, x_t, u_t = apply_action_dim_mask_for_flow(actions, noise, mask, torch.tensor([0.25]))
    assert torch.equal(noise_m != 0, mask)
    assert torch.equal(actions_m != 0, mask)
    assert torch.equal(x_t != 0, mask)
    assert torch.equal(u_t != 0, mask)


def test_inference_denoise_update_zeros_invalid_21d_and_23d_dims():
    x_t = torch.ones(2, 3, 32)
    v_t = torch.ones_like(x_t) * 2.0
    mask = torch.zeros_like(x_t, dtype=torch.bool)
    mask[0, :, :21] = True
    mask[1, :, :23] = True
    mask[:, 2, :] = False

    updated = apply_action_mask_for_denoise_update(x_t, v_t, torch.tensor(-0.1), mask)
    assert torch.allclose(updated[0, :2, :21], torch.full((2, 21), 0.8))
    assert torch.allclose(updated[1, :2, :23], torch.full((2, 23), 0.8))
    assert not updated[0, :, 21:].any()
    assert not updated[1, :, 23:].any()
    assert not updated[:, 2, :].any()


def test_pi0_config_inputs_spec_keeps_action_time_valid_mask_optional():
    pi0_config = pytest.importorskip("openpi.models.pi0_config")
    cfg = pi0_config.Pi0Config(pi05=True, enable_front_camera_tokens=False)
    obs_spec, _ = cfg.inputs_spec(batch_size=2)
    assert obs_spec.action_time_valid_mask is None
    assert cfg.fake_obs(batch_size=2).action_time_valid_mask is None

    cam_cfg = pi0_config.Pi0Config(pi05=True, enable_front_camera_tokens=True)
    cam_obs_spec, _ = cam_cfg.inputs_spec(batch_size=2)
    assert cam_obs_spec.action_time_valid_mask is None
    assert cam_cfg.fake_obs(batch_size=2).action_time_valid_mask is None


class _TinySelfAttn:
    def __init__(self):
        self.q_proj = nn.Linear(1, 1)


class _TinyLayer:
    def __init__(self):
        self.self_attn = _TinySelfAttn()


class _TinyLanguageModel:
    def __init__(self):
        self.layers = [_TinyLayer()]
        self.config = type("Config", (), {})()


class _TinyPaligemma(nn.Module):
    def __init__(self, width: int, vocab_size: int):
        super().__init__()
        self.language_model = _TinyLanguageModel()
        self.lm_head = nn.Linear(width, vocab_size, bias=False)


class _TinyExpert:
    def __init__(self):
        self.model = type("ExpertModel", (), {"config": type("Config", (), {})()})()


class _TinyPaliGemmaWithExpert(nn.Module):
    def __init__(self, width: int, vocab_size: int = 64):
        super().__init__()
        self.width = width
        self.image_proj = nn.Linear(3, width)
        self.lang_emb = nn.Embedding(vocab_size, width)
        self.paligemma = _TinyPaligemma(width, vocab_size)
        self.gemma_expert = _TinyExpert()
        self.action_expert_call_count = 0

    def embed_image(self, image):
        if image.ndim == 4 and image.shape[1] == 3:
            pooled = image.mean(dim=(2, 3))
        else:
            pooled = image.mean(dim=(1, 2))
        return self.image_proj(pooled).unsqueeze(1)

    def embed_language_tokens(self, tokens):
        return self.lang_emb(tokens.clamp(min=0, max=self.lang_emb.num_embeddings - 1))

    def forward(self, *, inputs_embeds, adarms_cond=None, attention_mask=None, **_kwargs):
        prefix_embs, suffix_embs = inputs_embeds
        prefix_out = None if prefix_embs is None else prefix_embs
        suffix_out = None if suffix_embs is None else suffix_embs
        if prefix_embs is not None and suffix_embs is not None:
            self.action_expert_call_count += 1
            if attention_mask is None:
                prefix_context = prefix_embs.mean(dim=1, keepdim=True)
            else:
                # Mirror the real transformer contract: a valid suffix query
                # cannot consume trailing padded prefix keys.
                visible = attention_mask[:, 0, -1, : prefix_embs.shape[1]] == 0
                visible_f = visible[:, :, None].to(prefix_embs.dtype)
                prefix_context = (prefix_embs * visible_f).sum(dim=1, keepdim=True)
                prefix_context = prefix_context / visible_f.sum(dim=1, keepdim=True).clamp_min(1)
            suffix_out = suffix_out + 0.01 * prefix_context
            # The real Gemma expert conditions on adarms_cond, so the double
            # must consume it too: otherwise time_mlp has no gradient path and
            # a real regression there would look like a passing test.
            expert_cond = adarms_cond[1] if isinstance(adarms_cond, (list, tuple)) else adarms_cond
            if expert_cond is not None:
                cond = expert_cond
                if cond.dim() == 2:
                    cond = cond[:, None, :]
                suffix_out = suffix_out + 0.01 * cond
        return (prefix_out, suffix_out), None


class _TinyConfig:
    pi05 = True
    action_dim = 32
    action_horizon = 50
    enable_front_history_images = False
    enable_front_camera_tokens = False
    front_camera_history_offsets = (-48, -32, -16, 0)
    front_camera_pose_embed_dim = 16
    front_camera_pose_trunk_depth = 1
    front_camera_pose_num_heads = 4
    front_camera_pose_mlp_ratio = 2
    front_camera_pose_num_iterations = 2
    front_camera_pose_causal_attn = True
    front_camera_pose_loss_weight = 0.2
    front_camera_pose_loss_weight_trans = 1.0
    front_camera_pose_loss_weight_rot = 1.0
    front_camera_pose_loss_weight_focal = 0.5
    front_camera_pose_loss_gamma = 0.6
    front_camera_pose_loss_normalize_trans = False
    front_camera_pose_loss_d_bar_floor = 0.01
    front_camera_use_temporal_embeddings = False
    training_objective = "flow_camera"


class _TinyObservation:
    pass


def _make_tiny_pi0(enable_camera: bool, *, enable_history: bool = False):
    pi0_pytorch = pytest.importorskip("openpi.models_pytorch.pi0_pytorch")
    model = object.__new__(pi0_pytorch.PI0Pytorch)
    nn.Module.__init__(model)
    model.config = _TinyConfig()
    model.config.enable_front_history_images = enable_history or enable_camera
    model.config.enable_front_camera_tokens = enable_camera
    model.pi05 = True
    model.gradient_checkpointing_enabled = False
    model.paligemma_with_expert = _TinyPaliGemmaWithExpert(width=16)
    model.action_in_proj = nn.Linear(32, 16)
    model.action_out_proj = nn.Linear(16, 32)
    model.time_mlp_in = nn.Linear(16, 16)
    model.time_mlp_out = nn.Linear(16, 16)
    model._preprocess_observation = lambda observation, train=True: observation
    model.enable_front_history_images = enable_history or enable_camera
    if enable_camera:
        model.enable_front_camera_tokens = True
        # Nonzero init mirrors the real model (normal_ std=1e-6). With the
        # identity-transformer test double, zero embeddings would make the
        # camera_projector input exactly zero and its weight grad vanish.
        model.front_camera_token_embeddings = nn.Parameter(torch.randn(2, 16) * 0.1)
        model.front_camera_temporal_embeddings = nn.Parameter(torch.zeros(4, 16))
        model.camera_projector = nn.Linear(16, 16)
        model.camera_head = CameraHead(dim_in=16, trunk_depth=1, num_heads=4, mlp_ratio=2, num_iterations=2)
    else:
        model.enable_front_camera_tokens = False
        model.front_camera_token_embeddings = None
        model.front_camera_temporal_embeddings = None
        model.camera_projector = None
        model.camera_head = None
    return model


def _make_tiny_observation(enable_camera: bool, *, action_dim_mask=None, action_time_valid_mask=None):
    obs = _TinyObservation()
    obs.images = {
        "left_wrist_0_rgb": torch.ones(1, 3, 8, 8),
        "right_wrist_0_rgb": torch.ones(1, 3, 8, 8) * 2,
    }
    if not enable_camera:
        obs.images = {"base_0_rgb": torch.ones(1, 3, 8, 8), **obs.images}
    obs.image_masks = {key: torch.ones(1, dtype=torch.bool) for key in obs.images}
    obs.state = torch.zeros(1, 32)
    obs.tokenized_prompt = torch.ones(1, 3, dtype=torch.long)
    obs.tokenized_prompt_mask = torch.ones(1, 3, dtype=torch.bool)
    obs.token_ar_mask = None
    obs.token_loss_mask = None
    obs.action_dim_mask = action_dim_mask
    obs.action_time_valid_mask = action_time_valid_mask

    if enable_camera:
        obs.front_history_images = torch.ones(1, 4, 3, 8, 8)
        obs.front_history_masks = torch.tensor([[False, False, True, True]])
        extrinsics = torch.eye(4).repeat(1, 4, 1, 1)
        extrinsics[:, 0, 0, 3] = 99.0
        extrinsics[:, 1, 1, 3] = 88.0
        extrinsics[:, 3, 0, 3] = -0.25
        obs.camera_extrinsics = extrinsics
        obs.camera_pose_valid = torch.tensor([[False, False, True, True]])
        obs.camera_fov = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [1.0, 1.2], [1.0, 1.2]]])
        obs.camera_fov_valid = torch.tensor([[False, False, True, True]])
        obs.camera_intrinsics = None
        obs.camera_image_hw = None
    return obs


def _make_tiny_vlm_observation():
    obs = _make_tiny_observation(enable_camera=True)
    obs.tokenized_prompt = torch.tensor([[1, 2, 3, 4, 5, 0]], dtype=torch.long)
    obs.tokenized_prompt_mask = torch.tensor([[True, True, True, True, True, False]])
    obs.token_ar_mask = torch.tensor([[0, 0, 0, 1, 1, 0]], dtype=torch.long)
    obs.token_loss_mask = torch.tensor([[False, False, False, True, True, False]])
    obs.vlm_view_type = torch.tensor([0], dtype=torch.long)
    obs.vlm_token_truncated = torch.tensor([False])
    return obs


def test_pi0pytorch_camera_disabled_forward_preserves_unstructured_mse_api():
    model = _make_tiny_pi0(enable_camera=False)
    obs = _make_tiny_observation(enable_camera=False)
    actions = torch.zeros(1, 50, 32)
    noise = torch.ones_like(actions) * 0.25
    time = torch.tensor([0.5])
    out = model.forward(obs, actions, noise=noise, time=time)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 50, 32)


def test_pi0pytorch_camera_enabled_forward_runs_camera_head_and_masks_action_input():
    model = _make_tiny_pi0(enable_camera=True)
    action_dim_mask = torch.zeros(1, 32, dtype=torch.bool)
    action_dim_mask[:, :21] = True
    action_time_valid_mask = torch.ones(1, 50, dtype=torch.bool)
    action_time_valid_mask[:, -10:] = False
    obs = _make_tiny_observation(
        enable_camera=True,
        action_dim_mask=action_dim_mask,
        action_time_valid_mask=action_time_valid_mask,
    )
    actions = torch.zeros(1, 50, 32)
    noise = torch.ones_like(actions) * 0.25
    time = torch.tensor([0.5])
    seen_action_inputs = []

    def capture_action_input(_module, inputs):
        seen_action_inputs.append(inputs[0].detach())

    handle = model.action_in_proj.register_forward_pre_hook(capture_action_input)
    out = model.forward(obs, actions, noise=noise, time=time)
    handle.remove()

    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.loss_action)
    assert torch.isfinite(out.loss_camera_pose)
    assert out.metrics["camera_pose_n_valid"].item() == 2
    assert out.metrics["camera_fov_n_valid"].item() == 2
    assert len(seen_action_inputs) == 1
    action_input = seen_action_inputs[0]
    assert not action_input[..., 21:].any()
    assert not action_input[:, -10:, :].any()


def test_pi0pytorch_history_only_forward_uses_history_without_camera_modules_or_loss():
    model = _make_tiny_pi0(enable_camera=False, enable_history=True)
    action_dim_mask = torch.zeros(1, 32, dtype=torch.bool)
    action_dim_mask[:, :23] = True
    obs = _make_tiny_observation(
        enable_camera=True,
        action_dim_mask=action_dim_mask,
        action_time_valid_mask=torch.ones(1, 50, dtype=torch.bool),
    )

    prefix, pad, _att, layout = model._embed_prefix_for_observation(obs)  # noqa: SLF001
    assert prefix.shape[1] == 9
    assert pad.shape == (1, 9)
    assert layout is None
    assert model.front_camera_token_embeddings is None
    assert model.front_camera_temporal_embeddings is None
    assert model.camera_projector is None
    assert model.camera_head is None

    actions = torch.zeros(1, 50, 32)
    out = model.forward(obs, actions, noise=torch.full_like(actions, 0.25), time=torch.tensor([0.5]))
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert out.loss_camera_pose.item() == 0.0
    assert out.metrics["camera_pose_n_valid"].item() == 0
    assert out.metrics["camera_fov_n_valid"].item() == 0
    assert model.paligemma_with_expert.image_proj.weight.grad is not None
    assert model.paligemma_with_expert.image_proj.weight.grad.abs().sum() > 0


def test_flow_action_weight_scales_loss_but_preserves_raw_metrics():
    model = _make_tiny_pi0(enable_camera=True)
    obs = _make_tiny_observation(enable_camera=True)
    obs.action_loss_weight = torch.tensor([0.2])
    actions = torch.zeros(1, 50, 32)
    noise = torch.ones_like(actions) * 0.25
    out = model.forward(obs, actions, noise=noise, time=torch.tensor([0.5]))

    assert torch.allclose(out.metrics["flow_weighted_mse_sum"], out.metrics["flow_mse_sum"] * 0.2)
    assert torch.allclose(
        out.metrics["flow_weighted_element_count"], out.metrics["flow_element_count"] * 0.2
    )
    assert torch.allclose(out.loss_action, out.metrics["flow_weighted_mse_sum"] / out.metrics["flow_element_count"])
    assert torch.allclose(
        out.loss_action,
        0.2 * out.metrics["flow_mse_sum"] / out.metrics["flow_element_count"],
    )


def test_pi0pytorch_vlm_discrete_objective_skips_action_expert_and_trains_vlm_camera_path():
    model = _make_tiny_pi0(enable_camera=True)
    model.config.training_objective = "vlm_discrete_camera"
    model.freeze_for_vlm_discrete_camera()
    obs = _make_tiny_vlm_observation()
    out = model.forward(obs, torch.zeros(1, 50, 32))
    out.loss.backward()

    assert torch.isfinite(out.loss)
    assert out.metrics["subtask_example_count"].item() == 1
    assert out.metrics["fast_example_count"].item() == 0
    assert model.paligemma_with_expert.action_expert_call_count == 0

    assert model.front_camera_token_embeddings.grad is not None
    assert torch.isfinite(model.front_camera_token_embeddings.grad).all()
    assert model.front_camera_token_embeddings.grad.abs().sum() > 0
    assert model.camera_projector.weight.grad is not None
    assert model.camera_projector.weight.grad.abs().sum() > 0
    assert model.camera_head.pose_branch.fc2.weight.grad is not None
    assert model.camera_head.pose_branch.fc2.weight.grad.abs().sum() > 0
    assert model.paligemma_with_expert.lang_emb.weight.grad is not None
    assert model.paligemma_with_expert.lang_emb.weight.grad.abs().sum() > 0
    assert model.paligemma_with_expert.paligemma.lm_head.weight.grad is not None
    assert model.paligemma_with_expert.paligemma.lm_head.weight.grad.abs().sum() > 0

    assert model.action_in_proj.weight.grad is None
    assert model.action_out_proj.weight.grad is None
    assert model.time_mlp_in.weight.grad is None
    assert model.time_mlp_out.weight.grad is None


def test_vlm_discrete_token_ce_is_mean_of_per_sample_means():
    model = _make_tiny_pi0(enable_camera=True)
    model.config.training_objective = "vlm_discrete_camera"
    text_hidden = torch.zeros(2, 5, 1)
    text_hidden[0, 0, 0] = 0
    text_hidden[1, 0, 0] = 1
    text_hidden[1, 1, 0] = 2
    text_hidden[1, 2, 0] = 3
    token_ids = torch.tensor(
        [
            [9, 0, 0, 0, 0],
            [9, 1, 2, 3, 0],
        ],
        dtype=torch.long,
    )
    token_mask = torch.ones_like(token_ids, dtype=torch.bool)
    loss_mask = torch.tensor(
        [
            [False, True, False, False, False],
            [False, True, True, True, False],
        ]
    )
    logits_table = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -2.0],
        ]
    )

    def fake_logits(hidden):
        return logits_table[hidden[:, 0].to(torch.long)]

    model._lm_logits_from_hidden = fake_logits
    loss, metrics = model._token_ce_loss(
        text_hidden,
        token_ids,
        token_mask,
        loss_mask,
        torch.tensor([0, 0]),
    )

    ce0 = torch.nn.functional.cross_entropy(logits_table[[0]], torch.tensor([0]), reduction="none")
    ce1 = torch.nn.functional.cross_entropy(logits_table[[1, 2, 3]], torch.tensor([1, 2, 3]), reduction="none")
    expected = torch.stack([ce0.mean(), ce1.mean()]).mean()
    token_weighted = torch.cat([ce0, ce1]).mean()
    assert torch.allclose(loss, expected)
    assert not torch.allclose(loss, token_weighted)
    assert metrics["valid_target_token_count"].item() == 4


def test_fast_ce_honors_per_example_action_loss_weight_without_corrupting_raw_metrics():
    model = _make_tiny_pi0(enable_camera=True)
    text_hidden = torch.zeros(2, 5, 1)
    text_hidden[1, 0, 0] = 1
    token_ids = torch.tensor([[9, 0, 0, 0, 0], [9, 1, 0, 0, 0]], dtype=torch.long)
    token_mask = torch.ones_like(token_ids, dtype=torch.bool)
    loss_mask = torch.tensor(
        [[False, True, False, False, False], [False, True, False, False, False]]
    )
    logits_table = torch.tensor([[3.0, 0.0], [0.0, 1.0]])

    model._lm_logits_from_hidden = lambda hidden: logits_table[hidden[:, 0].to(torch.long)]
    loss, metrics = model._token_ce_loss(
        text_hidden,
        token_ids,
        token_mask,
        loss_mask,
        torch.tensor([1, 1]),
        action_loss_weight=torch.tensor([0.2, 1.0]),
    )

    per_example = torch.stack(
        [
            torch.nn.functional.cross_entropy(logits_table[[0]], torch.tensor([0])),
            torch.nn.functional.cross_entropy(logits_table[[1]], torch.tensor([1])),
        ]
    )
    expected = (0.2 * per_example[0] + per_example[1]) / 2.0
    assert torch.allclose(loss, expected)
    assert metrics["fast_example_count"].item() == 2
    assert torch.allclose(metrics["fast_loss_sum"], per_example.sum())
    assert torch.allclose(metrics["fast_weight_sum"], torch.tensor(1.2))
    assert torch.allclose(metrics["fast_weighted_loss_sum"], 0.2 * per_example[0] + per_example[1])
    assert metrics["valid_target_token_count"].item() == 2


def test_vlm_discrete_ce_projects_only_text_target_hidden_not_camera_tokens():
    model = _make_tiny_pi0(enable_camera=True)
    model.config.training_objective = "vlm_discrete_camera"
    obs = _make_tiny_vlm_observation()
    seen_hidden_lengths = []
    original_logits = model._lm_logits_from_hidden

    def spy_logits(hidden):
        seen_hidden_lengths.append(hidden.shape[0])
        return original_logits(hidden)

    model._lm_logits_from_hidden = spy_logits
    out = model.forward(obs, torch.zeros(1, 50, 32))

    assert torch.isfinite(out.loss)
    assert seen_hidden_lengths == [int(obs.token_loss_mask[:, 1:].sum().item())]
    assert model.paligemma_with_expert.action_expert_call_count == 0


def test_pi0pytorch_camera_head_training_only_and_suffix_temporal_pad_mask():
    model = _make_tiny_pi0(enable_camera=True)
    obs = _make_tiny_observation(enable_camera=True)
    actions = torch.zeros(1, 50, 32)
    noise = torch.ones_like(actions) * 0.25
    time = torch.tensor([0.5])

    calls = {"camera_head": 0}
    original_forward = model.camera_head.forward

    def counting_camera_head(*args, **kwargs):
        calls["camera_head"] += 1
        return original_forward(*args, **kwargs)

    model.camera_head.forward = counting_camera_head
    model.forward(obs, actions, noise=noise, time=time)
    assert calls["camera_head"] == 1

    def forbidden_camera_head(*_args, **_kwargs):
        raise AssertionError("CameraHead must not run during sample_actions")

    model.camera_head.forward = forbidden_camera_head
    action_time_valid_mask = torch.ones(1, 50, dtype=torch.bool)
    action_time_valid_mask[:, -10:] = False
    obs.action_time_valid_mask = action_time_valid_mask
    sample = model.sample_actions("cpu", obs, noise=noise, num_steps=2)
    assert sample.shape == (1, 50, 32)
    assert not sample[:, -10:, :].any()

    _, suffix_pad_masks, _, _ = model.embed_suffix(
        obs.state,
        noise,
        time,
        action_time_valid_mask=action_time_valid_mask,
    )
    assert suffix_pad_masks.shape == (1, 50)
    assert suffix_pad_masks[:, :40].all()
    assert not suffix_pad_masks[:, -10:].any()


def test_pi0pytorch_rtc_sampling_is_finite_masked_and_camera_head_free():
    model = _make_tiny_pi0(enable_camera=True)
    obs = _make_tiny_observation(enable_camera=True)
    obs.action_dim_mask = torch.ones(1, 50, 23, dtype=torch.bool)
    noise = torch.full((1, 50, 32), 0.25)
    previous_remaining = torch.zeros(1, 25, 32)

    def forbidden_camera_head(*_args, **_kwargs):
        raise AssertionError("CameraHead must not run during RTC sampling")

    model.camera_head.forward = forbidden_camera_head
    denoise_calls = 0
    original_denoise_step = model.denoise_step

    def counting_denoise_step(*args, **kwargs):
        nonlocal denoise_calls
        denoise_calls += 1
        return original_denoise_step(*args, **kwargs)

    model.denoise_step = counting_denoise_step
    sample = model.sample_actions_rtc(
        "cpu",
        obs,
        previous_actions=previous_remaining,
        inference_delay_steps=1,
        execution_horizon_steps=25,
        noise=noise,
        num_steps=2,
        max_guidance_weight=5.0,
    )

    assert sample.shape == (1, 50, 32)
    assert torch.isfinite(sample).all()
    assert not sample[..., 23:].any()
    assert denoise_calls == 2
    assert all(parameter.grad is None for parameter in model.parameters())


class _CheckpointToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Linear(2, 2)
        self.front_camera_token_embeddings = nn.Parameter(torch.zeros(2, 4))
        self.camera_projector = nn.Linear(4, 4)
        self.camera_head = nn.Linear(4, 9)


def test_checkpoint_loader_allows_only_camera_missing_when_enabled():
    model = _CheckpointToyModel()
    state = model.state_dict()
    no_camera = {k: v for k, v in state.items() if not k.startswith(("front_camera_", "camera_projector", "camera_head"))}
    report = load_state_dict_with_camera_whitelist(model, no_camera, allow_camera_missing=True)
    assert set(report.missing_keys) == set(state) - set(no_camera)


def test_checkpoint_loader_materializes_known_tied_weights():
    model = nn.Module()
    model.paligemma_with_expert = nn.Module()
    model.paligemma_with_expert.paligemma = nn.Module()
    model.paligemma_with_expert.paligemma.lm_head = nn.Linear(3, 4, bias=False)
    model.paligemma_with_expert.paligemma.model = nn.Module()
    model.paligemma_with_expert.paligemma.model.language_model = nn.Module()
    model.paligemma_with_expert.paligemma.model.language_model.embed_tokens = nn.Embedding(4, 3)

    state = model.state_dict()
    state.pop("paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight")
    report = load_state_dict_with_camera_whitelist(model, state, allow_camera_missing=False)
    assert not report.missing_keys


def test_checkpoint_loader_rejects_missing_original_unexpected_and_shape_mismatch():
    model = _CheckpointToyModel()
    state = model.state_dict()
    missing_core = {k: v for k, v in state.items() if k != "core.weight"}
    with pytest.raises(RuntimeError, match="missing"):
        load_state_dict_with_camera_whitelist(model, missing_core, allow_camera_missing=True)

    unexpected = dict(state)
    unexpected["extra.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="unexpected"):
        load_state_dict_with_camera_whitelist(model, unexpected, allow_camera_missing=True)

    bad_shape = dict(state)
    bad_shape["core.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="shape_mismatches"):
        load_state_dict_with_camera_whitelist(model, bad_shape, allow_camera_missing=True)

    with pytest.raises(RuntimeError, match="missing"):
        load_state_dict_with_camera_whitelist(model, missing_core, allow_camera_missing=False)


def test_gradient_checkpointing_enable_uses_hf_layer_api():
    pi0_pytorch = pytest.importorskip("openpi.models_pytorch.pi0_pytorch")
    model = _make_tiny_pi0(enable_camera=False)
    calls = []

    class _GCStub(nn.Module):
        def __init__(self, name):
            super().__init__()
            self._name = name
            self.gradient_checkpointing = False

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            calls.append((self._name, gradient_checkpointing_kwargs))

    model.paligemma_with_expert.paligemma.language_model = _GCStub("language_model")
    model.paligemma_with_expert.paligemma.vision_tower = _GCStub("vision_tower")
    model.paligemma_with_expert.gemma_expert.model = _GCStub("expert")

    pi0_pytorch.PI0Pytorch.gradient_checkpointing_enable(model)

    assert model.gradient_checkpointing_enabled
    assert {name for name, _ in calls} == {"language_model", "vision_tower", "expert"}
    assert all(kwargs == {"use_reentrant": False} for _, kwargs in calls)
    # Bare attributes remain set for the fused prefix+suffix manual path.
    assert model.paligemma_with_expert.paligemma.language_model.gradient_checkpointing
    assert model.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing
    assert model.paligemma_with_expert.gemma_expert.model.gradient_checkpointing


def _tiny_real_gemma(num_layers=2, hidden=32, num_heads=4):
    transformers = pytest.importorskip("transformers")
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma.modeling_gemma import GemmaModel

    cfg = CONFIG_MAPPING["gemma"](
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_attention_heads=num_heads,
        num_key_value_heads=1,
        head_dim=hidden // num_heads,
        num_hidden_layers=num_layers,
        vocab_size=64,
        hidden_activation="gelu_pytorch_tanh",
        use_adarms=False,
        adarms_cond_dim=None,
    )
    torch.manual_seed(0)
    return GemmaModel(cfg)


def _tiny_fused_gemma_pair(hidden=32, num_heads=8):
    from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel

    class _PaliHolder(nn.Module):
        def __init__(self, language_model):
            super().__init__()
            self.language_model = language_model
            self.model = types.SimpleNamespace(language_model=language_model)
            self.config = types.SimpleNamespace(
                text_config=types.SimpleNamespace(num_hidden_layers=len(language_model.layers))
            )

    class _ExpertHolder(nn.Module):
        def __init__(self, language_model):
            super().__init__()
            self.model = language_model

    fused = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
    nn.Module.__init__(fused)
    fused.paligemma = _PaliHolder(_tiny_real_gemma(num_layers=2, hidden=hidden, num_heads=num_heads))
    fused.gemma_expert = _ExpertHolder(_tiny_real_gemma(num_layers=2, hidden=hidden, num_heads=num_heads))
    return fused, copy.deepcopy(fused)


def test_cached_rotary_and_residual_alias_preserve_fused_output_and_gradients(monkeypatch):
    reference, optimized = _tiny_fused_gemma_pair()
    reference.train()
    optimized.train()
    prefix = torch.randn(2, 5, 32)
    suffix = torch.randn(2, 3, 32)
    position_ids = torch.arange(8)[None].expand(2, -1)
    attention_mask = torch.zeros(2, 1, 8, 8)

    monkeypatch.delenv("PI05_CACHE_FUSED_ROTARY", raising=False)
    monkeypatch.delenv("PI05_DISABLE_RESIDUAL_CLONE", raising=False)
    ref_outputs, _ = reference(
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=[prefix, suffix],
        use_cache=False,
        adarms_cond=[None, None],
    )
    ref_loss = sum(output.float().square().mean() for output in ref_outputs)
    ref_loss.backward()

    monkeypatch.setenv("PI05_CACHE_FUSED_ROTARY", "1")
    monkeypatch.setenv("PI05_DISABLE_RESIDUAL_CLONE", "1")
    opt_outputs, _ = optimized(
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=[prefix, suffix],
        use_cache=False,
        adarms_cond=[None, None],
    )
    opt_loss = sum(output.float().square().mean() for output in opt_outputs)
    opt_loss.backward()

    for ref_output, opt_output in zip(ref_outputs, opt_outputs, strict=True):
        torch.testing.assert_close(ref_output, opt_output, atol=0, rtol=0)
    torch.testing.assert_close(ref_loss, opt_loss, atol=0, rtol=0)
    optimized_params = dict(optimized.named_parameters())
    for name, ref_param in reference.named_parameters():
        opt_param = optimized_params[name]
        if ref_param.grad is None or opt_param.grad is None:
            assert ref_param.grad is None and opt_param.grad is None, name
        else:
            # Removing an identity clone can change accumulation order without
            # changing the derivative. Keep this gate far tighter than BF16
            # training noise while allowing float32 roundoff.
            torch.testing.assert_close(ref_param.grad, opt_param.grad, atol=1e-7, rtol=1e-6, msg=name)


def test_hf_gemma_decoder_layers_recompute_with_gc_and_outputs_match():
    model = _tiny_real_gemma()
    model.train()
    emb = torch.randn(2, 6, 32)

    out_off = model(inputs_embeds=emb, use_cache=False).last_hidden_state
    loss_off = out_off.square().mean()
    loss_off.backward()
    grads_off = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)

    layer_calls = {"n": 0}
    for layer in model.layers:
        layer.register_forward_pre_hook(lambda *_args: layer_calls.__setitem__("n", layer_calls["n"] + 1))

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    out_on = model(inputs_embeds=emb, use_cache=False).last_hidden_state
    calls_after_forward = layer_calls["n"]
    loss_on = out_on.square().mean()
    loss_on.backward()
    calls_after_backward = layer_calls["n"]

    # Each decoder layer must run once in forward and once more during the
    # backward recomputation.
    assert calls_after_forward == len(model.layers)
    assert calls_after_backward == 2 * len(model.layers)
    assert torch.allclose(out_off, out_on, atol=1e-6)
    assert torch.isfinite(loss_on)
    for name, param in model.named_parameters():
        if name in grads_off:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
            assert torch.allclose(grads_off[name], param.grad, atol=1e-5), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for memory measurement")
def test_gc_reduces_peak_gpu_memory():
    device = torch.device("cuda:0")
    emb = torch.randn(4, 512, 256, device=device)

    def peak_mem(enable_gc):
        model = _tiny_real_gemma(num_layers=6, hidden=256).to(device)
        model.train()
        if enable_gc:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        out = model(inputs_embeds=emb, use_cache=False).last_hidden_state
        out.square().mean().backward()
        peak = torch.cuda.max_memory_allocated(device)
        del model
        torch.cuda.empty_cache()
        return peak

    peak_off = peak_mem(False)
    peak_on = peak_mem(True)
    assert peak_on < peak_off, f"GC peak {peak_on} should be below no-GC peak {peak_off}"


def _spy_embed_image(model, counter):
    original = model.paligemma_with_expert.embed_image

    def spy(image):
        counter["n"] += 1
        return original(image)

    model.paligemma_with_expert.embed_image = spy


def test_fully_masked_wrist_views_are_skipped_and_output_equivalent():
    torch.manual_seed(0)
    model = _make_tiny_pi0(enable_camera=True)
    model.config.training_objective = "vlm_discrete_camera"
    obs = _make_tiny_vlm_observation()
    obs.image_masks = {key: torch.zeros(1, dtype=torch.bool) for key in obs.images}

    counter = {"n": 0}
    _spy_embed_image(model, counter)

    model._skip_fully_masked_views = False
    out_ref = model.forward(obs, torch.zeros(1, 50, 32))
    calls_unskipped = counter["n"]

    counter["n"] = 0
    model._skip_fully_masked_views = True
    out_skip = model.forward(obs, torch.zeros(1, 50, 32))
    calls_skipped = counter["n"]

    # Two wrist views were skipped: only the 4 history-slot embeds remain.
    assert calls_unskipped - calls_skipped == 2
    assert torch.allclose(out_ref.loss, out_skip.loss, atol=1e-6)
    assert torch.allclose(out_ref.loss_action, out_skip.loss_action, atol=1e-6)
    assert torch.allclose(out_ref.loss_camera_pose, out_skip.loss_camera_pose, atol=1e-6)


def test_partially_masked_wrist_views_are_not_skipped():
    model = _make_tiny_pi0(enable_camera=True)
    model.config.training_objective = "vlm_discrete_camera"
    obs = _make_tiny_vlm_observation()
    keys = list(obs.images)
    obs.image_masks = {
        keys[0]: torch.ones(1, dtype=torch.bool),
        keys[1]: torch.zeros(1, dtype=torch.bool),
    }

    counter = {"n": 0}
    _spy_embed_image(model, counter)
    model._skip_fully_masked_views = True
    out = model.forward(obs, torch.zeros(1, 50, 32))

    # All history slots share one batched embed call, plus the one valid
    # wrist view; only the fully masked view is skipped.
    assert counter["n"] == 2
    assert torch.isfinite(out.loss)


# --- human_joint_camera -------------------------------------------------------


def test_view_constants_match_the_dataset_definitions():
    """The model defines these locally to avoid importing the dataset module."""
    pi0_pytorch = pytest.importorskip("openpi.models_pytorch.pi0_pytorch")
    dataset = pytest.importorskip("openpi.training.human_vlm_dataset")
    assert pi0_pytorch.VIEW_HIGH_LEVEL == dataset.VIEW_HIGH_LEVEL
    assert pi0_pytorch.VIEW_LOW_LEVEL == dataset.VIEW_LOW_LEVEL


def _make_joint_observation(view_types, *, horizon=10, action_dim=32):
    """Batch of tiny joint-objective examples with the given view ids."""
    n = len(view_types)

    def rep(t):
        return t.repeat(n, *([1] * (t.dim() - 1)))

    base = _make_tiny_vlm_observation()
    obs = _TinyObservation()
    obs.images = {k: rep(v) for k, v in base.images.items()}
    obs.image_masks = {k: rep(v) for k, v in base.image_masks.items()}
    obs.state = rep(base.state)
    obs.tokenized_prompt = rep(base.tokenized_prompt)
    obs.tokenized_prompt_mask = rep(base.tokenized_prompt_mask)
    obs.token_ar_mask = rep(base.token_ar_mask)
    obs.token_loss_mask = rep(base.token_loss_mask)
    obs.vlm_token_truncated = rep(base.vlm_token_truncated)
    obs.front_history_images = rep(base.front_history_images)
    obs.front_history_masks = rep(base.front_history_masks)
    obs.camera_extrinsics = rep(base.camera_extrinsics)
    obs.camera_pose_valid = rep(base.camera_pose_valid)
    obs.camera_fov = rep(base.camera_fov)
    obs.camera_fov_valid = rep(base.camera_fov_valid)
    obs.camera_intrinsics = None
    obs.camera_image_hw = None
    obs.vlm_view_type = torch.tensor(view_types, dtype=torch.long)

    # Conditioning prefix with no supervised positions (no GT FAST tokens).
    obs.flow_condition_tokens = torch.tensor([[1, 2, 3, 0, 0, 0]], dtype=torch.long).repeat(n, 1)
    obs.flow_condition_mask = torch.tensor([[True, True, True, False, False, False]]).repeat(n, 1)
    obs.flow_condition_ar_mask = torch.zeros(n, 6, dtype=torch.long)

    obs.flow_actions = torch.zeros(n, horizon, action_dim)
    dim_mask = torch.zeros(n, horizon, action_dim, dtype=torch.bool)
    dim_mask[:, :, :21] = True
    obs.flow_action_mask = dim_mask
    obs.flow_time_valid_mask = torch.ones(n, horizon, dtype=torch.bool)
    obs.action_dim_mask = dim_mask
    obs.action_time_valid_mask = torch.ones(n, horizon, dtype=torch.bool)
    return obs


def _make_joint_model():
    model = _make_tiny_pi0(enable_camera=True)
    model.config.training_objective = "human_joint_camera"
    model.config.action_horizon = 10
    model.config.joint_loss_weight_subtask = 1.0
    model.config.joint_loss_weight_fast = 1.0
    model.config.joint_loss_weight_flow = 1.0
    return model


def test_joint_high_level_only_batch_never_runs_action_expert():
    model = _make_joint_model()
    obs = _make_joint_observation([0, 0])
    out = model.forward(obs, torch.zeros(2, 10, 32))

    assert torch.isfinite(out.loss)
    assert model.paligemma_with_expert.action_expert_call_count == 0
    assert float(out.metrics["high_level_examples"]) == 2.0
    assert "low_level_examples" not in out.metrics
    # An absent subset still contributes a graph-connected zero.
    assert float(out.metrics["loss_fast"]) == 0.0
    assert float(out.metrics["loss_flow"]) == 0.0
    assert float(out.metrics["loss_subtask"]) != 0.0


def test_joint_low_level_batch_runs_action_expert_and_trains_both_paths():
    model = _make_joint_model()
    obs = _make_joint_observation([1, 1])
    out = model.forward(obs, torch.zeros(2, 10, 32))
    out.loss.backward()

    assert torch.isfinite(out.loss)
    assert model.paligemma_with_expert.action_expert_call_count > 0
    assert float(out.metrics["low_level_examples"]) == 2.0
    assert float(out.metrics["loss_fast"]) != 0.0
    # Both the discrete and continuous action paths receive gradient.
    assert model.action_in_proj.weight.grad is not None
    assert model.action_in_proj.weight.grad.abs().sum() > 0
    assert model.action_out_proj.weight.grad is not None
    assert model.action_out_proj.weight.grad.abs().sum() > 0
    assert model.camera_head.pose_branch.fc2.weight.grad is not None
    assert model.camera_head.pose_branch.fc2.weight.grad.abs().sum() > 0


def test_joint_front_history_cache_preserves_loss_and_gradients_and_skips_one_embed():
    """Sharing front-history SigLIP output must be mathematically transparent."""
    torch.manual_seed(17)
    reference = _make_joint_model()
    shared = _make_joint_model()
    shared.load_state_dict(reference.state_dict())
    reference._share_front_history_embeddings = False
    shared._share_front_history_embeddings = True

    obs = _make_joint_observation([1, 1])
    actions = torch.zeros(2, 10, 32)
    noise = torch.full_like(actions, 0.25)
    time = torch.full((2,), 0.5)
    ref_calls = {"n": 0}
    shared_calls = {"n": 0}
    _spy_embed_image(reference, ref_calls)
    _spy_embed_image(shared, shared_calls)

    ref_out = reference.forward(obs, actions, noise=noise, time=time)
    ref_out.loss.backward()
    shared_out = shared.forward(obs, actions, noise=noise, time=time)
    shared_out.loss.backward()

    assert shared_calls["n"] == ref_calls["n"] - 1
    assert torch.allclose(shared_out.loss, ref_out.loss, atol=1e-7, rtol=1e-6)
    for key in ("loss_fast", "loss_flow", "loss_camera_pose"):
        assert torch.allclose(shared_out.metrics[key], ref_out.metrics[key], atol=1e-7, rtol=1e-6), key

    reference_grads = {name: param.grad for name, param in reference.named_parameters()}
    for name, param in shared.named_parameters():
        ref_grad = reference_grads[name]
        assert (param.grad is None) == (ref_grad is None), name
        if param.grad is not None:
            assert torch.allclose(param.grad, ref_grad, atol=1e-7, rtol=1e-5), name


def test_trailing_token_trim_preserves_joint_loss_and_gradients():
    torch.manual_seed(23)
    reference = _make_joint_model()
    trimmed = _make_joint_model()
    trimmed.load_state_dict(reference.state_dict())
    reference._trim_trailing_token_padding = False
    trimmed._trim_trailing_token_padding = True

    obs = _make_joint_observation([0, 1])
    actions = torch.zeros(2, 10, 32)
    noise = torch.full_like(actions, 0.25)
    time = torch.full((2,), 0.5)

    trimmed_prompt = trimmed._trim_observation_prompt(obs)
    assert trimmed_prompt.tokenized_prompt.shape[1] == 5
    assert trimmed_prompt.flow_condition_tokens.shape[1] == 5
    condition = trimmed._trim_token_tensors(
        obs.flow_condition_tokens,
        obs.flow_condition_mask,
        obs.flow_condition_ar_mask,
    )
    assert all(value.shape[1] == 3 for value in condition)

    ref_out = reference.forward(obs, actions, noise=noise, time=time)
    ref_out.loss.backward()
    trim_out = trimmed.forward(obs, actions, noise=noise, time=time)
    trim_out.loss.backward()

    assert torch.allclose(trim_out.loss, ref_out.loss, atol=1e-7, rtol=1e-6)
    for key in ("loss_subtask", "loss_fast", "loss_flow", "loss_camera_pose"):
        assert torch.allclose(trim_out.metrics[key], ref_out.metrics[key], atol=1e-7, rtol=1e-6), key

    reference_grads = {name: param.grad for name, param in reference.named_parameters()}
    for name, param in trimmed.named_parameters():
        ref_grad = reference_grads[name]
        assert (param.grad is None) == (ref_grad is None), name
        if param.grad is not None:
            assert torch.allclose(param.grad, ref_grad, atol=1e-7, rtol=1e-5), name


def test_outer_image_checkpoint_switch_preserves_output_and_input_gradient():
    torch.manual_seed(29)
    checkpointed = _make_joint_model()
    direct = _make_joint_model()
    direct.load_state_dict(checkpointed.state_dict())
    checkpointed.gradient_checkpointing_enabled = True
    direct.gradient_checkpointing_enabled = True
    checkpointed._checkpoint_whole_image_encoder = True
    direct._checkpoint_whole_image_encoder = False

    image_a = torch.randn(2, 3, 8, 8, requires_grad=True)
    image_b = image_a.detach().clone().requires_grad_(True)
    out_a = checkpointed._embed_image(image_a)
    out_b = direct._embed_image(image_b)
    out_a.square().mean().backward()
    out_b.square().mean().backward()

    assert torch.allclose(out_a, out_b, atol=1e-7, rtol=1e-6)
    assert torch.allclose(image_a.grad, image_b.grad, atol=1e-7, rtol=1e-6)


def test_lightweight_checkpoint_switch_preserves_joint_loss_and_gradients():
    obs = _make_joint_observation([0, 1])
    actions = torch.zeros(2, 10, 32)
    noise = torch.full_like(actions, 0.25)
    time = torch.full((2,), 0.5)

    checkpointed = _make_joint_model()
    direct = _make_joint_model()
    direct.load_state_dict(checkpointed.state_dict())
    checkpointed.gradient_checkpointing_enabled = True
    direct.gradient_checkpointing_enabled = True
    checkpointed._checkpoint_lightweight_ops = True
    direct._checkpoint_lightweight_ops = False

    out_a = checkpointed.forward(obs, actions, noise=noise, time=time)
    out_b = direct.forward(obs, actions, noise=noise, time=time)
    out_a.loss.backward()
    out_b.loss.backward()

    torch.testing.assert_close(out_a.loss, out_b.loss, atol=1e-7, rtol=1e-6)
    for name, param_a in checkpointed.named_parameters():
        param_b = dict(direct.named_parameters())[name]
        if param_a.grad is None or param_b.grad is None:
            assert param_a.grad is None and param_b.grad is None, name
        else:
            torch.testing.assert_close(param_a.grad, param_b.grad, atol=1e-6, rtol=1e-5, msg=name)


def test_joint_mixed_batch_reports_both_subsets_and_finite_gradients():
    model = _make_joint_model()
    obs = _make_joint_observation([0, 1])
    out = model.forward(obs, torch.zeros(2, 10, 32))
    out.loss.backward()

    assert torch.isfinite(out.loss)
    assert float(out.metrics["high_level_examples"]) == 1.0
    assert float(out.metrics["low_level_examples"]) == 1.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


def test_joint_objective_reports_explicit_detached_branch_metrics():
    model = _make_joint_model()
    obs = _make_joint_observation([0, 1])
    out = model.forward(obs, torch.zeros(2, 10, 32))

    assert out.loss.requires_grad
    # Compatibility field remains detached and ambiguous; validation should use
    # the explicit branch metrics below.
    assert not out.loss_action.requires_grad
    for key in (
        "loss_subtask",
        "loss_fast",
        "loss_flow",
        "loss_camera_pose",
        "loss_non_camera",
        "loss_weighted_subtask",
        "loss_weighted_fast",
        "loss_weighted_flow",
        "loss_weighted_camera",
    ):
        assert key in out.metrics
        assert torch.isfinite(out.metrics[key])
        assert not out.metrics[key].requires_grad

    model.zero_grad(set_to_none=True)
    out.loss.backward()
    assert model.camera_head.pose_branch.fc2.weight.grad is not None
    assert model.action_out_proj.weight.grad is not None


def test_joint_flow_conditioning_never_sees_fast_target_tokens():
    """The Action Expert must not attend to the answer it regresses."""
    model = _make_joint_model()
    obs = _make_joint_observation([1])
    seen_prompts = []
    original = model._preprocess_observation

    def spy(observation, train=True):
        seen_prompts.append(observation.tokenized_prompt.clone())
        return original(observation, train=train)

    model._preprocess_observation = spy
    model.forward(obs, torch.zeros(1, 10, 32))

    assert len(seen_prompts) == 2, "low-level examples run a FAST pass and a flow pass"
    fast_prompt, flow_prompt = seen_prompts
    fast_targets = fast_prompt[obs.token_loss_mask]
    flow_valid = flow_prompt[obs.flow_condition_mask]
    assert not set(fast_targets.tolist()) & set(flow_valid.tolist())


def test_joint_camera_loss_counted_once_per_example_not_per_forward():
    """Low-level examples run twice; camera must not get double weight."""
    model = _make_joint_model()
    obs = _make_joint_observation([1])
    camera_calls = {"n": 0}
    original = model._camera_loss_from_prefix_out

    def counting(*args, **kwargs):
        camera_calls["n"] += 1
        return original(*args, **kwargs)

    model._camera_loss_from_prefix_out = counting
    model.forward(obs, torch.zeros(1, 10, 32))
    assert camera_calls["n"] == 1


def test_joint_loss_weights_scale_their_terms():
    """Zeroing a weight must remove exactly that term from the total.

    The flow pass samples noise and time, so both runs are given fixed values;
    otherwise the comparison would drift by the sampling noise alone.
    """
    obs = _make_joint_observation([0, 1])
    actions = torch.zeros(2, 10, 32)
    noise = torch.ones(2, 10, 32) * 0.25
    time = torch.full((2,), 0.5)

    base = _make_joint_model()
    ref = base.forward(obs, actions, noise=noise, time=time)

    scaled = _make_joint_model()
    scaled.load_state_dict(base.state_dict())
    scaled.config.joint_loss_weight_subtask = 0.0
    out = scaled.forward(obs, actions, noise=noise, time=time)

    expected = float(ref.loss) - float(ref.metrics["loss_subtask"])
    assert abs(float(out.loss) - expected) < 1e-4


def test_vlm_discrete_camera_behavior_is_unchanged():
    """The old objective must keep its exact prefix-only semantics."""
    model = _make_tiny_pi0(enable_camera=True)
    model.config.training_objective = "vlm_discrete_camera"
    model.freeze_for_vlm_discrete_camera()
    obs = _make_tiny_vlm_observation()
    out = model.forward(obs, torch.zeros(1, 50, 32))
    out.loss.backward()

    assert torch.isfinite(out.loss)
    assert model.paligemma_with_expert.action_expert_call_count == 0
    assert model.action_in_proj.weight.grad is None
    assert model.front_camera_token_embeddings.grad.abs().sum() > 0


def test_joint_objective_keeps_action_expert_trainable():
    """human_joint_camera must NOT freeze the continuous action path."""
    model = _make_joint_model()
    # freeze_for_vlm_discrete_camera is only for the old objective; the joint
    # objective leaves these trainable.
    for name, param in model.named_parameters():
        if name.startswith(("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")):
            assert param.requires_grad, f"{name} must stay trainable for human_joint_camera"


def test_old_objective_still_freezes_the_action_expert_path():
    model = _make_tiny_pi0(enable_camera=True)
    model.config.training_objective = "vlm_discrete_camera"
    model.freeze_for_vlm_discrete_camera()
    for name, param in model.named_parameters():
        if name.startswith(("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")):
            assert not param.requires_grad, f"{name} must stay frozen for vlm_discrete_camera"


def test_high_level_only_batch_leaves_action_expert_without_gradient():
    model = _make_joint_model()
    obs = _make_joint_observation([0, 0])
    out = model.forward(obs, torch.zeros(2, 10, 32))
    out.loss.backward()

    assert model.paligemma_with_expert.action_expert_call_count == 0
    for name, param in model.named_parameters():
        if name.startswith(("action_in_proj", "action_out_proj")):
            grad = param.grad
            assert grad is None or grad.abs().sum() == 0, f"{name} got gradient from a high-level-only batch"
    # The camera and language paths still learn.
    assert model.camera_head.pose_branch.fc2.weight.grad.abs().sum() > 0
    assert model.front_camera_token_embeddings.grad.abs().sum() > 0


def test_low_level_batch_gives_finite_nonzero_gradient_to_every_expected_module():
    model = _make_joint_model()
    obs = _make_joint_observation([1, 1])
    out = model.forward(obs, torch.zeros(2, 10, 32))
    out.loss.backward()

    expected = {
        "action_in_proj": model.action_in_proj.weight,
        "action_out_proj": model.action_out_proj.weight,
        "time_mlp_in": model.time_mlp_in.weight,
        "time_mlp_out": model.time_mlp_out.weight,
        "camera_projector": model.camera_projector.weight,
        "camera_head": model.camera_head.pose_branch.fc2.weight,
        "front_camera_token_embeddings": model.front_camera_token_embeddings,
    }
    for name, param in expected.items():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient is not finite"
        assert param.grad.abs().sum() > 0, f"{name} gradient is all zero"
