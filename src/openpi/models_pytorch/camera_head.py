from __future__ import annotations

from collections.abc import Callable
import dataclasses
import re

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812


@dataclasses.dataclass(frozen=True)
class FrontCameraTokenLayout:
    valid_mask: torch.Tensor
    type_ids: torch.Tensor
    token_indices: torch.Tensor


@dataclasses.dataclass
class Pi0LossOutput:
    loss: torch.Tensor
    loss_action: torch.Tensor
    loss_camera_pose: torch.Tensor
    metrics: dict[str, torch.Tensor]


@dataclasses.dataclass(frozen=True)
class CheckpointLoadReport:
    checkpoint_tensor_count: int
    loaded_tensor_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]


ALLOWED_CAMERA_MISSING_PATTERNS = (
    re.compile(r"front_camera_token_embeddings"),
    re.compile(r"front_camera_temporal_embeddings"),
    re.compile(r"camera_projector\..*"),
    re.compile(r"camera_head\..*"),
)

TIED_WEIGHT_ALIASES = {
    "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight": (
        "paligemma_with_expert.paligemma.lm_head.weight"
    ),
}


def build_front_camera_token_layout(
    history_valid_mask: torch.Tensor,
    *,
    image_patch_count: int | None = None,
) -> FrontCameraTokenLayout:
    """Builds per-slot camera-token bookkeeping.

    type_ids are 0 for C_first, 1 for C_rest, and -1 for invalid history slots.
    If image_patch_count is provided, token_indices are absolute prefix positions
    for [front patches, C] repeated over the four history slots.
    """
    if history_valid_mask.ndim != 2:
        raise ValueError(f"history_valid_mask must be [B, S], got {history_valid_mask.shape}")
    valid_mask = history_valid_mask.to(torch.bool)
    first_valid = valid_mask & (torch.cumsum(valid_mask.to(torch.int64), dim=1) == 1)
    type_ids = torch.where(
        valid_mask,
        torch.where(
            first_valid,
            torch.zeros_like(valid_mask, dtype=torch.long),
            torch.ones_like(valid_mask, dtype=torch.long),
        ),
        torch.full_like(valid_mask, -1, dtype=torch.long),
    )

    slot_indices = torch.arange(valid_mask.shape[1], device=valid_mask.device, dtype=torch.long)
    if image_patch_count is None:
        token_indices = slot_indices[None, :].expand_as(type_ids)
    else:
        token_indices = ((slot_indices + 1) * image_patch_count + slot_indices)[None, :].expand_as(type_ids)
    return FrontCameraTokenLayout(valid_mask=valid_mask, type_ids=type_ids, token_indices=token_indices)


def make_front_camera_prefix_att_masks(
    history_valid_mask: torch.Tensor,
    *,
    front_patch_count: int,
    wrist_patch_counts: tuple[int, ...],
    wrist_valid_masks: tuple[torch.Tensor, ...],
    lang_mask: torch.Tensor,
    lang_att_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, FrontCameraTokenLayout]:
    """Returns pad and AR-boundary masks for the π0.5 front-camera prefix.

    History blocks are causal across time. After the final camera token, left
    wrist, right wrist, and language/state remain one bidirectional prefix block,
    matching the original π0.5 post-image prefix semantics. For discrete VLM
    pretraining, `lang_att_mask` may be provided with standard prefix-LM
    semantics: textual prompt tokens are 0, response tokens are 1. This lets
    response tokens attend visual/camera/prompt context causally while preventing
    camera/prompt prefix tokens from attending response targets.
    """
    layout = build_front_camera_token_layout(history_valid_mask, image_patch_count=front_patch_count)
    bsz, n_slots = layout.valid_mask.shape
    device = layout.valid_mask.device
    pad_chunks: list[torch.Tensor] = []
    att_values: list[int] = []

    for slot in range(n_slots):
        slot_mask = layout.valid_mask[:, slot]
        pad_chunks.append(slot_mask[:, None].expand(bsz, front_patch_count))
        # Slot 0 front patches share the initial block. Later front frames start
        # a new block after the previous camera token.
        att_values.extend([1 if slot > 0 and i == 0 else 0 for i in range(front_patch_count)])
        pad_chunks.append(slot_mask[:, None])
        # Every camera token starts a boundary so later history can see it, while
        # that token cannot see future history or post-camera context.
        att_values.append(1)

    post_camera_started = False
    for wrist_count, wrist_mask in zip(wrist_patch_counts, wrist_valid_masks, strict=True):
        wrist_mask = wrist_mask.to(torch.bool)
        pad_chunks.append(wrist_mask[:, None].expand(bsz, wrist_count))
        att_values.extend([0 if post_camera_started or i > 0 else 1 for i in range(wrist_count)])
        post_camera_started = post_camera_started or wrist_count > 0

    lang_mask = lang_mask.to(torch.bool)
    pad_chunks.append(lang_mask)
    if lang_mask.shape[1] > 0:
        if lang_att_mask is None:
            att_values.extend([0 if post_camera_started or i > 0 else 1 for i in range(lang_mask.shape[1])])
        else:
            if lang_att_mask.shape != lang_mask.shape:
                raise ValueError(f"lang_att_mask shape {lang_att_mask.shape} must match lang_mask {lang_mask.shape}")
            lang_att = lang_att_mask.to(device=device, dtype=torch.bool)
            if not post_camera_started:
                # No wrist view carried the post-camera block boundary (all
                # fully masked views were skipped), so the first language token
                # must open it; otherwise prompt tokens would join the final
                # history block and camera tokens could attend the prompt.
                lang_att = lang_att.clone()
                lang_att[:, 0] = True
            pad_masks = torch.cat(pad_chunks, dim=1)
            base_att = torch.tensor(att_values, dtype=torch.bool, device=device)[None, :].expand(bsz, -1)
            att_masks = torch.cat([base_att, lang_att], dim=1)
            return pad_masks, att_masks, layout

    pad_masks = torch.cat(pad_chunks, dim=1)
    att_masks = torch.tensor(att_values, dtype=torch.bool, device=device)[None, :].expand(bsz, -1)
    return pad_masks, att_masks, layout


def make_front_history_prefix_att_masks(
    history_valid_mask: torch.Tensor,
    *,
    front_patch_count: int,
    wrist_patch_counts: tuple[int, ...],
    wrist_valid_masks: tuple[torch.Tensor, ...],
    lang_mask: torch.Tensor,
    lang_att_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build temporal prefix masks for history images without camera tokens."""
    if history_valid_mask.ndim != 2:
        raise ValueError(f"history_valid_mask must be [B, S], got {history_valid_mask.shape}")
    valid_mask = history_valid_mask.to(torch.bool)
    bsz, n_slots = valid_mask.shape
    device = valid_mask.device
    pad_chunks: list[torch.Tensor] = []
    att_values: list[int] = []

    for slot in range(n_slots):
        slot_mask = valid_mask[:, slot]
        pad_chunks.append(slot_mask[:, None].expand(bsz, front_patch_count))
        att_values.extend([1 if slot > 0 and i == 0 else 0 for i in range(front_patch_count)])

    post_history_started = False
    for wrist_count, wrist_mask in zip(wrist_patch_counts, wrist_valid_masks, strict=True):
        wrist_mask_bool = wrist_mask.to(torch.bool)
        pad_chunks.append(wrist_mask_bool[:, None].expand(bsz, wrist_count))
        att_values.extend([0 if post_history_started or i > 0 else 1 for i in range(wrist_count)])
        post_history_started = post_history_started or wrist_count > 0

    lang_mask = lang_mask.to(torch.bool)
    pad_chunks.append(lang_mask)
    if lang_mask.shape[1] > 0:
        if lang_att_mask is None:
            att_values.extend([0 if post_history_started or i > 0 else 1 for i in range(lang_mask.shape[1])])
        else:
            if lang_att_mask.shape != lang_mask.shape:
                raise ValueError(f"lang_att_mask shape {lang_att_mask.shape} must match lang_mask {lang_mask.shape}")
            lang_att = lang_att_mask.to(device=device, dtype=torch.bool)
            if not post_history_started:
                lang_att = lang_att.clone()
                lang_att[:, 0] = True
            pad_masks = torch.cat(pad_chunks, dim=1)
            base_att = torch.tensor(att_values, dtype=torch.bool, device=device)[None, :].expand(bsz, -1)
            return pad_masks, torch.cat([base_att, lang_att], dim=1)

    pad_masks = torch.cat(pad_chunks, dim=1)
    att_masks = torch.tensor(att_values, dtype=torch.bool, device=device)[None, :].expand(bsz, -1)
    return pad_masks, att_masks


class LayerScale(nn.Module):
    """Cambrian-P/VGGT LayerScale."""

    def __init__(self, dim: int, init_values: float | Tensor = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Mlp(nn.Module):
    """Cambrian-P/VGGT MLP."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Cambrian-P/VGGT attention plus optional camera-slot key padding."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
        is_causal: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim should be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        self.is_causal = is_causal
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, valid_mask: torch.Tensor | None = None) -> Tensor:
        bsz, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if valid_mask is None and self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=self.is_causal,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            allowed = torch.ones((bsz, 1, seq_len, seq_len), dtype=torch.bool, device=attn.device)
            if self.is_causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=attn.device),
                    diagonal=1,
                )
                allowed = allowed & ~causal_mask[None, None, :, :]
            if valid_mask is not None:
                safe_valid = valid_mask.to(torch.bool)
                allowed = allowed & safe_valid[:, None, None, :]
                empty_queries = ~allowed.any(dim=-1, keepdim=True)
                if empty_queries.any():
                    eye = torch.eye(seq_len, dtype=torch.bool, device=attn.device)[None, None, :, :]
                    allowed = allowed | (empty_queries & eye)
            attn = attn.masked_fill(~allowed, float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(bsz, seq_len, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Cambrian-P/VGGT transformer block plus optional valid-slot masking."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
        is_causal: bool = False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
            is_causal=is_causal,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x: Tensor, pos=None, valid_mask: torch.Tensor | None = None) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), pos=pos, valid_mask=valid_mask))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        if valid_mask is not None:
            x = torch.where(valid_mask[:, :, None].to(torch.bool), x, torch.zeros_like(x))
        return x


class CameraHead(nn.Module):
    """Cambrian-P/VGGT iterative camera head for [xyz, qx, qy, qz, qw, fov_y, fov_x]."""

    def __init__(
        self,
        dim_in: int = 2048,
        *,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",
        is_causal: bool = True,
        num_iterations: int = 4,
    ):
        super().__init__()
        if pose_encoding_type != "absT_quaR_FoV":
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")
        self.target_dim = 9
        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        self.is_causal = is_causal
        self.num_iterations = num_iterations
        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=dim_in,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    is_causal=is_causal,
                )
                for _ in range(trunk_depth)
            ]
        )
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    def forward(
        self,
        aggregated_tokens_list: torch.Tensor | list[torch.Tensor],
        *,
        valid_mask: torch.Tensor | None = None,
        num_iterations: int | None = None,
    ) -> list[torch.Tensor]:
        if isinstance(aggregated_tokens_list, list):
            tokens = aggregated_tokens_list[-1]
            if tokens.ndim == 4:
                tokens = tokens[:, :, 0]
        else:
            tokens = aggregated_tokens_list
        if tokens.ndim != 3:
            raise ValueError(f"camera features must be [B,S,C] or [B,S,1,C], got {tokens.shape}")

        pose_tokens = self.token_norm(tokens)
        block_valid_mask = None
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=pose_tokens.device, dtype=torch.bool)
            pose_tokens = torch.where(valid_mask[:, :, None], pose_tokens, torch.zeros_like(pose_tokens))
            block_valid_mask = None if bool(valid_mask.all().item()) else valid_mask
        return self.trunk_fn(
            pose_tokens,
            num_iterations=num_iterations or self.num_iterations,
            valid_mask=block_valid_mask,
            output_valid_mask=valid_mask,
        )

    def trunk_fn(
        self,
        pose_tokens: torch.Tensor,
        *,
        num_iterations: int,
        valid_mask: torch.Tensor | None = None,
        output_valid_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        batch_size, seq_len, _ = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list: list[torch.Tensor] = []

        for _ in range(num_iterations):
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(batch_size, seq_len, -1))
            else:
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens
            if valid_mask is not None:
                pose_tokens_modulated = torch.where(
                    valid_mask[:, :, None],
                    pose_tokens_modulated,
                    torch.zeros_like(pose_tokens_modulated),
                )

            for block in self.trunk:
                pose_tokens_modulated = block(pose_tokens_modulated, valid_mask=valid_mask)

            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))
            pred_pose_enc = pred_pose_enc_delta if pred_pose_enc is None else pred_pose_enc + pred_pose_enc_delta
            activated_pose = activate_pose(
                pred_pose_enc,
                trans_act=self.trans_act,
                quat_act=self.quat_act,
                fl_act=self.fl_act,
            )
            if output_valid_mask is not None:
                activated_pose = torch.where(
                    output_valid_mask[:, :, None],
                    activated_pose,
                    torch.zeros_like(activated_pose),
                )
            pred_pose_enc_list.append(activated_pose)

        return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def base_pose_act(pose_enc: torch.Tensor, act_type: str = "linear") -> torch.Tensor:
    if act_type == "linear":
        return pose_enc
    if act_type == "inv_log":
        return torch.sign(pose_enc) * torch.expm1(torch.abs(pose_enc))
    if act_type == "exp":
        return torch.exp(pose_enc)
    if act_type == "relu":
        return F.relu(pose_enc)
    raise ValueError(f"Unknown act_type: {act_type}")


def activate_pose(pred_pose_enc: torch.Tensor, trans_act: str = "linear", quat_act: str = "linear", fl_act: str = "relu"):
    trans = base_pose_act(pred_pose_enc[..., :3], trans_act)
    quat = base_pose_act(pred_pose_enc[..., 3:7], quat_act)
    fov = base_pose_act(pred_pose_enc[..., 7:9], fl_act)
    return torch.cat([trans, quat, fov], dim=-1)


def standardize_quaternion_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return torch.where(quat[..., 3:4] < 0, -quat, quat)


def canonicalize_gt_quaternion_xyzw(quat: torch.Tensor) -> torch.Tensor:
    quat = F.normalize(quat.to(torch.float32), dim=-1, eps=1e-6)
    return standardize_quaternion_xyzw(quat)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def matrix_to_quaternion_xyzw(matrix: torch.Tensor) -> torch.Tensor:
    """Cambrian-P/VGGT mat_to_quat, scalar-last xyzw, canonicalized to qw >= 0."""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))
    out = out[..., [1, 2, 3, 0]]
    return standardize_quaternion_xyzw(out)


def ref_from_camera_matrix_to_pose7(ref_from_camera: torch.Tensor) -> torch.Tensor:
    """Interprets input as T_ref_from_camera and returns [xyz, qx, qy, qz, qw]."""
    if ref_from_camera.shape[-1] == 16:
        ref_from_camera = ref_from_camera.reshape(*ref_from_camera.shape[:-1], 4, 4)
    elif ref_from_camera.shape[-1] == 12:
        ref_from_camera = ref_from_camera.reshape(*ref_from_camera.shape[:-1], 3, 4)

    if ref_from_camera.shape[-2:] == (4, 4):
        rot = ref_from_camera[..., :3, :3]
        trans = ref_from_camera[..., :3, 3]
    elif ref_from_camera.shape[-2:] == (3, 4):
        rot = ref_from_camera[..., :3, :3]
        trans = ref_from_camera[..., :3, 3]
    else:
        raise ValueError(f"Unsupported T_ref_from_camera shape: {ref_from_camera.shape}")
    return torch.cat([trans.to(torch.float32), matrix_to_quaternion_xyzw(rot)], dim=-1)


def intrinsics_to_fov(intrinsics: torch.Tensor, image_hw: torch.Tensor) -> torch.Tensor:
    """Returns principal-point-aware physical [fov_y, fov_x] in radians."""
    intrinsics = intrinsics.to(torch.float32)
    image_hw = image_hw.to(torch.float32)
    h = image_hw[..., 0]
    w = image_hw[..., 1]
    fx = intrinsics[..., 0, 0].clamp(min=1e-6)
    fy = intrinsics[..., 1, 1].clamp(min=1e-6)
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    fov_y = torch.atan2(cy, fy) + torch.atan2((h - 1.0) - cy, fy)
    fov_x = torch.atan2(cx, fx) + torch.atan2((w - 1.0) - cx, fx)
    return torch.stack([fov_y, fov_x], dim=-1)


def camera_pose_loss(
    pred_pose_list: list[torch.Tensor] | torch.Tensor,
    target_pose9: torch.Tensor,
    *,
    pose_valid_mask: torch.Tensor,
    fov_valid_mask: torch.Tensor | None = None,
    trans_weight: float = 1.0,
    rot_weight: float = 1.0,
    fov_weight: float = 0.5,
    gamma: float = 0.6,
    normalize_trans: bool = False,
    d_bar_floor: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_list = pred_pose_list if isinstance(pred_pose_list, list) else [pred_pose_list]
    target = target_pose9.to(torch.float32)
    pose_valid = pose_valid_mask.to(torch.bool)
    fov_valid = pose_valid if fov_valid_mask is None else (pose_valid & fov_valid_mask.to(torch.bool))
    zero = pred_list[-1].sum() * 0.0

    d_bar = torch.ones(target.shape[:2], device=target.device, dtype=torch.float32)
    if normalize_trans:
        d_bar = _translation_d_bar(target[..., :3], pose_valid, floor=d_bar_floor)[:, None].expand_as(pose_valid)

    total_t = zero
    total_r = zero
    total_fov = zero
    n_stages = len(pred_list)
    for stage_idx, pred_pose9 in enumerate(pred_list):
        pred = pred_pose9.to(torch.float32)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        if pose_valid.any():
            err_t = (pred[..., :3][pose_valid] - target[..., :3][pose_valid]).abs().clamp(max=100)
            if normalize_trans:
                err_t = err_t / d_bar[pose_valid].unsqueeze(-1)
            trans_loss = err_t.mean()
            target_q = canonicalize_gt_quaternion_xyzw(target[..., 3:7])
            rot_loss = (pred[..., 3:7][pose_valid] - target_q[pose_valid]).abs().mean()
        else:
            trans_loss = zero
            rot_loss = zero
        if fov_valid.any():
            fov_loss = (pred[..., 7:9][fov_valid] - target[..., 7:9][fov_valid]).abs().mean()
        else:
            fov_loss = zero

        total_t = total_t + trans_loss * stage_weight
        total_r = total_r + rot_loss * stage_weight
        total_fov = total_fov + fov_loss * stage_weight

    loss_t = total_t / n_stages
    loss_r = total_r / n_stages
    loss_fov = total_fov / n_stages
    total = trans_weight * loss_t + rot_weight * loss_r + fov_weight * loss_fov

    # Unreduced numerator/denominator for DDP-correct logging. A mean of
    # per-rank means is not the mean: a rank holding 2 valid poses would count
    # as much as one holding 200. Only summing numerators and denominators
    # separately, then dividing once after the all-reduce, gives the true
    # global conditional mean. Taken from the FINAL stage so the reported
    # error is the model's actual output, not a gamma-weighted stage blend.
    final = pred_list[-1].to(torch.float32)
    if pose_valid.any():
        e_t = (final[..., :3][pose_valid] - target[..., :3][pose_valid]).abs().clamp(max=100)
        if normalize_trans:
            e_t = e_t / d_bar[pose_valid].unsqueeze(-1)
        tq = canonicalize_gt_quaternion_xyzw(target[..., 3:7])
        e_r = (final[..., 3:7][pose_valid] - tq[pose_valid]).abs()
        trans_sum, trans_n = e_t.sum().detach(), torch.tensor(
            float(e_t.numel()), device=e_t.device
        )
        rot_sum, rot_n = e_r.sum().detach(), torch.tensor(float(e_r.numel()), device=e_r.device)
    else:
        trans_sum = rot_sum = zero.detach()
        trans_n = rot_n = torch.zeros((), device=target.device)
    if fov_valid.any():
        e_f = (final[..., 7:9][fov_valid] - target[..., 7:9][fov_valid]).abs()
        fov_sum, fov_n = e_f.sum().detach(), torch.tensor(float(e_f.numel()), device=e_f.device)
    else:
        fov_sum = zero.detach()
        fov_n = torch.zeros((), device=target.device)

    metrics = {
        "loss_camera_trans": loss_t.detach(),
        "loss_camera_rot": loss_r.detach(),
        "loss_camera_fov": loss_fov.detach(),
        "camera_pose_n_valid": pose_valid.sum().detach().to(torch.float32),
        "camera_fov_n_valid": fov_valid.sum().detach().to(torch.float32),
        "camera_trans_sum": trans_sum.to(torch.float32),
        "camera_trans_count": trans_n.to(torch.float32),
        "camera_rot_sum": rot_sum.to(torch.float32),
        "camera_rot_count": rot_n.to(torch.float32),
        "camera_fov_sum": fov_sum.to(torch.float32),
        "camera_fov_count": fov_n.to(torch.float32),
    }
    return total, metrics


def _translation_d_bar(gt_translations: torch.Tensor, valid_mask: torch.Tensor, floor: float) -> torch.Tensor:
    batch_size, _, _ = gt_translations.shape
    floor_t = torch.tensor(float(floor), device=gt_translations.device, dtype=gt_translations.dtype)
    d_bar = floor_t.expand(batch_size).clone()
    for batch_idx in range(batch_size):
        valid_idx = torch.nonzero(valid_mask[batch_idx], as_tuple=False).flatten()
        if valid_idx.numel() < 2:
            continue
        t_valid = gt_translations[batch_idx, valid_idx]
        mean_step = (t_valid[1:] - t_valid[:-1]).norm(dim=-1).mean()
        if torch.isfinite(mean_step):
            d_bar[batch_idx] = torch.maximum(mean_step, floor_t)
    return d_bar


def apply_action_dim_mask_for_flow(
    actions: torch.Tensor,
    noise: torch.Tensor,
    action_mask: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masks padded action dimensions/timesteps before flow matching construction."""
    mask = action_mask.to(device=actions.device, dtype=torch.bool)
    noise_masked = torch.where(mask, noise, torch.zeros_like(noise))
    actions_masked = torch.where(mask, actions, torch.zeros_like(actions))
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise_masked + (1 - time_expanded) * actions_masked
    u_t = torch.where(mask, noise_masked - actions_masked, torch.zeros_like(actions_masked))
    return noise_masked, actions_masked, x_t, u_t


def combine_action_dim_time_masks(
    actions: torch.Tensor,
    *,
    action_dim_mask: torch.Tensor | None = None,
    action_time_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Builds the final positive-valid [B,H,D] action mask."""
    if action_dim_mask is None:
        dim_mask = torch.ones_like(actions, dtype=torch.bool)
    else:
        dim_mask = action_dim_mask.to(device=actions.device, dtype=torch.bool)
        if dim_mask.ndim == 2:
            dim_mask = dim_mask[:, None, :]
        if dim_mask.shape[-1] < actions.shape[-1]:
            dim_mask = F.pad(dim_mask, (0, actions.shape[-1] - dim_mask.shape[-1]), value=False)
        if dim_mask.shape[1] == 1:
            dim_mask = dim_mask.expand(actions.shape[0], actions.shape[1], actions.shape[2])
        if dim_mask.shape != actions.shape:
            raise ValueError(f"action_dim_mask shape {dim_mask.shape} does not match actions {actions.shape}")

    if action_time_valid_mask is None:
        time_mask = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
    else:
        time_mask = action_time_valid_mask.to(device=actions.device, dtype=torch.bool)
        if time_mask.ndim == 3 and time_mask.shape[-1] == 1:
            time_mask = time_mask[..., 0]
        if time_mask.ndim != 2:
            raise ValueError(f"action_time_valid_mask must be [B,H], got {time_mask.shape}")
        if time_mask.shape[1] < actions.shape[1]:
            time_mask = F.pad(time_mask, (0, actions.shape[1] - time_mask.shape[1]), value=False)
        time_mask = time_mask[:, : actions.shape[1]]
        if time_mask.shape != actions.shape[:2]:
            raise ValueError(f"action_time_valid_mask shape {time_mask.shape} does not match actions {actions.shape[:2]}")

    return dim_mask & time_mask[:, :, None]


def apply_action_mask_for_denoise_update(
    x_t: torch.Tensor,
    v_t: torch.Tensor,
    dt: torch.Tensor | float,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Euler update for inference while keeping invalid action dims/timesteps exactly zero."""
    mask = action_mask.to(device=x_t.device, dtype=torch.bool)
    v_t = torch.where(mask, v_t, torch.zeros_like(v_t))
    return torch.where(mask, x_t + dt * v_t, torch.zeros_like(x_t))


def load_state_dict_with_camera_whitelist(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    allow_camera_missing: bool,
) -> CheckpointLoadReport:
    model_state = model.state_dict()
    state_dict = dict(state_dict)
    for missing_key, source_key in TIED_WEIGHT_ALIASES.items():
        if (
            missing_key not in state_dict
            and source_key in state_dict
            and missing_key in model_state
            and tuple(state_dict[source_key].shape) == tuple(model_state[missing_key].shape)
        ):
            state_dict[missing_key] = state_dict[source_key]

    checkpoint_keys = set(state_dict)
    model_keys = set(model_state)
    unexpected = tuple(sorted(checkpoint_keys - model_keys))
    raw_missing = tuple(sorted(model_keys - checkpoint_keys))

    shape_mismatches = tuple(
        sorted(
            key
            for key in checkpoint_keys & model_keys
            if tuple(state_dict[key].shape) != tuple(model_state[key].shape)
        )
    )

    def is_allowed_missing(key: str) -> bool:
        return allow_camera_missing and any(pattern.fullmatch(key) for pattern in ALLOWED_CAMERA_MISSING_PATTERNS)

    bad_missing = tuple(key for key in raw_missing if not is_allowed_missing(key))
    if unexpected or bad_missing or shape_mismatches:
        raise RuntimeError(
            "Checkpoint is incompatible: "
            f"missing={bad_missing}, unexpected={unexpected}, shape_mismatches={shape_mismatches}"
        )

    loadable = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    model.load_state_dict(loadable, strict=False)
    return CheckpointLoadReport(
        checkpoint_tensor_count=len(state_dict),
        loaded_tensor_count=len(loadable),
        missing_keys=raw_missing,
        unexpected_keys=unexpected,
        shape_mismatches=shape_mismatches,
    )
