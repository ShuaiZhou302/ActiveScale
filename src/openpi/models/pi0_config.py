import dataclasses
from typing import TYPE_CHECKING
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"
    # Explicit opt-in training objective. The default preserves the released
    # pi0/pi0.5 flow-matching behavior.
    training_objective: Literal["flow_camera", "vlm_discrete_camera", "human_joint_camera"] = "flow_camera"

    # Optional APVLA front-history path. Camera tokens imply history input for
    # backward compatibility; history can also be enabled alone for ablation.
    enable_front_history_images: bool = False
    enable_front_camera_tokens: bool = False
    front_camera_history_offsets: tuple[int, ...] = (-48, -32, -16, 0)
    front_camera_pose_loss_weight: float = 0.2
    # human_joint_camera loss weights. Each term is normalized by its own
    # valid examples/tokens/dims before these outer weights apply, so a view
    # with longer targets cannot dominate purely through target length.
    joint_loss_weight_subtask: float = 1.0
    joint_loss_weight_fast: float = 1.0
    joint_loss_weight_flow: float = 1.0
    front_camera_pose_embed_dim: int = 2048
    front_camera_pose_trunk_depth: int = 4
    front_camera_pose_num_heads: int = 16
    front_camera_pose_mlp_ratio: int = 4
    front_camera_pose_num_iterations: int = 4
    front_camera_pose_causal_attn: bool = True
    front_camera_pose_loss_gamma: float = 0.6
    front_camera_pose_loss_weight_trans: float = 1.0
    front_camera_pose_loss_weight_rot: float = 1.0
    front_camera_pose_loss_weight_focal: float = 0.5
    front_camera_pose_loss_normalize_trans: bool = False
    front_camera_pose_loss_d_bar_floor: float = 0.01
    front_camera_use_temporal_embeddings: bool = False

    def __post_init__(self):
        if len(self.front_camera_history_offsets) != 4:
            raise ValueError("front_camera_history_offsets must contain exactly four slots")
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        if self.training_objective not in {"flow_camera", "vlm_discrete_camera", "human_joint_camera"}:
            raise ValueError(f"Unsupported training_objective={self.training_objective!r}")
        if self.training_objective == "human_joint_camera" and not self.pi05:
            raise ValueError("human_joint_camera requires pi05=True")
        if self.training_objective == "vlm_discrete_camera" and not self.pi05:
            raise ValueError("vlm_discrete_camera is only implemented for pi0.5")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        front_history_enabled = self.enable_front_history_images or self.enable_front_camera_tokens

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                vlm_view_type=jax.ShapeDtypeStruct([batch_size], jnp.int32),
                vlm_token_truncated=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
                front_history_images=(
                    jax.ShapeDtypeStruct(
                        [batch_size, len(self.front_camera_history_offsets), *_model.IMAGE_RESOLUTION, 3],
                        jnp.float32,
                    )
                    if front_history_enabled
                    else None
                ),
                front_history_masks=(
                    jax.ShapeDtypeStruct([batch_size, len(self.front_camera_history_offsets)], jnp.bool_)
                    if front_history_enabled
                    else None
                ),
                camera_extrinsics=(
                    jax.ShapeDtypeStruct([batch_size, len(self.front_camera_history_offsets), 4, 4], jnp.float32)
                    if self.enable_front_camera_tokens
                    else None
                ),
                camera_pose_valid=(
                    jax.ShapeDtypeStruct([batch_size, len(self.front_camera_history_offsets)], jnp.bool_)
                    if self.enable_front_camera_tokens
                    else None
                ),
                camera_fov=(
                    jax.ShapeDtypeStruct([batch_size, len(self.front_camera_history_offsets), 2], jnp.float32)
                    if self.enable_front_camera_tokens
                    else None
                ),
                camera_fov_valid=(
                    jax.ShapeDtypeStruct([batch_size, len(self.front_camera_history_offsets)], jnp.bool_)
                    if self.enable_front_camera_tokens
                    else None
                ),
                camera_image_hw=(
                    jax.ShapeDtypeStruct([batch_size, len(self.front_camera_history_offsets), 2], jnp.float32)
                    if self.enable_front_camera_tokens
                    else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
