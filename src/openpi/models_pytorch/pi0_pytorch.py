import copy
import dataclasses
import logging
import math
import os
from typing import Any

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.camera_head import CameraHead
from openpi.models_pytorch.camera_head import apply_action_mask_for_denoise_update
from openpi.models_pytorch.camera_head import Pi0LossOutput
from openpi.models_pytorch.camera_head import apply_action_dim_mask_for_flow
from openpi.models_pytorch.camera_head import build_front_camera_token_layout
from openpi.models_pytorch.camera_head import camera_pose_loss
from openpi.models_pytorch.camera_head import ref_from_camera_matrix_to_pose7
from openpi.models_pytorch.camera_head import combine_action_dim_time_masks
from openpi.models_pytorch.camera_head import intrinsics_to_fov
from openpi.models_pytorch.camera_head import make_front_camera_prefix_att_masks
from openpi.models_pytorch.camera_head import make_front_history_prefix_att_masks
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.rtc import guidance_scale as rtc_guidance_scale
from openpi.models_pytorch.rtc import guidance_vjp as rtc_guidance_vjp
from openpi.models_pytorch.rtc import soft_prefix_weights as rtc_soft_prefix_weights


# View ids for the hierarchical human objective. Defined here so the model does
# not import the training dataset module (and its pyarrow/PIL dependencies);
# a test asserts these stay identical to human_vlm_dataset's definitions.
VIEW_HIGH_LEVEL = 0
VIEW_LOW_LEVEL = 1


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks

    # Padded prefix tokens can occur for missing front-history slots. Leaving the
    # corresponding query row fully masked can produce NaN gradients in fused
    # attention even though those tokens are later ignored. Give every query a
    # self edge; valid tokens already have this edge, and invalid tokens still
    # remain invisible to all other tokens.
    eye = torch.eye(att_2d_masks.shape[-1], dtype=torch.bool, device=att_2d_masks.device)
    return att_2d_masks | eye[None, :, :]


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        self.enable_front_camera_tokens = getattr(config, "enable_front_camera_tokens", False)
        self.enable_front_history_images = (
            getattr(config, "enable_front_history_images", False) or self.enable_front_camera_tokens
        )
        if self.enable_front_history_images and not self.pi05:
            raise ValueError("front history images are only implemented for the pi0.5 PyTorch path")
        if self.enable_front_camera_tokens:
            self.front_camera_token_embeddings = nn.Parameter(torch.zeros(2, paligemma_config.width))
            self.front_camera_temporal_embeddings = nn.Parameter(
                torch.zeros(len(config.front_camera_history_offsets), paligemma_config.width)
            )
            self.camera_projector = nn.Linear(paligemma_config.width, config.front_camera_pose_embed_dim)
            self.camera_head = CameraHead(
                dim_in=config.front_camera_pose_embed_dim,
                trunk_depth=config.front_camera_pose_trunk_depth,
                num_heads=config.front_camera_pose_num_heads,
                mlp_ratio=config.front_camera_pose_mlp_ratio,
                is_causal=config.front_camera_pose_causal_attn,
                num_iterations=config.front_camera_pose_num_iterations,
            )
            nn.init.normal_(self.front_camera_token_embeddings, std=1e-6)
            nn.init.zeros_(self.front_camera_temporal_embeddings)
        else:
            self.front_camera_token_embeddings = None
            self.front_camera_temporal_embeddings = None
            self.camera_projector = None
            self.camera_head = None

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False
        # Skip SigLIP + sequence positions for image views whose mask is False
        # for the whole batch (output-equivalent; big win for human batches).
        self._skip_fully_masked_views = True
        # Performance experiments are opt-in until their full H100 A/B gates
        # pass. Trimming only removes trailing columns that are padding for the
        # entire routed micro-batch; the tokenizer's T512 admission limit is
        # unchanged.
        self._trim_trailing_token_padding = os.environ.get("PI05_TRIM_TRAILING_TOKEN_PADDING", "0") == "1"
        # The vision tower already checkpoints its transformer layers. This
        # switch removes the additional checkpoint around the whole encoder.
        self._checkpoint_whole_image_encoder = (
            os.environ.get("PI05_DISABLE_OUTER_IMAGE_CHECKPOINT", "0") != "1"
        )
        # These projections are small compared with the transformer stacks.
        # Checkpointing each one separately adds recomputation and dispatcher
        # overhead while saving little memory. Keep the legacy behavior by
        # default until the same-config H100 benchmark gate passes.
        self._checkpoint_lightweight_ops = (
            os.environ.get("PI05_DISABLE_LIGHTWEIGHT_CHECKPOINTS", "0") != "1"
        )

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization.

        The prefix-only path (vlm_discrete_camera) runs the stock transformers
        GemmaModel/SigLIP forward, whose decoder layers only checkpoint when
        the HF `gradient_checkpointing_enable` API has set the per-layer flag
        and `_gradient_checkpointing_func`. Setting the bare module attribute
        (the previous behavior) only disabled use_cache and checkpointed
        nothing. The fused prefix+suffix path reads the bare attributes, so
        both are set.
        """
        self.gradient_checkpointing_enabled = True
        gc_kwargs = {"use_reentrant": False}
        for module in (
            self.paligemma_with_expert.paligemma.language_model,
            self.paligemma_with_expert.paligemma.vision_tower,
            self.paligemma_with_expert.gemma_expert.model,
        ):
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
            module.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _embed_image(self, image: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_checkpoint_whole_image_encoder", True):
            return self._apply_checkpoint(self.paligemma_with_expert.embed_image, image)
        return self.paligemma_with_expert.embed_image(image)

    def _apply_lightweight_checkpoint(self, func, *args):
        if getattr(self, "_checkpoint_lightweight_ops", True):
            return self._apply_checkpoint(func, *args)
        return func(*args)

    @staticmethod
    def _trim_token_tensors(tokens, mask, *aligned):
        """Drop only columns that are trailing padding for the whole batch."""
        if tokens is None or mask is None:
            return (tokens, mask, *aligned)
        if tokens.ndim != 2 or mask.ndim != 2 or tokens.shape != mask.shape:
            raise ValueError(f"token/mask shapes must match [B,L], got {tokens.shape} and {mask.shape}")
        active_columns = mask.to(torch.bool).any(dim=0)
        if bool(active_columns.any()):
            length = int(torch.nonzero(active_columns, as_tuple=False)[-1, 0]) + 1
        else:
            length = 1
        values = (tokens, mask, *aligned)
        trimmed = []
        for value in values:
            if value is None:
                trimmed.append(None)
            elif value.ndim < 2 or value.shape[:2] != tokens.shape:
                raise ValueError(f"aligned token tensor has incompatible shape {value.shape}, expected {tokens.shape}")
            else:
                trimmed.append(value[:, :length])
        return tuple(trimmed)

    @classmethod
    def _trim_observation_prompt(cls, observation):
        prompt = cls._trim_token_tensors(
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.token_ar_mask,
            observation.token_loss_mask,
        )
        updates = {
            "tokenized_prompt": prompt[0],
            "tokenized_prompt_mask": prompt[1],
            "token_ar_mask": prompt[2],
            "token_loss_mask": prompt[3],
        }

        flow_tokens = getattr(observation, "flow_condition_tokens", None)
        flow_mask = getattr(observation, "flow_condition_mask", None)
        if flow_tokens is not None and flow_mask is not None:
            flow = cls._trim_token_tensors(
                flow_tokens,
                flow_mask,
                getattr(observation, "flow_condition_ar_mask", None),
            )
            # Observation uses one symbolic `l` axis for both token groups.
            # Keep that contract while dropping only shared trailing padding.
            length = max(prompt[0].shape[1], flow[0].shape[1])
            updates = {
                "tokenized_prompt": observation.tokenized_prompt[:, :length],
                "tokenized_prompt_mask": observation.tokenized_prompt_mask[:, :length],
                "token_ar_mask": (
                    None if observation.token_ar_mask is None else observation.token_ar_mask[:, :length]
                ),
                "token_loss_mask": (
                    None if observation.token_loss_mask is None else observation.token_loss_mask[:, :length]
                ),
                "flow_condition_tokens": flow_tokens[:, :length],
                "flow_condition_mask": flow_mask[:, :length],
                "flow_condition_ar_mask": (
                    None
                    if getattr(observation, "flow_condition_ar_mask", None) is None
                    else observation.flow_condition_ar_mask[:, :length]
                ),
            }
        return cls._observation_replace(observation, **updates)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        image_keys = _preprocessing.IMAGE_KEYS
        if self.enable_front_history_images and getattr(observation, "front_history_images", None) is not None:
            image_keys = ("left_wrist_0_rgb", "right_wrist_0_rgb")
        return _preprocessing.preprocess_observation_pytorch(observation, train=train, image_keys=image_keys)

    def _prepare_front_history_images(self, processed_observation):
        history = getattr(processed_observation, "front_history_images", None)
        if history is None:
            raise ValueError("front history input requires observation.front_history_images")
        if history.dtype == torch.uint8:
            history = history.to(torch.float32) / 255.0 * 2.0 - 1.0
        else:
            history = history.to(torch.float32)

        if history.ndim != 5:
            raise ValueError(f"front_history_images must be [B,S,H,W,C] or [B,S,C,H,W], got {history.shape}")

        bsz, n_slots = history.shape[:2]
        expected_slots = len(self.config.front_camera_history_offsets)
        if n_slots != expected_slots:
            raise ValueError(f"front_history_images has {n_slots} slots, expected {expected_slots}")
        is_channels_first = history.shape[2] == 3
        if is_channels_first:
            history_hwc = history.permute(0, 1, 3, 4, 2).reshape(bsz * n_slots, *history.shape[3:5], 3)
        else:
            history_hwc = history.reshape(bsz * n_slots, *history.shape[2:5])

        if history_hwc.shape[1:3] != _preprocessing.IMAGE_RESOLUTION:
            history_hwc = _preprocessing.image_tools.resize_with_pad_torch(
                history_hwc, *_preprocessing.IMAGE_RESOLUTION
            )
        history_chw = history_hwc.permute(0, 3, 1, 2).reshape(
            bsz, n_slots, 3, *_preprocessing.IMAGE_RESOLUTION
        )

        history_valid = getattr(processed_observation, "front_history_masks", None)
        if history_valid is None:
            raise ValueError("front history input requires observation.front_history_masks")
        history_valid = history_valid.to(device=history_chw.device, dtype=torch.bool)
        if history_valid.shape != (bsz, n_slots):
            raise ValueError(f"front_history_masks must be [B,{n_slots}], got {history_valid.shape}")
        return history_chw, history_valid

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, lang_att_masks: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self._embed_image(img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_lightweight_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        if lang_att_masks is None:
            att_masks += [0] * num_lang_embs
            lang_att_tensor = None
        else:
            lang_att_tensor = lang_att_masks.to(device=lang_masks.device, dtype=torch.bool)
            if lang_att_tensor.shape != lang_masks.shape:
                raise ValueError(f"lang_att_masks shape {lang_att_tensor.shape} must match lang_masks {lang_masks.shape}")

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        bsize = pad_masks.shape[0]
        base_att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)[None, :].expand(
            bsize, len(att_masks)
        )
        if lang_att_tensor is None:
            att_masks = base_att_masks
        else:
            att_masks = torch.cat([base_att_masks, lang_att_tensor], dim=1)

        return embs, pad_masks, att_masks

    def _embed_front_history(self, processed_observation):
        """Embed the four front-history slots once for reuse across joint-objective passes."""
        history_images, history_valid = self._prepare_front_history_images(processed_observation)
        bsz, n_slots = history_valid.shape
        flat_history = history_images.reshape(bsz * n_slots, *history_images.shape[2:])
        history_emb = self._embed_image(flat_history)
        num_front_patches = history_emb.shape[1]
        history_emb = history_emb.reshape(bsz, n_slots, num_front_patches, -1)
        return history_emb, history_valid

    def embed_prefix_with_front_camera(self, processed_observation, front_history_cache=None):
        """Embed front-history frames with interleaved camera tokens, then wrist and language context."""
        if self.front_camera_token_embeddings is None or self.camera_head is None:
            raise ValueError("front camera token modules are not initialized")

        if front_history_cache is None:
            history_emb, history_valid = self._embed_front_history(processed_observation)
        else:
            history_emb, history_valid = front_history_cache
        bsz, n_slots = history_valid.shape
        num_front_patches = history_emb.shape[2]
        if history_emb.shape[:2] != (bsz, n_slots):
            raise ValueError(
                "front_history_cache batch/slot shape does not match its validity mask: "
                f"{tuple(history_emb.shape[:2])} vs {(bsz, n_slots)}"
            )

        layout = build_front_camera_token_layout(history_valid, image_patch_count=num_front_patches)
        token_type_ids = torch.clamp(layout.type_ids, min=0)
        camera_tokens = self.front_camera_token_embeddings[token_type_ids]
        camera_tokens = torch.where(layout.valid_mask[:, :, None], camera_tokens, torch.zeros_like(camera_tokens))
        if getattr(self.config, "front_camera_use_temporal_embeddings", False):
            camera_tokens = camera_tokens + self.front_camera_temporal_embeddings[None, :n_slots]

        embs: list[torch.Tensor] = []
        for slot in range(n_slots):
            embs.append(history_emb[:, slot])
            embs.append(camera_tokens[:, slot : slot + 1])

        wrist_embs: list[torch.Tensor] = []
        wrist_patch_counts: list[int] = []
        wrist_valid_masks: list[torch.Tensor] = []
        for img, img_mask in zip(
            processed_observation.images.values(),
            processed_observation.image_masks.values(),
            strict=True,
        ):
            img_mask = img_mask.to(torch.bool)  # noqa: PLW2901
            if getattr(self, "_skip_fully_masked_views", True) and not bool(img_mask.any()):
                # Fully masked view (e.g. human batches without wrist cameras):
                # its patch tokens are pad-masked out of attention and the loss,
                # so skipping the SigLIP forward and the sequence positions is
                # output-equivalent and saves compute/memory.
                continue
            img_emb = self._embed_image(img)
            wrist_embs.append(img_emb)
            wrist_patch_counts.append(img_emb.shape[1])
            wrist_valid_masks.append(img_mask)
        embs.extend(wrist_embs)

        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_lightweight_checkpoint(lang_embed_func, processed_observation.tokenized_prompt)
        embs.append(lang_emb)

        prefix_embs = torch.cat(embs, dim=1)
        prefix_pad_masks, prefix_att_masks, layout = make_front_camera_prefix_att_masks(
            history_valid,
            front_patch_count=num_front_patches,
            wrist_patch_counts=tuple(wrist_patch_counts),
            wrist_valid_masks=tuple(wrist_valid_masks),
            lang_mask=processed_observation.tokenized_prompt_mask,
            lang_att_mask=getattr(processed_observation, "token_ar_mask", None),
        )
        return prefix_embs, prefix_pad_masks, prefix_att_masks, layout

    def embed_prefix_with_front_history(self, processed_observation, front_history_cache=None):
        """Embed front-history frames directly, without camera tokens or a camera head."""
        if front_history_cache is None:
            history_emb, history_valid = self._embed_front_history(processed_observation)
        else:
            history_emb, history_valid = front_history_cache
        bsz, n_slots = history_valid.shape
        num_front_patches = history_emb.shape[2]
        if history_emb.shape[:2] != (bsz, n_slots):
            raise ValueError(
                "front_history_cache batch/slot shape does not match its validity mask: "
                f"{tuple(history_emb.shape[:2])} vs {(bsz, n_slots)}"
            )

        embs: list[torch.Tensor] = [history_emb[:, slot] for slot in range(n_slots)]
        wrist_patch_counts: list[int] = []
        wrist_valid_masks: list[torch.Tensor] = []
        for img, img_mask in zip(
            processed_observation.images.values(),
            processed_observation.image_masks.values(),
            strict=True,
        ):
            img_mask = img_mask.to(torch.bool)  # noqa: PLW2901
            if getattr(self, "_skip_fully_masked_views", True) and not bool(img_mask.any()):
                continue
            img_emb = self._embed_image(img)
            embs.append(img_emb)
            wrist_patch_counts.append(img_emb.shape[1])
            wrist_valid_masks.append(img_mask)

        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            return lang_emb * math.sqrt(lang_emb.shape[-1])

        embs.append(self._apply_lightweight_checkpoint(lang_embed_func, processed_observation.tokenized_prompt))
        prefix_embs = torch.cat(embs, dim=1)
        prefix_pad_masks, prefix_att_masks = make_front_history_prefix_att_masks(
            history_valid,
            front_patch_count=num_front_patches,
            wrist_patch_counts=tuple(wrist_patch_counts),
            wrist_valid_masks=tuple(wrist_valid_masks),
            lang_mask=processed_observation.tokenized_prompt_mask,
            lang_att_mask=getattr(processed_observation, "token_ar_mask", None),
        )
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def _embed_prefix_for_observation(
        self,
        processed_observation,
        *,
        baseline_lang_att_mask=None,
        front_history_cache=None,
    ):
        """Select baseline, history-only, or history-plus-camera-token prefix construction."""
        if self.enable_front_camera_tokens:
            return self.embed_prefix_with_front_camera(
                processed_observation, front_history_cache=front_history_cache
            )
        if self.enable_front_history_images:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix_with_front_history(
                processed_observation, front_history_cache=front_history_cache
            )
            return prefix_embs, prefix_pad_masks, prefix_att_masks, None

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            list(processed_observation.images.values()),
            list(processed_observation.image_masks.values()),
            processed_observation.tokenized_prompt,
            processed_observation.tokenized_prompt_mask,
            baseline_lang_att_mask,
        )
        return prefix_embs, prefix_pad_masks, prefix_att_masks, None

    def embed_suffix(self, state, noisy_actions, timestep, action_time_valid_mask: torch.Tensor | None = None):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_lightweight_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_lightweight_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_lightweight_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_lightweight_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        if action_time_valid_mask is None:
            action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        else:
            action_time_mask = action_time_valid_mask.to(device=timestep.device, dtype=torch.bool)
            if action_time_mask.ndim == 3 and action_time_mask.shape[-1] == 1:
                action_time_mask = action_time_mask[..., 0]
            if action_time_mask.shape[1] < action_time_dim:
                action_time_mask = F.pad(action_time_mask, (0, action_time_dim - action_time_mask.shape[1]), value=False)
            action_time_mask = action_time_mask[:, :action_time_dim]
            if action_time_mask.shape != (bsize, action_time_dim):
                raise ValueError(
                    f"action_time_valid_mask shape {action_time_mask.shape} does not match action tokens "
                    f"{(bsize, action_time_dim)}"
                )
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (action_time_dim - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def _get_action_dim_mask(self, processed_observation, actions):
        mask = getattr(processed_observation, "action_dim_mask", None)
        if mask is None:
            return torch.ones_like(actions, dtype=torch.bool)
        mask = mask.to(device=actions.device, dtype=torch.bool)
        if mask.ndim == 2:
            mask = mask[:, None, :]
        if mask.shape[-1] < actions.shape[-1]:
            mask = F.pad(mask, (0, actions.shape[-1] - mask.shape[-1]), value=False)
        if mask.shape[1] == 1:
            mask = mask.expand(actions.shape[0], actions.shape[1], actions.shape[2])
        if mask.shape != actions.shape:
            raise ValueError(f"action_dim_mask shape {mask.shape} does not match actions {actions.shape}")
        return mask

    def _get_action_time_valid_mask(self, processed_observation, actions):
        mask = getattr(processed_observation, "action_time_valid_mask", None)
        if mask is None:
            return torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
        mask = mask.to(device=actions.device, dtype=torch.bool)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.ndim != 2:
            raise ValueError(f"action_time_valid_mask must be [B,H], got {mask.shape}")
        if mask.shape[1] < actions.shape[1]:
            mask = F.pad(mask, (0, actions.shape[1] - mask.shape[1]), value=False)
        mask = mask[:, : actions.shape[1]]
        if mask.shape != actions.shape[:2]:
            raise ValueError(f"action_time_valid_mask shape {mask.shape} does not match actions {actions.shape[:2]}")
        return mask

    def _get_action_mask(self, processed_observation, actions):
        return combine_action_dim_time_masks(
            actions,
            action_dim_mask=getattr(processed_observation, "action_dim_mask", None),
            action_time_valid_mask=getattr(processed_observation, "action_time_valid_mask", None),
        )

    def _camera_pose_targets(self, processed_observation, layout):
        extrinsics = getattr(processed_observation, "camera_extrinsics", None)
        if extrinsics is None:
            return None, None, None

        extrinsics = extrinsics.to(device=layout.valid_mask.device)
        n_slots = layout.valid_mask.shape[1]
        if extrinsics.ndim == 2 and extrinsics.shape[-1] in (n_slots * 12, n_slots * 16):
            extrinsics = extrinsics.reshape(extrinsics.shape[0], n_slots, -1)

        if extrinsics.shape[-1] in (7, 9):
            pose7 = extrinsics[..., :7].to(torch.float32)
            target_pose = torch.zeros(*pose7.shape[:-1], 9, dtype=torch.float32, device=pose7.device)
            target_pose[..., :7] = pose7
            if extrinsics.shape[-1] == 9:
                target_pose[..., 7:9] = extrinsics[..., 7:9].to(torch.float32)
                fov_valid_from_pose = torch.isfinite(target_pose[..., 7:9]).all(dim=-1)
            else:
                fov_valid_from_pose = None
        else:
            pose7 = ref_from_camera_matrix_to_pose7(extrinsics)
            target_pose = torch.zeros(*pose7.shape[:-1], 9, dtype=torch.float32, device=pose7.device)
            target_pose[..., :7] = pose7
            fov_valid_from_pose = None

        fov = getattr(processed_observation, "camera_fov", None)
        fov_valid = getattr(processed_observation, "camera_fov_valid", None)
        if fov is None:
            intrinsics = getattr(processed_observation, "camera_intrinsics", None)
            image_hw = getattr(processed_observation, "camera_image_hw", None)
            if intrinsics is not None and image_hw is not None:
                fov = intrinsics_to_fov(intrinsics.to(target_pose.device), image_hw.to(target_pose.device))
        if fov is not None:
            if fov.ndim == 2 and fov.shape[-1] == n_slots * 2:
                fov = fov.reshape(fov.shape[0], n_slots, 2)
            target_pose[..., 7:9] = fov.to(target_pose.device, dtype=torch.float32)
            if fov_valid is None:
                fov_valid = torch.isfinite(fov).all(dim=-1)

        pose_valid = getattr(processed_observation, "camera_pose_valid", None)
        if pose_valid is None:
            raise ValueError("camera_extrinsics is present but camera_pose_valid is missing")
        pose_valid = pose_valid.to(target_pose.device, dtype=torch.bool)
        pose_valid = pose_valid & layout.valid_mask.to(target_pose.device)

        if fov_valid is None:
            if fov_valid_from_pose is not None:
                fov_valid = fov_valid_from_pose.to(target_pose.device, dtype=torch.bool) & pose_valid
            else:
                fov_valid = torch.zeros_like(pose_valid)
        else:
            fov_valid = fov_valid.to(target_pose.device, dtype=torch.bool) & pose_valid
        return target_pose, pose_valid, fov_valid

    def freeze_for_vlm_discrete_camera(self) -> None:
        """Freeze the continuous Action Expert path for human VLM pretraining."""
        frozen_prefixes = (
            "paligemma_with_expert.gemma_expert",
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        )
        for name, param in self.named_parameters():
            if name.startswith(frozen_prefixes):
                param.requires_grad_(False)

    def _camera_loss_from_prefix_out(self, processed_observation, prefix_out, camera_layout):
        if not self.enable_front_camera_tokens or camera_layout is None:
            zero = prefix_out.sum() * 0.0
            return zero, {
                "camera_pose_n_valid": torch.zeros((), device=prefix_out.device),
                "camera_fov_n_valid": torch.zeros((), device=prefix_out.device),
            }
        camera_hidden = prefix_out.gather(
            1,
            camera_layout.token_indices[:, :, None].expand(-1, -1, prefix_out.shape[-1]),
        )
        camera_hidden = camera_hidden.to(dtype=self.camera_projector.weight.dtype)
        camera_pred_list = self.camera_head(
            self.camera_projector(camera_hidden),
            valid_mask=camera_layout.valid_mask,
        )
        target_pose, pose_valid, fov_valid = self._camera_pose_targets(processed_observation, camera_layout)
        if target_pose is None:
            return camera_pred_list[-1].sum() * 0.0, {
                "camera_pose_n_valid": torch.zeros((), device=prefix_out.device),
                "camera_fov_n_valid": torch.zeros((), device=prefix_out.device),
            }
        return camera_pose_loss(
            camera_pred_list,
            target_pose,
            pose_valid_mask=pose_valid,
            fov_valid_mask=fov_valid,
            trans_weight=self.config.front_camera_pose_loss_weight_trans,
            rot_weight=self.config.front_camera_pose_loss_weight_rot,
            fov_weight=self.config.front_camera_pose_loss_weight_focal,
            gamma=self.config.front_camera_pose_loss_gamma,
            normalize_trans=getattr(self.config, "front_camera_pose_loss_normalize_trans", False),
            d_bar_floor=getattr(self.config, "front_camera_pose_loss_d_bar_floor", 0.01),
        )

    def _lm_logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project PaliGemma hidden states with the tied output embedding."""
        lm_head = self.paligemma_with_expert.paligemma.lm_head
        if hidden.dtype != lm_head.weight.dtype:
            hidden = hidden.to(dtype=lm_head.weight.dtype)
        return lm_head(hidden).to(dtype=torch.float32)

    def _token_ce_loss(
        self,
        text_hidden,
        token_ids,
        token_mask,
        loss_mask,
        view_type,
        action_loss_weight=None,
    ):
        if token_ids is None or token_mask is None or loss_mask is None:
            raise ValueError("vlm_discrete_camera requires tokenized_prompt, tokenized_prompt_mask and token_loss_mask")
        if token_ids.shape[1] < 2:
            zero = text_hidden.sum() * 0.0
            return zero, {
                "loss_token_ce": zero.detach(),
                "loss_subtask_ce": zero.detach(),
                "loss_fast_ce": zero.detach(),
                "subtask_example_count": torch.zeros((), device=text_hidden.device),
                "fast_example_count": torch.zeros((), device=text_hidden.device),
                "valid_target_token_count": torch.zeros((), device=text_hidden.device),
            }

        shifted_hidden = text_hidden[:, :-1]
        targets = token_ids[:, 1:].to(device=shifted_hidden.device, dtype=torch.long)
        valid = token_mask[:, 1:].to(device=shifted_hidden.device, dtype=torch.bool) & loss_mask[:, 1:].to(
            device=shifted_hidden.device, dtype=torch.bool
        )
        bsz, target_len = targets.shape
        flat_valid = valid.reshape(-1)
        if flat_valid.any():
            hidden_valid = shifted_hidden.reshape(bsz * target_len, shifted_hidden.shape[-1])[flat_valid]
            targets_valid = targets.reshape(-1)[flat_valid]
            logits_valid = self._lm_logits_from_hidden(hidden_valid)
            ce_valid = F.cross_entropy(logits_valid, targets_valid, reduction="none")
            example_ids = (
                torch.arange(bsz, device=shifted_hidden.device)[:, None]
                .expand(bsz, target_len)
                .reshape(-1)[flat_valid]
            )
            per_example_sum = torch.zeros(bsz, device=shifted_hidden.device, dtype=ce_valid.dtype)
            per_example_sum.index_add_(0, example_ids, ce_valid)
            per_example_count = torch.zeros(bsz, device=shifted_hidden.device, dtype=ce_valid.dtype)
            per_example_count.index_add_(0, example_ids, torch.ones_like(ce_valid))
        else:
            per_example_sum = torch.zeros(bsz, device=shifted_hidden.device, dtype=torch.float32)
            per_example_count = torch.zeros(bsz, device=shifted_hidden.device, dtype=torch.float32)
        per_example_loss = per_example_sum / per_example_count.clamp_min(1.0)
        example_valid = per_example_count > 0
        if action_loss_weight is None:
            example_weight = torch.ones(bsz, device=text_hidden.device, dtype=per_example_loss.dtype)
        else:
            example_weight = action_loss_weight.to(
                device=text_hidden.device, dtype=per_example_loss.dtype
            ).reshape(-1)
            if example_weight.shape != (bsz,):
                raise ValueError(f"action_loss_weight must be [B], got {example_weight.shape}")
            if not torch.isfinite(example_weight).all() or bool((example_weight < 0).any()):
                raise ValueError("action_loss_weight must be finite and non-negative")
        valid_weight = example_weight * example_valid.to(example_weight.dtype)
        if bool(example_valid.sum() > 0):
            token_loss = (per_example_loss * valid_weight).sum() / example_valid.sum()
        else:
            token_loss = text_hidden.sum() * 0.0

        if view_type is None:
            view_type = torch.full((text_hidden.shape[0],), -1, device=text_hidden.device, dtype=torch.long)
        else:
            view_type = view_type.to(device=text_hidden.device, dtype=torch.long)
        subtask_mask = example_valid & (view_type == 0)
        fast_mask = example_valid & (view_type == 1)
        subtask_weight = example_weight * subtask_mask.to(example_weight.dtype)
        fast_weight = example_weight * fast_mask.to(example_weight.dtype)
        loss_subtask = (
            (per_example_loss * subtask_weight).sum() / subtask_weight.sum()
            if bool(subtask_weight.sum() > 0)
            else token_loss * 0.0
        )
        loss_fast = (
            (per_example_loss * fast_weight).sum() / fast_mask.sum()
            if bool(fast_mask.sum() > 0)
            else token_loss * 0.0
        )
        return token_loss, {
            "loss_token_ce": token_loss.detach(),
            "loss_subtask_ce": loss_subtask.detach(),
            "loss_fast_ce": loss_fast.detach(),
            "subtask_example_count": subtask_mask.sum().detach().to(torch.float32),
            "fast_example_count": fast_mask.sum().detach().to(torch.float32),
            "valid_target_token_count": per_example_count.sum().detach(),
            # Unreduced numerator/denominator so the trainer can form a true
            # global conditional mean instead of averaging rank-local means,
            # which silently weights a rank holding 1 example the same as a
            # rank holding 64.
            "subtask_loss_sum": per_example_loss[subtask_mask].sum().detach(),
            "fast_loss_sum": per_example_loss[fast_mask].sum().detach(),
            "subtask_weighted_loss_sum": (per_example_loss * subtask_weight).sum().detach(),
            "subtask_weight_sum": subtask_weight.sum().detach(),
            "fast_weighted_loss_sum": (per_example_loss * fast_weight).sum().detach(),
            "fast_weight_sum": fast_weight.sum().detach(),
        }

    def _run_prefix_only(self, processed_observation, front_history_cache=None):
        """PaliGemma prefix-only forward. Returns (prefix_out, camera_layout).

        The Action Expert suffix is deliberately not built, so this path can
        never execute it.
        """
        if processed_observation.token_ar_mask is None:
            raise ValueError("prefix-only forward requires token_ar_mask for prefix-LM attention")
        prefix_embs, prefix_pad_masks, prefix_att_masks, camera_layout = self._embed_prefix_for_observation(
            processed_observation,
            baseline_lang_att_mask=processed_observation.token_ar_mask,
            front_history_cache=front_history_cache,
        )
        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        position_ids = (torch.cumsum(prefix_pad_masks, dim=1) - 1).clamp_min(0)
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks).to(dtype=prefix_embs.dtype)
        (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            adarms_cond=[None, None],
        )
        if suffix_out is not None:
            raise RuntimeError("prefix-only forward must not execute the Action Expert suffix path")
        return prefix_out, camera_layout

    def _forward_vlm_discrete_camera(self, observation) -> Pi0LossOutput:
        processed_observation = self._preprocess_observation(observation, train=True)
        if processed_observation.token_ar_mask is None:
            raise ValueError("vlm_discrete_camera requires token_ar_mask for prefix-LM attention")

        prefix_embs, prefix_pad_masks, prefix_att_masks, camera_layout = self._embed_prefix_for_observation(
            processed_observation,
            baseline_lang_att_mask=processed_observation.token_ar_mask,
        )

        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = (torch.cumsum(prefix_pad_masks, dim=1) - 1).clamp_min(0)
        # The prefix-only path runs stock transformers layers whose SDPA
        # kernel requires the additive mask to match the query dtype.
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks).to(dtype=prefix_embs.dtype)

        (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            adarms_cond=[None, None],
        )
        if suffix_out is not None:
            raise RuntimeError("vlm_discrete_camera must not execute the Action Expert suffix path")

        text_len = processed_observation.tokenized_prompt.shape[1]
        text_start = prefix_out.shape[1] - text_len
        text_hidden = prefix_out[:, text_start : text_start + text_len].to(dtype=torch.float32)
        token_loss, token_metrics = self._token_ce_loss(
            text_hidden,
            processed_observation.tokenized_prompt,
            processed_observation.tokenized_prompt_mask,
            processed_observation.token_loss_mask,
            getattr(processed_observation, "vlm_view_type", None),
        )
        loss_camera, camera_metrics = self._camera_loss_from_prefix_out(processed_observation, prefix_out, camera_layout)
        total_loss = token_loss + self.config.front_camera_pose_loss_weight * loss_camera
        metrics = {
            "loss_action": token_loss.detach(),
            "loss_camera_pose": loss_camera.detach(),
            "token_truncation_count": (
                getattr(processed_observation, "vlm_token_truncated").to(token_loss.device, dtype=torch.float32).sum()
                if getattr(processed_observation, "vlm_token_truncated", None) is not None
                else torch.zeros((), device=token_loss.device)
            ).detach(),
            **token_metrics,
            **camera_metrics,
        }
        return Pi0LossOutput(loss=total_loss, loss_action=token_loss, loss_camera_pose=loss_camera, metrics=metrics)

    @staticmethod
    def _observation_replace(observation, **updates):
        """Return a copy with `updates` applied.

        Observation is a frozen dataclass, so setattr raises; the smoke
        driver's SimpleNamespace is not, so both shapes must work.
        """
        if dataclasses.is_dataclass(observation) and getattr(
            observation, "__dataclass_params__", None
        ) is not None and observation.__dataclass_params__.frozen:
            return dataclasses.replace(observation, **updates)
        new = copy.copy(observation)
        for key, value in updates.items():
            setattr(new, key, value)
        return new

    @classmethod
    def _select_observation(cls, observation, index: torch.Tensor):
        """Slice every batched field of an observation-like object by `index`."""
        updates: dict[str, Any] = {}
        for name, value in vars(observation).items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                updates[name] = value[index]
            elif isinstance(value, dict):
                updates[name] = {
                    k: (v[index] if isinstance(v, torch.Tensor) and v.dim() > 0 else v)
                    for k, v in value.items()
                }
        return cls._observation_replace(observation, **updates)

    def _flow_loss_with_camera(self, observation, actions, noise=None, time=None, front_history_cache=None):
        """Run the standard flow path and split out its flow / camera terms."""
        out = self._forward_flow_camera(
            observation,
            actions,
            noise=noise,
            time=time,
            front_history_cache=front_history_cache,
        )
        if isinstance(out, torch.Tensor):
            # Unmasked legacy path returns raw per-element MSE.
            return out.mean(), out.sum() * 0.0, {}
        return out.loss_action, out.loss_camera_pose, dict(getattr(out, "metrics", {}) or {})

    def _forward_human_joint_camera(self, observation, actions, noise=None, time=None) -> Pi0LossOutput:
        """pi0.5-style hierarchical objective over a mixed high/low-level batch.

        HIGH-LEVEL examples predict the subtask text and supervise CameraHead;
        the Action Expert is never executed for them.

        LOW-LEVEL examples run twice on purpose:
          pass 1 teacher-forces the FAST tokens for next-token CE only;
          pass 2 conditions the Action Expert on a prefix that contains NO
          ground-truth FAST tokens and supervises the continuous flow action.
        CameraHead is supervised once per example, from pass 2, so a low-level
        example does not get double camera gradient just because it runs twice.
        """
        view_type = getattr(observation, "vlm_view_type", None)
        if view_type is None:
            raise ValueError("human_joint_camera requires observation.vlm_view_type")
        view_type = view_type.reshape(-1)
        high_idx = torch.nonzero(view_type == VIEW_HIGH_LEVEL, as_tuple=False).reshape(-1)
        low_idx = torch.nonzero(view_type == VIEW_LOW_LEVEL, as_tuple=False).reshape(-1)

        device = view_type.device
        zero = None  # graph-connected zero, materialized from a real tensor below
        metrics: dict[str, torch.Tensor] = {}
        loss_subtask = None
        loss_fast = None
        loss_flow = None
        loss_camera_terms: list[torch.Tensor] = []

        # ---- high-level subset -------------------------------------------
        if high_idx.numel() > 0:
            high_obs = self._select_observation(observation, high_idx)
            if getattr(self, "_trim_trailing_token_padding", False):
                high_obs = self._trim_observation_prompt(high_obs)
            processed = self._preprocess_observation(high_obs, train=True)
            prefix_out, camera_layout = self._run_prefix_only(processed)
            text_len = processed.tokenized_prompt.shape[1]
            text_start = prefix_out.shape[1] - text_len
            text_hidden = prefix_out[:, text_start : text_start + text_len].to(dtype=torch.float32)
            loss_subtask, sub_metrics = self._token_ce_loss(
                text_hidden,
                processed.tokenized_prompt,
                processed.tokenized_prompt_mask,
                processed.token_loss_mask,
                getattr(processed, "vlm_view_type", None),
            )
            cam, cam_metrics = self._camera_loss_from_prefix_out(processed, prefix_out, camera_layout)
            loss_camera_terms.append(cam)
            zero = prefix_out.sum() * 0.0
            metrics["high_level_examples"] = torch.tensor(float(high_idx.numel()), device=device)
            metrics.update({f"high_{k}": v for k, v in sub_metrics.items()})
            metrics.update({f"high_{k}": v for k, v in cam_metrics.items()})

        # ---- low-level subset --------------------------------------------
        if low_idx.numel() > 0:
            low_obs = self._select_observation(observation, low_idx)
            if getattr(self, "_trim_trailing_token_padding", False):
                low_obs = self._trim_observation_prompt(low_obs)

            # Pass 1: teacher-forced FAST tokens, discrete CE only.
            processed_fast = self._preprocess_observation(low_obs, train=True)
            front_history_cache = None
            if self.enable_front_camera_tokens and getattr(self, "_share_front_history_embeddings", True):
                front_history_cache = self._embed_front_history(processed_fast)
            fast_prefix_out, _fast_layout = self._run_prefix_only(
                processed_fast, front_history_cache=front_history_cache
            )
            text_len = processed_fast.tokenized_prompt.shape[1]
            text_start = fast_prefix_out.shape[1] - text_len
            fast_hidden = fast_prefix_out[:, text_start : text_start + text_len].to(dtype=torch.float32)
            loss_fast, fast_metrics = self._token_ce_loss(
                fast_hidden,
                processed_fast.tokenized_prompt,
                processed_fast.tokenized_prompt_mask,
                processed_fast.token_loss_mask,
                getattr(processed_fast, "vlm_view_type", None),
                getattr(processed_fast, "action_loss_weight", None),
            )
            if zero is None:
                zero = fast_prefix_out.sum() * 0.0
            metrics["low_level_examples"] = torch.tensor(float(low_idx.numel()), device=device)
            metrics.update({f"fast_{k}": v for k, v in fast_metrics.items()})

            # Pass 2: flow. The conditioning prefix carries no GT FAST tokens.
            condition_tokens = getattr(low_obs, "flow_condition_tokens", None)
            if condition_tokens is None:
                raise ValueError("human_joint_camera low-level examples require flow_condition_tokens")
            condition_mask = low_obs.flow_condition_mask
            condition_ar_mask = low_obs.flow_condition_ar_mask
            flow_actions = getattr(low_obs, "flow_actions", None)
            if flow_actions is None:
                flow_actions = actions[low_idx] if actions is not None else None
            if flow_actions is None:
                raise ValueError("human_joint_camera low-level examples require flow_actions")
            flow_obs = self._observation_replace(
                low_obs,
                tokenized_prompt=condition_tokens,
                tokenized_prompt_mask=condition_mask,
                token_ar_mask=condition_ar_mask,
                token_loss_mask=torch.zeros_like(condition_mask),
                action_dim_mask=getattr(low_obs, "flow_action_mask", None),
                action_time_valid_mask=getattr(low_obs, "flow_time_valid_mask", None),
            )

            # Externally supplied noise/time are batched over the FULL batch,
            # so they must be sliced to the low-level subset as well.
            low_noise = noise[low_idx] if isinstance(noise, torch.Tensor) else noise
            low_time = time[low_idx] if isinstance(time, torch.Tensor) and time.dim() > 0 else time
            loss_flow, flow_camera, flow_metrics = self._flow_loss_with_camera(
                flow_obs,
                flow_actions,
                noise=low_noise,
                time=low_time,
                front_history_cache=front_history_cache,
            )
            loss_camera_terms.append(flow_camera)
            metrics.update({f"flow_{k}": v for k, v in flow_metrics.items()})

        if zero is None:
            raise ValueError("human_joint_camera received an empty batch")

        # Absent subsets contribute a graph-connected zero so DDP still sees a
        # consistent loss expression and the logged count is honestly zero.
        loss_subtask = zero if loss_subtask is None else loss_subtask
        loss_fast = zero if loss_fast is None else loss_fast
        loss_flow = zero if loss_flow is None else loss_flow
        loss_camera = (
            torch.stack(loss_camera_terms).sum() / len(loss_camera_terms) if loss_camera_terms else zero
        )

        cfg = self.config
        weighted = {
            "subtask": getattr(cfg, "joint_loss_weight_subtask", 1.0) * loss_subtask,
            "fast": getattr(cfg, "joint_loss_weight_fast", 1.0) * loss_fast,
            "flow": getattr(cfg, "joint_loss_weight_flow", 1.0) * loss_flow,
            "camera": cfg.front_camera_pose_loss_weight * loss_camera,
        }
        # Opt-in only: keeping references to live graph tensors would pin the
        # autograd graph for the whole step during normal training.
        if getattr(self, "capture_branch_losses", False):
            self.last_branch_losses = dict(weighted)
        total = weighted["subtask"] + weighted["fast"] + weighted["flow"] + weighted["camera"]
        metrics.update(
            {
                "loss_subtask": loss_subtask.detach(),
                "loss_fast": loss_fast.detach(),
                "loss_flow": loss_flow.detach(),
                "loss_camera_pose": loss_camera.detach(),
                # Explicit non-camera branch totals for validation/checkpoint
                # selection. `Pi0LossOutput.loss_action` is kept only for
                # compatibility with older logging code and should not be used
                # to decide joint-objective checkpoints.
                "loss_non_camera": (loss_subtask + loss_fast + loss_flow).detach(),
                "loss_weighted_subtask": weighted["subtask"].detach(),
                "loss_weighted_fast": weighted["fast"].detach(),
                "loss_weighted_flow": weighted["flow"].detach(),
                "loss_weighted_camera": weighted["camera"].detach(),
            }
        )
        return Pi0LossOutput(
            loss=total, loss_action=loss_fast.detach(), loss_camera_pose=loss_camera, metrics=metrics
        )

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        objective = getattr(self.config, "training_objective", "flow_camera")
        if objective == "vlm_discrete_camera":
            return self._forward_vlm_discrete_camera(observation)
        if objective == "human_joint_camera":
            return self._forward_human_joint_camera(observation, actions, noise=noise, time=time)
        return self._forward_flow_camera(observation, actions, noise=noise, time=time)

    def _forward_flow_camera(self, observation, actions, noise=None, time=None, front_history_cache=None):
        """Original OpenPI flow-matching path with the camera side branch."""

        processed_observation = self._preprocess_observation(observation, train=True)
        state = processed_observation.state

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        action_mask = self._get_action_mask(processed_observation, actions)
        action_time_valid_mask = action_mask.any(dim=-1)
        noise, _actions_for_flow, x_t, u_t = apply_action_dim_mask_for_flow(actions, noise, action_mask, time)

        prefix_embs, prefix_pad_masks, prefix_att_masks, camera_layout = self._embed_prefix_for_observation(
            processed_observation,
            front_history_cache=front_history_cache,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state,
            x_t,
            time,
            action_time_valid_mask=action_time_valid_mask,
        )
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        position_ids = (torch.cumsum(pad_masks, dim=1) - 1).clamp_min(0)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, suffix_out

        forward_args = (
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )
        if os.environ.get("PI05_DISABLE_OUTER_FLOW_CHECKPOINT", "0") == "1":
            # Gemma still checkpoints every transformer layer. This switch
            # only removes the additional checkpoint around the entire fused
            # flow pass so we can measure whether its full-model recompute is
            # worth the activation-memory saving.
            prefix_out, suffix_out = forward_func(*forward_args)
        else:
            prefix_out, suffix_out = self._apply_checkpoint(forward_func, *forward_args)

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_lightweight_checkpoint(action_out_proj_func, suffix_out)
        mse = F.mse_loss(u_t, v_t, reduction="none")
        if (
            not self.enable_front_camera_tokens
            and getattr(processed_observation, "action_dim_mask", None) is None
            and getattr(processed_observation, "action_time_valid_mask", None) is None
        ):
            return mse

        action_mask_f = action_mask.to(mse.dtype)
        flow_mse_sum = (mse * action_mask_f).sum()
        flow_element_count = action_mask_f.sum()
        action_loss_weight = getattr(processed_observation, "action_loss_weight", None)
        if action_loss_weight is None:
            weighted_action_mask = action_mask_f
        else:
            action_loss_weight = action_loss_weight.to(device=mse.device, dtype=mse.dtype).reshape(-1)
            if action_loss_weight.shape != (mse.shape[0],):
                raise ValueError(f"action_loss_weight must be [B], got {action_loss_weight.shape}")
            if not torch.isfinite(action_loss_weight).all() or bool((action_loss_weight < 0).any()):
                raise ValueError("action_loss_weight must be finite and non-negative")
            weighted_action_mask = action_mask_f * action_loss_weight[:, None, None]
        flow_weighted_mse_sum = (mse * weighted_action_mask).sum()
        flow_weighted_element_count = weighted_action_mask.sum()
        # Keep the denominator unweighted: action_loss_weight is a real loss
        # coefficient, not merely a rebalancing distribution. A pure-human
        # batch with weight 0.2 must therefore produce 0.2x the action loss.
        loss_action = flow_weighted_mse_sum / flow_element_count.clamp_min(1.0)

        # One shared implementation with the prefix-only path: duplicating the
        # camera head/loss in two places is how the two silently drift apart.
        loss_camera, camera_metrics = self._camera_loss_from_prefix_out(
            processed_observation, prefix_out, camera_layout
        )

        total_loss = loss_action + self.config.front_camera_pose_loss_weight * loss_camera
        metrics = {
            "loss_action": loss_action.detach(),
            "loss_camera_pose": loss_camera.detach(),
            "action_dim_n_valid": action_mask_f.sum().detach(),
            "flow_mse_sum": flow_mse_sum.detach(),
            "flow_element_count": flow_element_count.detach(),
            "flow_weighted_mse_sum": flow_weighted_mse_sum.detach(),
            "flow_weighted_element_count": flow_weighted_element_count.detach(),
            **camera_metrics,
        }
        return Pi0LossOutput(loss=total_loss, loss_action=loss_action, loss_camera_pose=loss_camera, metrics=metrics)

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        processed_observation = self._preprocess_observation(observation, train=False)
        state = processed_observation.state
        action_mask = self._get_action_mask(processed_observation, noise)
        action_time_valid_mask = action_mask.any(dim=-1)

        prefix_embs, prefix_pad_masks, prefix_att_masks, _ = self._embed_prefix_for_observation(
            processed_observation
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = (torch.cumsum(prefix_pad_masks, dim=1) - 1).clamp_min(0)

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = torch.where(action_mask, noise, torch.zeros_like(noise))
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                action_time_valid_mask=action_time_valid_mask,
            )
            # Euler step - use new tensor assignment instead of in-place operation
            x_t = apply_action_mask_for_denoise_update(x_t, v_t, dt, action_mask)
            time += dt
        return torch.where(action_mask, x_t, torch.zeros_like(x_t))

    def sample_actions_rtc(
        self,
        device,
        observation,
        previous_actions,
        inference_delay_steps,
        execution_horizon_steps,
        noise=None,
        num_steps=5,
        max_guidance_weight=5.0,
    ) -> Tensor:
        """Sample an RTC-guided chunk without changing the trained policy.

        ``previous_actions`` starts at the observation time: actions already
        executed before this inference began have been removed. It is padded on
        the right to the model horizon, while the RTC mask keeps that fresh tail
        unconstrained.
        """
        bsize = observation.state.shape[0]
        horizon = self.config.action_horizon
        action_dim = self.config.action_dim
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if previous_actions.ndim != 3 or previous_actions.shape[0] != bsize:
            raise ValueError(
                f"previous_actions must be [batch, remaining_horizon, action_dim], got {tuple(previous_actions.shape)}"
            )
        if previous_actions.shape[1] > horizon or previous_actions.shape[2] > action_dim:
            raise ValueError(
                "previous_actions exceeds model action shape: "
                f"{tuple(previous_actions.shape)} vs (*, {horizon}, {action_dim})"
            )
        if not 0 <= execution_horizon_steps <= horizon:
            raise ValueError(f"execution_horizon_steps must be in [0, {horizon}], got {execution_horizon_steps}")
        expected_remaining = horizon - execution_horizon_steps
        if previous_actions.shape[1] != expected_remaining:
            raise ValueError(
                "previous_actions length must equal horizon - execution_horizon_steps: "
                f"{previous_actions.shape[1]} != {horizon} - {execution_horizon_steps}"
            )

        if noise is None:
            noise = self.sample_noise((bsize, horizon, action_dim), device)
        previous_actions = previous_actions.to(device=device, dtype=noise.dtype)
        if previous_actions.shape[2] < action_dim:
            previous_actions = F.pad(
                previous_actions,
                (0, action_dim - previous_actions.shape[2]),
            )
        previous_actions = F.pad(
            previous_actions,
            (0, 0, 0, execution_horizon_steps),
        )

        with torch.no_grad():
            processed_observation = self._preprocess_observation(observation, train=False)
            state = processed_observation.state
            action_mask = self._get_action_mask(processed_observation, noise)
            action_time_valid_mask = action_mask.any(dim=-1)

            prefix_embs, prefix_pad_masks, prefix_att_masks, _ = self._embed_prefix_for_observation(
                processed_observation
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = (torch.cumsum(prefix_pad_masks, dim=1) - 1).clamp_min(0)
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
            self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

            x_t = torch.where(action_mask, noise, torch.zeros_like(noise))

        time_weights = rtc_soft_prefix_weights(
            horizon,
            int(inference_delay_steps),
            int(execution_horizon_steps),
            device=device,
            dtype=torch.float32,
        )
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        for denoise_index in range(num_steps):
            time = torch.tensor(
                1.0 - denoise_index / num_steps,
                dtype=torch.float32,
                device=device,
            )
            with torch.enable_grad():
                x_for_grad = x_t.detach().requires_grad_(True)
                expanded_time = time.expand(bsize)
                v_t = self.denoise_step(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    x_for_grad,
                    expanded_time,
                    action_time_valid_mask=action_time_valid_mask,
                )
                # This repository integrates from time=1 (noise) to time=0
                # (action), so x - time*v is the clean-action estimate.
                clean_estimate = x_for_grad - time * v_t
                guidance = rtc_guidance_vjp(
                    clean_estimate,
                    x_for_grad,
                    previous_actions,
                    time_weights,
                    action_mask,
                )
                guided_v = (
                    v_t
                    - rtc_guidance_scale(
                        time,
                        max_guidance_weight,
                    ).to(v_t.dtype)
                    * guidance
                )
            with torch.no_grad():
                x_t = apply_action_mask_for_denoise_update(
                    x_for_grad.detach(),
                    guided_v.detach(),
                    dt,
                    action_mask,
                )

        return torch.where(action_mask, x_t, torch.zeros_like(x_t)).detach()

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        action_time_valid_mask: torch.Tensor | None = None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state,
            x_t,
            timestep,
            action_time_valid_mask=action_time_valid_mask,
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = (prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1).clamp_min(0)

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
