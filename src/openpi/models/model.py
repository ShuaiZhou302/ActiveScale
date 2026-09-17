import abc
from collections.abc import Sequence
import dataclasses
import enum
import gc
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors.torch
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.models_pytorch.camera_head import load_state_dict_with_camera_whitelist
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#     "vlm_view_type": int32[*b],  # Optional, 0=subtask view, 1=FAST action view
#     "vlm_token_truncated": bool[*b],  # Optional, token sequence was truncated
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]
    # Positive-valid mask for the canonical [left, right, camera] pose blocks.
    # Human pack loaders zero invalid blocks before tokenization/model input;
    # this field preserves the source validity semantics for audits and logs.
    state_pose_valid: at.Bool[ArrayT, "*b sp"] | None = None

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    # Human VLM discrete pretraining view type: 0=subtask, 1=FAST action.
    vlm_view_type: at.Int[ArrayT, "*b"] | None = None
    # Stable per-sample source id so the trainer can aggregate per-source
    # metrics across DataLoader workers and DDP ranks.
    source_id: at.Int[ArrayT, "*b"] | None = None
    # Per-example multiplier for action-only objectives. This lets mixed
    # human/robot batches reduce human FAST+flow supervision without changing
    # subtask or CameraHead losses.
    action_loss_weight: at.Float[ArrayT, "*b"] | None = None
    vlm_token_truncated: at.Bool[ArrayT, "*b"] | None = None

    # APVLA front-camera-token fields. These are optional and ignored by baseline pi0/pi0.5.
    # Distinct dim symbols on purpose: `images` is stored channels-first for
    # the torch path but annotated "*b h w c", so reusing h/w/c here would
    # make jaxtyping bind conflicting sizes for channels-last history frames.
    front_history_images: at.Float[ArrayT, "*b n fh fw fc"] | None = None
    front_history_masks: at.Bool[ArrayT, "*b n"] | None = None
    # Legacy name retained deliberately: this stores T_ref_from_camera for camera-token training.
    # This is the same pose direction used by the camera component of the action chunk.
    camera_extrinsics: at.Float[ArrayT, "*b n 4 4"] | None = None
    camera_pose_valid: at.Bool[ArrayT, "*b n"] | None = None
    camera_intrinsics: at.Float[ArrayT, "*b n 3 3"] | None = None
    camera_image_hw: at.Float[ArrayT, "*b n 2"] | None = None
    camera_fov: at.Float[ArrayT, "*b n 2"] | None = None
    camera_fov_valid: at.Bool[ArrayT, "*b n"] | None = None
    # Positive-valid action mask. Shape may be [B, D] or [B, H, D]; the human
    # VLM packs emit the per-timestep [B, H, D] form, so both are accepted.
    action_dim_mask: at.Bool[ArrayT, "*b d"] | at.Bool[ArrayT, "*b ah d"] | None = None
    # Positive-valid temporal mask for action chunks. Data converters should pass
    # `~action_is_pad` here when suffix frames are padded at segment/episode end.
    action_time_valid_mask: at.Bool[ArrayT, "*b ah"] | None = None
    # human_joint_camera low-level fields. flow_condition_* is the prefix the
    # Action Expert conditions on and deliberately carries no GT FAST tokens.
    flow_condition_tokens: at.Int[ArrayT, "*b l"] | None = None
    flow_condition_mask: at.Bool[ArrayT, "*b l"] | None = None
    flow_condition_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Distinct symbols on purpose: `fh` is already bound to the history frame
    # HEIGHT by front_history_images, so reusing it for the flow horizon would
    # force 10 == 224 and reject every real batch.
    flow_actions: at.Float[ArrayT, "*b fah fad"] | None = None
    flow_action_mask: at.Bool[ArrayT, "*b fah fad"] | None = None
    flow_time_valid_mask: at.Bool[ArrayT, "*b fah"] | None = None
    has_flow_target: at.Bool[ArrayT, "*b"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        # Front history frames arrive as uint8 [B, S, H, W, C] from the offline
        # human VLM packs. Scale them the same way as the wrist images but keep
        # the channels-last layout that the camera-token path expects.
        front_history = data.get("front_history_images")
        if front_history is not None and getattr(front_history, "dtype", None) in (np.uint8, torch.uint8):
            if isinstance(front_history, np.ndarray):
                data["front_history_images"] = front_history.astype(np.float32) / 255.0 * 2.0 - 1.0
            else:
                data["front_history_images"] = front_history.to(torch.float32) / 255.0 * 2.0 - 1.0
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            state_pose_valid=data.get("state_pose_valid"),
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            vlm_view_type=data.get("vlm_view_type"),
            source_id=data.get("source_id"),
            action_loss_weight=data.get("action_loss_weight"),
            vlm_token_truncated=data.get("vlm_token_truncated"),
            front_history_images=data.get("front_history_images"),
            front_history_masks=data.get("front_history_masks"),
            camera_extrinsics=data.get("camera_extrinsics"),
            camera_pose_valid=data.get("camera_pose_valid"),
            camera_intrinsics=data.get("camera_intrinsics"),
            camera_image_hw=data.get("camera_image_hw"),
            camera_fov=data.get("camera_fov"),
            camera_fov_valid=data.get("camera_fov_valid"),
            action_dim_mask=data.get("action_dim_mask"),
            action_time_valid_mask=data.get("action_time_valid_mask"),
            flow_condition_tokens=data.get("flow_condition_tokens"),
            flow_condition_mask=data.get("flow_condition_mask"),
            flow_condition_ar_mask=data.get("flow_condition_ar_mask"),
            flow_actions=data.get("flow_actions"),
            flow_action_mask=data.get("flow_action_mask"),
            flow_time_valid_mask=data.get("flow_time_valid_mask"),
            has_flow_target=data.get("has_flow_target"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        vlm_view_type=observation.vlm_view_type,
        vlm_token_truncated=observation.vlm_token_truncated,
        front_history_images=observation.front_history_images,
        front_history_masks=observation.front_history_masks,
        camera_extrinsics=observation.camera_extrinsics,
        camera_pose_valid=observation.camera_pose_valid,
        camera_intrinsics=observation.camera_intrinsics,
        camera_image_hw=observation.camera_image_hw,
        camera_fov=observation.camera_fov,
        camera_fov_valid=observation.camera_fov_valid,
        action_dim_mask=observation.action_dim_mask,
        action_time_valid_mask=observation.action_time_valid_mask,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        state_dict = safetensors.torch.load_file(weight_path, device="cpu")
        report = load_state_dict_with_camera_whitelist(
            model,
            state_dict,
            allow_camera_missing=getattr(train_config.model, "enable_front_camera_tokens", False),
        )
        del state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Loaded PyTorch weights with report: %s", report)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    if not hasattr(jax.monitoring, "record_scalar"):
        # Orbax versions in some cluster envs still call the older JAX monitoring hook.
        jax.monitoring.record_scalar = lambda *_, **__: None  # type: ignore[attr-defined]

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        if isinstance(metadata, dict):
            params_metadata = metadata["params"]
        else:
            metadata_tree = getattr(metadata, "tree", None)
            if metadata_tree is None:
                item_metadata = getattr(metadata, "item_metadata", None)
                metadata_tree = getattr(item_metadata, "tree", None)
            if metadata_tree is None or "params" not in metadata_tree:
                raise ValueError(f"Unexpected checkpoint metadata format at {params_path}: {type(metadata)!r}")
            params_metadata = metadata_tree["params"]
        item = {"params": params_metadata}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
