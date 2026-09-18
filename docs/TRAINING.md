# Training recipe

## Mid-training

| Parameter | Value |
| --- | --- |
| Initialization | official pi0.5 base, converted to PyTorch |
| Human/robot sampling | 1:1 |
| Action horizon | 50 |
| History slots | 4, offsets `[-48, -32, -16, 0]` |
| Global batch | 256 |
| Precision | bf16 |
| Optimizer | AdamW |
| Adam | beta1 0.9, beta2 0.95, eps 1e-8 |
| Weight decay | 1e-10 |
| Gradient clipping | 1.0 |
| Gradient norm implementation | scalar (`PI05_GRAD_CLIP_FOREACH=0`) |
| AdamW implementation | scalar (`PI05_ADAMW_FOREACH=0`) |
| Learning rate | constant 5e-5 after warmup |
| Warmup | 1,000 optimizer steps |
| EMA | disabled |

Convert an official openpi pi0.5 JAX checkpoint before mid-training:

```bash
python scripts/convert_jax_model_to_pytorch.py \
  --checkpoint-dir /path/to/pi05-base \
  --config-name pi05_aloha \
  --output-path /path/to/pi05-base-pytorch
```

The low-level view jointly supervises FAST token cross-entropy and flow
matching. The high-level view supervises subtask token cross-entropy.
CameraHead receives valid human and Piper camera targets. Public robot datasets
without an admitted head pose mask that target.

Outer objective weights are:

| Term | Weight |
| --- | --- |
| subtask CE | 1.0 |
| FAST CE | 1.0 |
| flow matching | 1.0 |
| CameraHead | 0.2 |
| human action multiplier | 0.2 |
| robot action multiplier | 1.0 |

Each term is normalized by its own valid examples/tokens/dimensions before the
outer weight is applied.

## Post-training

Post-training starts from the mid-trained checkpoint and uses the same model
architecture and H50 action space. Task-specific Piper data supervise flow
matching and CameraHead wherever valid. Choose the run length and checkpoint
interval for the size and diversity of the target dataset. The launcher uses
the selected run length as the cosine-decay horizon, from a peak learning rate
of 5e-5 to 5e-6 after a 1,000-step warmup.

## Attention

The public release uses the original Hugging Face eager attention path for
both prefix-only and fused prefix/action-expert training.

The experimental `beta` branch also provides an exact packed block-causal
backend. Enable it only after its local smoke test passes:

```bash
ACTIVESCALE_ATTENTION_BACKEND=packed_flash \
  bash scripts/train_activescale.sh midtrain <steps> [save_interval]
```

The release runner also sets `PI05_GRAD_CLIP_FOREACH=0` and
`PI05_ADAMW_FOREACH=0`. These retain the same global-norm clipping and AdamW
rules while avoiding PyTorch multi-tensor foreach kernel stalls observed with
rank-dependent mixed-objective gradient sets. `auto` restores PyTorch's
automatic implementation choice; it is not recommended until the target
PyTorch/CUDA stack passes a multi-rank optimizer smoke test.

## Reproducing a run

1. Fill `configs/activescale.env`.
2. Run `bash scripts/train_activescale.sh smoke`.
3. Inspect source-wise losses, valid-mask counts, and one decoded batch.
4. Run `bash scripts/train_activescale.sh midtrain <steps> [save_interval]`.
5. Archive the resolved config, Git commit, norm stats, selection manifests,
   and dataset fingerprints with the checkpoint.
