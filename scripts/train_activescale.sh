#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-smoke}
TRAIN_STEPS=${2:-${TRAIN_STEPS:-}}
SAVE_INTERVAL=${3:-${SAVE_INTERVAL:-}}
EXTRA_ARGS=()
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ENV_FILE=${ACTIVESCALE_ENV_FILE:-"$ROOT/configs/activescale.env"}

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
EXP_NAME=${EXP_NAME:-"activescale_${MODE}_$(date +%Y%m%d_%H%M%S)"}

case "$MODE" in
  smoke)
    CONFIG=activescale_pi05_human_robot_h50_smoke
    TRAIN_STEPS=${TRAIN_STEPS:-10}
    SAVE_INTERVAL=${SAVE_INTERVAL:-$TRAIN_STEPS}
    ;;
  midtrain)
    CONFIG=activescale_pi05_human_robot_h50
    : "${TRAIN_STEPS:?usage: $0 midtrain <steps> [save_interval]}"
    SAVE_INTERVAL=${SAVE_INTERVAL:-$TRAIN_STEPS}
    ;;
  posttrain)
    CONFIG=activescale_pi05_piper_posttrain_h50
    : "${TRAIN_STEPS:?usage: $0 posttrain <steps> [save_interval]}"
    SAVE_INTERVAL=${SAVE_INTERVAL:-$TRAIN_STEPS}
    EXTRA_ARGS+=(--lr-schedule.decay-steps "$TRAIN_STEPS")
    ;;
  *)
    echo "usage: $0 {smoke|midtrain|posttrain} [steps] [save_interval]" >&2
    exit 2
    ;;
esac

if ! [[ "$TRAIN_STEPS" =~ ^[1-9][0-9]*$ && "$SAVE_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "steps and save_interval must be positive integers" >&2
  exit 2
fi

: "${ACTIVESCALE_OUTPUT_ROOT:?Set ACTIVESCALE_OUTPUT_ROOT or source configs/activescale.env}"
: "${ACTIVESCALE_DATA_ROOT:?Set ACTIVESCALE_DATA_ROOT}"
: "${ACTIVESCALE_CACHE_ROOT:?Set ACTIVESCALE_CACHE_ROOT}"
: "${ACTIVESCALE_ASSETS_ROOT:?Set ACTIVESCALE_ASSETS_ROOT}"
if [[ "$MODE" == posttrain ]]; then
  : "${ACTIVESCALE_MIDTRAIN_CHECKPOINT:?Set ACTIVESCALE_MIDTRAIN_CHECKPOINT}"
else
  : "${ACTIVESCALE_BASE_CHECKPOINT:?Set ACTIVESCALE_BASE_CHECKPOINT}"
fi

case "${ACTIVESCALE_ATTENTION_BACKEND:-reference}" in
  reference)
    export PI05_USE_BLOCKWISE_VARLEN_FLASH=0
    export PI05_USE_PREFIX_BLOCKWISE_VARLEN_FLASH=0
    ;;
  packed_flash)
    export PI05_USE_BLOCKWISE_VARLEN_FLASH=1
    export PI05_USE_PREFIX_BLOCKWISE_VARLEN_FLASH=1
    ;;
  *)
    echo "ACTIVESCALE_ATTENTION_BACKEND must be reference or packed_flash" >&2
    exit 2
    ;;
esac
export PI05_DISABLE_OUTER_FLOW_CHECKPOINT=${PI05_DISABLE_OUTER_FLOW_CHECKPOINT:-0}
# PyTorch's foreach norm kernel can stall on rank-dependent mixed-objective
# gradient sets. The scalar reduction computes the same global norm reliably.
export PI05_GRAD_CLIP_FOREACH=${PI05_GRAD_CLIP_FOREACH:-0}
export PI05_ADAMW_FOREACH=${PI05_ADAMW_FOREACH:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$ACTIVESCALE_OUTPUT_ROOT"
cd "$ROOT"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  scripts/train_pytorch.py "$CONFIG" \
  --exp-name "$EXP_NAME" \
  --checkpoint-base-dir "$ACTIVESCALE_OUTPUT_ROOT" \
  --num-train-steps "$TRAIN_STEPS" \
  --save-interval "$SAVE_INTERVAL" \
  "${EXTRA_ARGS[@]}"
