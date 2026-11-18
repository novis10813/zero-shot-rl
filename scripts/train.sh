#!/usr/bin/env bash
# Dedicated FB pre-training launcher (wraps zero_shot_rl.train.main_exorl).
#
# Examples:
#   bash scripts/train.sh --seed 0 --dataset "$HOME/zsrl-datasets/walker/rnd/dataset.npz"
#   SEED=0 ALGO=vcfb bash scripts/train.sh --dataset "$HOME/zsrl-datasets/walker/rnd/dataset.npz"
#   bash scripts/train.sh --algo fb --domain walker --exploration rnd --steps 1000000 --seed 1
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

ALGO=${ALGO:-fb}
DOMAIN=${DOMAIN:-walker}
EXPLORATION=${EXPLORATION:-rnd}
DATASET_PATH=${DATASET_PATH:-}
EVAL_TASKS=${EVAL_TASKS:-"stand walk run flip"}
SEED=${SEED:-42}
LEARNING_STEPS=${LEARNING_STEPS:-1000000}
DATASET_TRANSITIONS=${DATASET_TRANSITIONS:-100000}
DISCOUNT=${DISCOUNT:-0.98}
Z_DIM=${Z_DIM:-50}
Z_INFERENCE_STEPS=${Z_INFERENCE_STEPS:-10000}
WANDB_LOGGING=${WANDB_LOGGING:-False}
BF16=${BF16:-False}
RUN_NAME=${RUN_NAME:-}
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash scripts/train.sh [options] [-- <extra main_exorl args>]

Dedicated offline RL / FB training launcher (zero_shot_rl.train.main_exorl).
For auxiliary models (ensemble, g0, context_encoder, dynamics_model, cvae_model),
use: PYTHONPATH=. uv run python -m zero_shot_rl.train <subcommand> [args...]

Options (env var equivalents in parentheses):
  --algo NAME              Algorithm: fb, vcfb, mcfb, cql, td3, sf-lap, rebrac, gciql (ALGO=fb)
  --domain NAME            ExORL domain: walker, quadruped, point_mass_maze, jaco (DOMAIN=walker)
  --exploration NAME       Dataset exploration algo: rnd, diayn, random (EXPLORATION=rnd)
  --dataset PATH           Path to dataset.npz (DATASET_PATH)
  --eval-tasks "T1 T2..."  Space-separated eval tasks (EVAL_TASKS="stand walk run flip")
  --seed INT               Random seed (SEED=42)
  --steps INT              Learning steps (LEARNING_STEPS=1000000)
  --transitions INT        Dataset transitions subsample (DATASET_TRANSITIONS=100000)
  --discount FLOAT         Discount factor (DISCOUNT=0.98)
  --z-dim INT              Latent z dimension (Z_DIM=50)
  --z-inference-steps INT  Transitions used to infer z (Z_INFERENCE_STEPS=10000)
  --wandb True|False       Enable W&B logging (WANDB_LOGGING=False)
  --bf16 True|False        Enable bf16 training (BF16=False)
  --run-name NAME          Optional run name (RUN_NAME)
  -h, --help               Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --algo) ALGO=$2; shift 2 ;;
    --domain) DOMAIN=$2; shift 2 ;;
    --exploration) EXPLORATION=$2; shift 2 ;;
    --dataset|--dataset_path) DATASET_PATH=$2; shift 2 ;;
    --eval-tasks|--eval_tasks) EVAL_TASKS=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --steps|--learning_steps) LEARNING_STEPS=$2; shift 2 ;;
    --transitions|--dataset_transitions) DATASET_TRANSITIONS=$2; shift 2 ;;
    --discount) DISCOUNT=$2; shift 2 ;;
    --z-dim|--z_dimension) Z_DIM=$2; shift 2 ;;
    --z-inference-steps|--z_inference_steps) Z_INFERENCE_STEPS=$2; shift 2 ;;
    --wandb|--wandb_logging) WANDB_LOGGING=$2; shift 2 ;;
    --bf16) BF16=$2; shift 2 ;;
    --run-name|--run_name) RUN_NAME=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

read -r -a TASKS_ARR <<< "$EVAL_TASKS"

CMD=(
  uv run python -m zero_shot_rl.train.main_exorl
  "$ALGO" "$DOMAIN" "$EXPLORATION"
  --eval_tasks "${TASKS_ARR[@]}"
  --seed "$SEED"
  --learning_steps "$LEARNING_STEPS"
  --dataset_transitions "$DATASET_TRANSITIONS"
  --discount "$DISCOUNT"
  --z_dimension "$Z_DIM"
  --z_inference_steps "$Z_INFERENCE_STEPS"
  --wandb_logging "$WANDB_LOGGING"
  --bf16 "$BF16"
)

if [[ -n "$DATASET_PATH" ]]; then
  CMD+=(--dataset_path "$DATASET_PATH")
fi
if [[ -n "$RUN_NAME" ]]; then
  CMD+=(--run_name "$RUN_NAME")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[TRAIN] algo=$ALGO domain=$DOMAIN exploration=$EXPLORATION seed=$SEED steps=$LEARNING_STEPS transitions=$DATASET_TRANSITIONS discount=$DISCOUNT z_dim=$Z_DIM wandb=$WANDB_LOGGING"
echo "[TRAIN] cmd: ${CMD[*]}"

export PYTHONPATH=. PYTHONUNBUFFERED=1 MUJOCO_GL=${MUJOCO_GL:-egl}
exec "${CMD[@]}"
