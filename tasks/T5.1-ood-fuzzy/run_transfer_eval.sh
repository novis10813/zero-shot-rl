#!/usr/bin/env bash
# Evaluate clean-trained ANFIS-z models zero-shot across the 4 Walker OOD dynamics.
#
# Usage (repo root):
#   CLEAN_MODELS_DIR=<path/to/clean_ft1000/models> OUT_ROOT=<out/transfer_ft1000> \
#     [CHECKPOINT=/home/sam/zsrl-artifacts/T4.1.3.1-residual-soft-q/zero-shot-rl-297/660000.pt] \
#     [TAG=final] bash tasks/T5.1-ood-fuzzy/run_transfer_eval.sh
set -euo pipefail

CLEAN_MODELS_DIR=${CLEAN_MODELS_DIR:?set CLEAN_MODELS_DIR to clean-trained models/ directory}
OUT_ROOT=${OUT_ROOT:?set OUT_ROOT to output root directory}
CHECKPOINT=${CHECKPOINT:-/home/sam/zsrl-artifacts/T4.1.3.1-residual-soft-q/zero-shot-rl-297/660000.pt}
TAG=${TAG:-final}

declare -A DYNAMICS=(
  ["friction_mild"]="tasks/T3.2-ood-robustness/configs/friction_mild.yaml"
  ["friction_high_mild"]="tasks/T4.1.2-dynamic-shift-detection/configs/friction_high_mild.yaml"
  ["actuator_mild"]="tasks/T3.2-ood-robustness/configs/actuator_mild.yaml"
  ["actuator_high_mild"]="tasks/T4.1.2-dynamic-shift-detection/configs/actuator_high_mild.yaml"
)

for name in friction_mild friction_high_mild actuator_mild actuator_high_mild; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. MUJOCO_GL=egl \
    uv run python -m zero_shot_rl.finetune.anfis_z \
      --anfis_config configs/anfis/walker.yaml \
      --config tasks/T5.1-ood-fuzzy/configs/reinforce.yaml \
      --checkpoint "$CHECKPOINT" \
      --dynamics "${DYNAMICS[$name]}" \
      --eval_models_dir "$CLEAN_MODELS_DIR" \
      --eval_model_tag "$TAG" \
      --out_dir "$OUT_ROOT/$name"
done
