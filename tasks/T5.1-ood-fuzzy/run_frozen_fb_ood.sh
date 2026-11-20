#!/usr/bin/env bash
# Evaluate frozen FB (z_r) on clean Walker + 4 OOD dynamics using eval seeds 5..24.
set -euo pipefail

CHECKPOINT=${CHECKPOINT:-/home/sam/zsrl-artifacts/T4.1.3.1-residual-soft-q/zero-shot-rl-297/660000.pt}
DATASET=${DATASET:-/home/sam/zsrl-datasets/walker/rnd/dataset.npz}
OUT_ROOT=${OUT_ROOT:-out/frozen_fb}

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. MUJOCO_GL=egl \
  uv run python -m zero_shot_rl.finetune.anfis_z \
    --anfis_config configs/anfis/walker.yaml \
    --config tasks/T5.1-ood-fuzzy/configs/reinforce.yaml \
    --checkpoint "$CHECKPOINT" \
    --dataset_path "$DATASET" \
    --eval_zr_only \
    --out_dir "$OUT_ROOT/clean"

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
      --dataset_path "$DATASET" \
      --dynamics "${DYNAMICS[$name]}" \
      --eval_zr_only \
      --out_dir "$OUT_ROOT/$name"
done
