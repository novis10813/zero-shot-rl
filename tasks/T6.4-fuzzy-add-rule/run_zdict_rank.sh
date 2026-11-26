#!/usr/bin/env bash
# Rank analysis of additive-rule ANFIS Z_dict across 4 / 8 / 37 sensors and zero_order / first_order.
#
# Usage (repo root):
#   CHECKPOINT=<fb 660000.pt> DATASET=<walker/rnd/dataset.npz> OUT_DIR=<out dir> \
#     bash tasks/T6.4-fuzzy-add-rule/run_zdict_rank.sh [extra args...]
set -euo pipefail

CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to the FB checkpoint}
DATASET=${DATASET:?set DATASET to walker/rnd/dataset.npz}
OUT_DIR=${OUT_DIR:?set OUT_DIR}

OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2} \
PYTHONPATH=. MUJOCO_GL=${MUJOCO_GL:-egl} \
  exec uv run python -m analysis.zdict_rank \
  --checkpoint "$CHECKPOINT" \
  --dataset_path "$DATASET" \
  --anfis_configs \
    tasks/T6.4-fuzzy-add-rule/configs/walker4_additive_zero.yaml \
    tasks/T6.4-fuzzy-add-rule/configs/walker4_additive_first.yaml \
    tasks/T6.4-fuzzy-add-rule/configs/walker8_additive_zero.yaml \
    tasks/T6.4-fuzzy-add-rule/configs/walker8_additive_first.yaml \
    tasks/T6.4-fuzzy-add-rule/configs/walker_all37_additive_zero.yaml \
    tasks/T6.4-fuzzy-add-rule/configs/walker_all37_additive_first.yaml \
  --out_dir "$OUT_DIR" \
  "$@"
