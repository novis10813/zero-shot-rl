#!/usr/bin/env bash
# Rank analysis of the ANFIS rule dictionary: fuzzy 4 / 8 sensors and linear 4 / 8 / 37 sensors.
#
# Usage (repo root):
#   CHECKPOINT=<fb 660000.pt> DATASET=<walker/rnd/dataset.npz> OUT_DIR=<out dir> \
#     bash tasks/T6.3-sensors/run_zdict_rank.sh [extra args...]
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
    configs/anfis/walker.yaml \
    tasks/T6.1-ood-fuzzy/configs/walker_actuator.yaml \
    tasks/T6.3-sensors/configs/walker4_linear.yaml \
    tasks/T6.3-sensors/configs/walker8_linear.yaml \
    tasks/T6.3-sensors/configs/walker_all37_linear.yaml \
  --out_dir "$OUT_DIR" \
  "$@"
