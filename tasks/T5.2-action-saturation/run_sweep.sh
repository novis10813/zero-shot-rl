#!/bin/bash
# Return of standard FB at z_r versus actuator gear (configs/gear_*.yaml, 100% to 30% of the XML
# gear on all six joints). The z_r of each task is inferred from the checkpoint first
# (zero_shot_rl.latent.infer --z_r_only), then zero_shot_rl.eval.dynamics_sweep rolls out
# EPISODES deterministic 1000-step episodes (environment seeds 0..EPISODES-1, the same in every
# config) per task and config. Writes out/zs.npz and out/sweep_<task>.csv.
#
# Usage (repo root):
#     CKPT=<fb.pt> DATASET=<walker rnd dataset.npz> bash tasks/T5.2-action-saturation/run_sweep.sh
set -euo pipefail

CKPT=${CKPT:?}
DATASET=${DATASET:?}
EPISODES=${EPISODES:-20}
WORKERS=${WORKERS:-12}
TASKS=${TASKS:-"walk run stand flip"}
T=tasks/T5.2-action-saturation

export PYTHONPATH=. MUJOCO_GL=egl OMP_NUM_THREADS=1
mkdir -p out
sha256sum "$CKPT"
uv run python -m zero_shot_rl.latent.infer --checkpoint "$CKPT" --dataset_path "$DATASET" \
  --z_r_only --tasks $TASKS --workers "$WORKERS" --out out/zs.npz
for TASK in $TASKS; do
  uv run python -m zero_shot_rl.eval.dynamics_sweep --checkpoint "$CKPT" --zs out/zs.npz \
    --task "$TASK" --episodes "$EPISODES" --workers "$WORKERS" \
    --dynamics $T/configs/gear_*.yaml --out "out/sweep_$TASK.csv"
done
sha256sum out/zs.npz out/sweep_*.csv
