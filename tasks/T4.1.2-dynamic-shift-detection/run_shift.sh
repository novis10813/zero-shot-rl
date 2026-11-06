#!/bin/bash
# Fit a shift detector for one FB training seed and score rollouts in the five T3.2 dynamics.
#
# Usage (repo root, on a machine with the T04 checkpoints and the walker rnd dataset):
#     SEED=0 CKPT_DIR=~/t05_fb_ckpt DATASET=~/zsrl-datasets/walker/rnd/dataset.npz \
#         bash tasks/T4.1.2-dynamic-shift-detection/run_shift.sh
# Writes out/detector_s$SEED.pt and out/shift_s${SEED}_<config>.npz.
set -eu

SEED=${SEED:?set SEED to 0, 1 or 2}
EPISODES=${EPISODES:-20}
CKPT_DIR=${CKPT_DIR:?}
DATASET=${DATASET:?}

CKPTS=(zsrl-32 zsrl-41 zsrl-42)   # T04 checkpoints of T06 (tasks/T06-latent-switching/configs/sources.json)
CQL_SWEEPS=(46 47 48)
CONFIGS=(baseline friction_mild friction_strong actuator_mild actuator_strong)
CKPT=$CKPT_DIR/${CKPTS[$SEED]}.pt
ZS=tasks/T06-latent-switching/data/inputs/zs_cql_sweep_zero-shot-rl-${CQL_SWEEPS[$SEED]}.npz

export PYTHONPATH=. MUJOCO_GL=egl
mkdir -p out
uv run python -m zero_shot_rl.latent.shift_detector --checkpoint "$CKPT" --dataset_path "$DATASET" \
  --zs "$ZS" --seed "$SEED" --out "out/detector_s$SEED.pt"

DYN=()
for c in "${CONFIGS[@]}"; do DYN+=("tasks/T3.2-ood-robustness/configs/$c.yaml"); done
uv run python -m zero_shot_rl.eval.shift_trace --checkpoint "$CKPT" --detector "out/detector_s$SEED.pt" \
  --zs "$ZS" --dynamics "${DYN[@]}" --episodes "$EPISODES" --workers 5 --out_dir "out/s$SEED"
for c in "${CONFIGS[@]}"; do mv "out/s$SEED/$c.npz" "out/shift_s${SEED}_$c.npz"; done
