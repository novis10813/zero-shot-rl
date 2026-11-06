#!/bin/bash
# Paired one-step test of the shift detector for one FB training seed.
#
# Usage (repo root, on a machine with the T04 checkpoints and the walker rnd dataset):
#     SEED=0 CKPT_DIR=~/t05_fb_ckpt DATASET=~/zsrl-datasets/walker/rnd/dataset.npz \
#         bash tasks/T4.1.2-dynamic-shift-detection/run_paired.sh
# Writes out/scored_dataset_s$SEED.npz and out/scored_rollout_s$SEED.npz: the stock rollout of
# episodes 0-19 replayed one step in every config of T3.2 and scored by the detector fitted on
# the dataset, and by the one fitted on stock rollouts of episodes 100-119.
set -eu

SEED=${SEED:?set SEED to 0, 1 or 2}
CKPT_DIR=${CKPT_DIR:?}
DATASET=${DATASET:?}

CKPTS=(zsrl-32 zsrl-41 zsrl-42)   # T04 checkpoints of T06 (tasks/T06-latent-switching/configs/sources.json)
CQL_SWEEPS=(46 47 48)
CONFIGS=(baseline friction_mild friction_strong actuator_mild actuator_strong)
CKPT=$CKPT_DIR/${CKPTS[$SEED]}.pt
ZS=tasks/T06-latent-switching/data/inputs/zs_cql_sweep_zero-shot-rl-${CQL_SWEEPS[$SEED]}.npz
CFG=tasks/T3.2-ood-robustness/configs

export PYTHONPATH=. MUJOCO_GL=egl
mkdir -p out
DYN=()
for c in "${CONFIGS[@]}"; do DYN+=("$CFG/$c.yaml"); done

uv run python -m zero_shot_rl.eval.paired_step --checkpoint "$CKPT" --zs "$ZS" \
  --dynamics "${DYN[@]}" --episodes 20 --first_episode 0 --out "out/paired_raw_s$SEED.npz"
uv run python -m zero_shot_rl.eval.paired_step --checkpoint "$CKPT" --zs "$ZS" \
  --dynamics "$CFG/baseline.yaml" --episodes 20 --first_episode 100 --out "out/train_rollouts_s$SEED.npz"

uv run python -m zero_shot_rl.latent.shift_detector --checkpoint "$CKPT" --dataset_path "$DATASET" \
  --zs "$ZS" --seed "$SEED" --out "out/detector_s$SEED.pt"
uv run python -m zero_shot_rl.latent.shift_detector --checkpoint "$CKPT" --rollouts "out/train_rollouts_s$SEED.npz" \
  --zs "$ZS" --seed "$SEED" --out "out/detector_rollout_s$SEED.pt"

uv run python -m zero_shot_rl.eval.paired_score --checkpoint "$CKPT" --detector "out/detector_s$SEED.pt" \
  --zs "$ZS" --paired "out/paired_raw_s$SEED.npz" --out "out/scored_dataset_s$SEED.npz"
uv run python -m zero_shot_rl.eval.paired_score --checkpoint "$CKPT" --detector "out/detector_rollout_s$SEED.pt" \
  --zs "$ZS" --paired "out/paired_raw_s$SEED.npz" --out "out/scored_rollout_s$SEED.npz"
