#!/bin/bash
# Action-conditioned dynamics detection of T4.1.2.1.
#
# Usage (repo root):
#     CKPT_DIR=<dir with zsrl-32.pt zsrl-41.pt zsrl-42.pt> DATASET=<walker/rnd/dataset.npz> bash tasks/T4.1.2.1-action-conditioned-detection/run_dynamics.sh
# Per FB training seed (the three T06 checkpoints, as in T4.1.2):
#   - test pairs: episodes 0-19 in the stock environment replayed one step in the stock and the
#     eight shifted dynamics -> out/paired_raw_s$SEED.npz
#   - training rollouts: stock episodes 100-119 -> out/train_rollouts_s$SEED.npz
# Then one dynamics model on the pooled training rollouts of the three seeds, and the scores of
# the pooled test pairs (the inputs of the model are standardised with the statistics of the full
# ExORL dataset) -> out/dynamics.pt, out/dynamics_scored.npz.
set -eu

CKPT_DIR=${CKPT_DIR:?}
DATASET=${DATASET:?}
CKPTS=(zsrl-32 zsrl-41 zsrl-42)   # tasks/T06-latent-switching/configs/sources.json
CQL_SWEEPS=(46 47 48)
CONFIGS=(baseline friction_mild friction_strong actuator_mild actuator_strong)   # tasks/T3.2-ood-robustness/configs
HIGH=(friction_high_mild friction_high_strong actuator_high_mild actuator_high_strong)   # tasks/T4.1.2-dynamic-shift-detection/configs
CFG=tasks/T3.2-ood-robustness/configs

export PYTHONPATH=. MUJOCO_GL=egl
mkdir -p out
DYN=()
for c in "${CONFIGS[@]}"; do DYN+=("$CFG/$c.yaml"); done
for c in "${HIGH[@]}"; do DYN+=("tasks/T4.1.2-dynamic-shift-detection/configs/$c.yaml"); done

for SEED in 0 1 2; do
  CKPT=$CKPT_DIR/${CKPTS[$SEED]}.pt
  ZS=tasks/T06-latent-switching/data/inputs/zs_cql_sweep_zero-shot-rl-${CQL_SWEEPS[$SEED]}.npz
  [ -f "out/paired_raw_s$SEED.npz" ] || uv run python -m zero_shot_rl.eval.paired_step --checkpoint "$CKPT" --zs "$ZS" \
    --dynamics "${DYN[@]}" --episodes 20 --first_episode 0 --out "out/paired_raw_s$SEED.npz"
  [ -f "out/train_rollouts_s$SEED.npz" ] || uv run python -m zero_shot_rl.eval.paired_step --checkpoint "$CKPT" --zs "$ZS" \
    --dynamics "$CFG/baseline.yaml" --episodes 20 --first_episode 100 --out "out/train_rollouts_s$SEED.npz"
done

uv run python -m zero_shot_rl.latent.dynamics_model \
  --rollouts out/train_rollouts_s0.npz out/train_rollouts_s1.npz out/train_rollouts_s2.npz \
  --stats_dataset "$DATASET" --val_episodes 4 --seed 0 --out out/dynamics.pt
uv run python -m zero_shot_rl.latent.dynamics_score --model out/dynamics.pt \
  --paired out/paired_raw_s0.npz out/paired_raw_s1.npz out/paired_raw_s2.npz --out out/dynamics_scored.npz
