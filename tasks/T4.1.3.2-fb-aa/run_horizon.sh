#!/bin/bash
# K-step paired trajectories of T4.1.3.2: how the gap of each shift grows with the number of steps.
#
# Usage (repo root):
#     CKPT_DIR=<dir with zsrl-32.pt zsrl-41.pt zsrl-42.pt> [HORIZON=100] bash tasks/T4.1.3.2-fb-aa/run_horizon.sh
# Per FB training seed (the three T06 checkpoints, as in T4.1.2), episodes 0-19 in the stock
# environment, start steps 0, 50, ..., 850, open- and closed-loop replay of HORIZON steps in the
# stock and the eight shifted dynamics -> out/horizon_s$SEED.npz (the seeds run in parallel).
set -euo pipefail

CKPT_DIR=${CKPT_DIR:?}
HORIZON=${HORIZON:-100}
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

collect() {
  SEED=$1
  uv run python -m zero_shot_rl.eval.paired_horizon \
    --checkpoint "$CKPT_DIR/${CKPTS[$SEED]}.pt" \
    --zs "tasks/T06-latent-switching/data/inputs/zs_cql_sweep_zero-shot-rl-${CQL_SWEEPS[$SEED]}.npz" \
    --dynamics "${DYN[@]}" --episodes 20 --first_episode 0 --horizon "$HORIZON" \
    --out "out/horizon_s$SEED.npz" > "out/horizon_s$SEED.log" 2>&1
}

PIDS=()
for SEED in 0 1 2; do collect "$SEED" & PIDS+=($!); done
for pid in "${PIDS[@]}"; do wait "$pid"; done
cat out/horizon_s*.log
sha256sum out/horizon_s*.npz
