#!/bin/bash
# Training data of the dynamics context encoder: walker episodes under the 300 dynamics sets of
# configs/multi_dynamics.yaml, 600 episodes per FB training seed (stand, walk, run, flip z_r).
#
# Usage (repo root):
#     CKPT_DIR=<dir with zsrl-32.pt zsrl-41.pt zsrl-42.pt> bash tasks/T4.1.3.2-fb-aa/run_multi_dynamics.sh
# The three seeds run in parallel, each with WORKERS processes -> out/multi_dynamics_seed$SEED.{npz,json}.
set -euo pipefail

CKPT_DIR=${CKPT_DIR:?}
WORKERS=${WORKERS:-5}
CKPTS=(zsrl-32 zsrl-41 zsrl-42)   # the T05 standard FB models, as in the T07 trajectories
LATENTS=(30 31 32)                # tasks/T05-opt-to-pessi/data/inputs/zs_fb_zero-shot-rl-<job>.npz

export PYTHONPATH=. MUJOCO_GL=egl
mkdir -p out
collect() {
  SEED=$1
  uv run python -m zero_shot_rl.data.multi_dynamics \
    --config tasks/T4.1.3.2-fb-aa/configs/multi_dynamics.yaml --seed "$SEED" \
    --checkpoint "$CKPT_DIR/${CKPTS[$SEED]}.pt" \
    --latents "tasks/T05-opt-to-pessi/data/inputs/zs_fb_zero-shot-rl-${LATENTS[$SEED]}.npz" \
    --out-dir out --workers "$WORKERS" > "out/multi_dynamics_seed$SEED.log" 2>&1
}

PIDS=()
for SEED in 0 1 2; do collect "$SEED" & PIDS+=($!); done
for pid in "${PIDS[@]}"; do wait "$pid"; done
grep -h "Wrote" out/multi_dynamics_seed*.log
sha256sum out/multi_dynamics_seed*.npz
