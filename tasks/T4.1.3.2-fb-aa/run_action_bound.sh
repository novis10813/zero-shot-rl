#!/bin/bash
# Action-bound ceiling of the one-step adaptation (zero_shot_rl.eval.action_bound): per FB training
# seed, 4 tasks x 5 stock episodes (environment seeds 300000-300019), every step, in the stock
# walker and the eight test dynamics of configs/test_dynamics.yaml, with the stock action and the
# best action inside [-1, 1] for the gear shift. CPU, the three seeds in parallel, no training.
#
# Usage (repo root):
#     CKPT_DIR=<dir with zsrl-32.pt zsrl-41.pt zsrl-42.pt> bash tasks/T4.1.3.2-fb-aa/run_action_bound.sh
# -> out/action_bound_s$SEED.{npz,json,log}
set -euo pipefail

CKPT_DIR=${CKPT_DIR:?}
CKPTS=(zsrl-32 zsrl-41 zsrl-42)
LATENTS=(30 31 32)

export PYTHONPATH=. MUJOCO_GL=egl OMP_NUM_THREADS=4
mkdir -p out
run() {
  SEED=$1
  uv run python -m zero_shot_rl.eval.action_bound \
    --checkpoint "$CKPT_DIR/${CKPTS[$SEED]}.pt" \
    --latents "tasks/T05-opt-to-pessi/data/inputs/zs_fb_zero-shot-rl-${LATENTS[$SEED]}.npz" \
    --dynamics tasks/T4.1.3.2-fb-aa/configs/test_dynamics.yaml \
    --out "out/action_bound_s$SEED.npz" --workers 5 > "out/action_bound_s$SEED.log" 2>&1
}
PIDS=()
for SEED in 0 1 2; do run "$SEED" & PIDS+=($!); done
STATUS=0
for pid in "${PIDS[@]}"; do wait "$pid" || STATUS=1; done
grep -h -v -i warn out/action_bound_s*.log
sha256sum out/action_bound_s*.npz
exit $STATUS
