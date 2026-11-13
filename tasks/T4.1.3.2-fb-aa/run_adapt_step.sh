#!/bin/bash
# One-step check of the action adaptation (zero_shot_rl.eval.adapt_step): per FB training seed,
# with its transformer context model of run_context.sh, 4 tasks x 5 stock episodes (environment
# seeds 300000-300019), every 8th step from step 32, in the stock walker and the eight test
# dynamics of configs/test_dynamics.yaml, K in {5, 20, 50} x lr in {0.01, 0.05}. CPU, the
# three seeds in parallel.
#
# Usage (repo root):
#     CKPT_DIR=<dir with zsrl-32.pt zsrl-41.pt zsrl-42.pt> MODEL_DIR=<dir with context_s*_transformer_best.pt> \
#     bash tasks/T4.1.3.2-fb-aa/run_adapt_step.sh
# -> out/adapt_step_s$SEED.{npz,json,log}
set -euo pipefail

CKPT_DIR=${CKPT_DIR:?}
MODEL_DIR=${MODEL_DIR:?}
CKPTS=(zsrl-32 zsrl-41 zsrl-42)
LATENTS=(30 31 32)

export PYTHONPATH=. MUJOCO_GL=egl OMP_NUM_THREADS=4
mkdir -p out
sha256sum "$MODEL_DIR"/context_s*_transformer_best.pt
run() {
  SEED=$1
  uv run python -m zero_shot_rl.eval.adapt_step \
    --checkpoint "$CKPT_DIR/${CKPTS[$SEED]}.pt" \
    --latents "tasks/T05-opt-to-pessi/data/inputs/zs_fb_zero-shot-rl-${LATENTS[$SEED]}.npz" \
    --model "$MODEL_DIR/context_s${SEED}_transformer_best.pt" \
    --dynamics tasks/T4.1.3.2-fb-aa/configs/test_dynamics.yaml \
    --out "out/adapt_step_s$SEED.npz" --workers 5 > "out/adapt_step_s$SEED.log" 2>&1
}
PIDS=()
for SEED in 0 1 2; do run "$SEED" & PIDS+=($!); done
STATUS=0
for pid in "${PIDS[@]}"; do wait "$pid" || STATUS=1; done
grep -h -v -i warn out/adapt_step_s*.log
sha256sum out/adapt_step_s*.npz
exit $STATUS
