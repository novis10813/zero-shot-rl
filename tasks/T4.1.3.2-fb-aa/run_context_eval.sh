#!/bin/bash
# Evaluation of the context models of run_context.sh: test episodes in the stock and the eight
# shifted dynamics (configs/test_dynamics.yaml, 240 episodes per dynamics: 20 per FB seed and task),
# then every model scores them with zero_shot_rl.latent.context_score.
#
# Usage (repo root):
#     CKPT_DIR=<dir with zsrl-32.pt zsrl-41.pt zsrl-42.pt> MODEL_DIR=<dir with context_s*_*_best.pt> \
#     STOCK_DIR=<dir with the T07 fb_trajectories_seed{0,1,2}.npz> bash tasks/T4.1.3.2-fb-aa/run_context_eval.sh
# -> out/test/multi_dynamics_seed{0,1,2}.{npz,json,log}, out/score_s$SEED_$ENCODER.{npz,log}
set -euo pipefail

CKPT_DIR=${CKPT_DIR:?}
MODEL_DIR=${MODEL_DIR:?}
STOCK_DIR=${STOCK_DIR:?}
CKPTS=(zsrl-32 zsrl-41 zsrl-42)
LATENTS=(30 31 32)

export PYTHONPATH=. MUJOCO_GL=egl OMP_NUM_THREADS=4
mkdir -p out/test
sha256sum "$MODEL_DIR"/context_s*_best.pt "$STOCK_DIR"/fb_trajectories_seed*.npz
collect() {
  SEED=$1
  uv run python -m zero_shot_rl.data.multi_dynamics \
    --config tasks/T4.1.3.2-fb-aa/configs/test_dynamics.yaml --seed "$SEED" \
    --checkpoint "$CKPT_DIR/${CKPTS[$SEED]}.pt" \
    --latents "tasks/T05-opt-to-pessi/data/inputs/zs_fb_zero-shot-rl-${LATENTS[$SEED]}.npz" \
    --out-dir out/test --workers 5 > "out/test/multi_dynamics_seed$SEED.log" 2>&1
}
PIDS=()
for SEED in 0 1 2; do collect "$SEED" & PIDS+=($!); done
for pid in "${PIDS[@]}"; do wait "$pid"; done
sha256sum out/test/multi_dynamics_seed*.npz

score() {
  SEED=$1 ENCODER=$2
  uv run python -m zero_shot_rl.latent.context_score \
    --model "$MODEL_DIR/context_s${SEED}_${ENCODER}_best.pt" --checkpoint "$CKPT_DIR/${CKPTS[$SEED]}.pt" \
    --test out/test/multi_dynamics_seed{0,1,2}.npz --stock "$STOCK_DIR"/fb_trajectories_seed{0,1,2}.npz \
    --out "out/score_s${SEED}_${ENCODER}.npz" > "out/score_s${SEED}_${ENCODER}.log" 2>&1
}
PIDS=()
for SEED in 0 1 2; do
  for ENCODER in transformer mean; do score "$SEED" "$ENCODER" & PIDS+=($!); done
done
STATUS=0
for pid in "${PIDS[@]}"; do wait "$pid" || STATUS=1; done
cat out/score_s*.log | grep -v -i warn
sha256sum out/score_s*.npz
exit $STATUS
