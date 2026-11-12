#!/bin/bash
# Context encoder and B_0 predictor of T4.1.3.2 (zero_shot_rl.latent.context_encoder): one model
# per FB training seed (its own B_0 as target) and encoder (transformer, mean), all six in
# parallel on one GPU.
#
# Usage (repo root):
#     CKPT_DIR=<dir with zsrl-32.pt zsrl-41.pt zsrl-42.pt> \
#     DATA_DIR=<dir with multi_dynamics_seed{0,1,2}.npz of run_multi_dynamics.sh> \
#     STOCK_DIR=<dir with the T07 fb_trajectories_seed{0,1,2}.npz> [STEPS=100000] \
#     bash tasks/T4.1.3.2-fb-aa/run_context.sh
# -> out/context_s$SEED_$ENCODER_{best,final}.pt, _history.json, .log
set -euo pipefail

CKPT_DIR=${CKPT_DIR:?}
DATA_DIR=${DATA_DIR:?}
STOCK_DIR=${STOCK_DIR:?}
STEPS=${STEPS:-100000}
CKPTS=(zsrl-32 zsrl-41 zsrl-42)   # the T05 standard FB models, as in the data

export PYTHONPATH=. OMP_NUM_THREADS=4
mkdir -p out
sha256sum "$DATA_DIR"/multi_dynamics_seed*.npz "$STOCK_DIR"/fb_trajectories_seed*.npz
train() {
  SEED=$1 ENCODER=$2
  uv run python -m zero_shot_rl.latent.context_encoder \
    --data "$DATA_DIR"/multi_dynamics_seed{0,1,2}.npz --stock "$STOCK_DIR"/fb_trajectories_seed{0,1,2}.npz \
    --checkpoint "$CKPT_DIR/${CKPTS[$SEED]}.pt" --encoder "$ENCODER" --steps "$STEPS" \
    --out-dir out --name "context_s${SEED}_${ENCODER}" > "out/context_s${SEED}_${ENCODER}.log" 2>&1
}

PIDS=()
for SEED in 0 1 2; do
  for ENCODER in transformer mean; do train "$SEED" "$ENCODER" & PIDS+=($!); done
done
STATUS=0
for pid in "${PIDS[@]}"; do wait "$pid" || STATUS=1; done
tail -n 3 out/context_s*.log
exit $STATUS
