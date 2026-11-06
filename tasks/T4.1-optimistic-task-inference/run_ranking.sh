#!/bin/bash
# Predicted against actual return of fixed candidate latents (zero_shot_rl.eval.latent_ranking)
# on one T04 FB seed and one dynamics config, with the beliefs of the OpTI-BFM and
# OpTI-BFM-TS prediction runs committed under data/inputs/predictions/.
#
#     tasks/T4.1-optimistic-task-inference/run_ranking.sh <seed 0|1|2> <shift>
#
# <shift> is baseline or the name of a tasks/T3.2-ood-robustness/configs yaml. Writes
# out/ranking_<shift>_s<seed>.csv. ARTIFACTS and DATASET as in run_online.sh.
set -eu

SEED=$1
SHIFT=$2
ARTIFACTS=${ARTIFACTS:-$HOME/zsrl-artifacts}
DATASET=${DATASET:-$HOME/zsrl-datasets/walker/rnd/dataset.npz}
CHECKPOINTS=(zsrl-32/800000.pt zsrl-41/680000.pt zsrl-42/200000.pt)
P=tasks/T4.1-optimistic-task-inference/data/inputs/predictions
DYNAMICS=()
if [ "$SHIFT" != baseline ]; then
  DYNAMICS=(--dynamics "tasks/T3.2-ood-robustness/configs/$SHIFT.yaml")
fi

export MUJOCO_GL=egl PYTHONPATH=.
uv run python -m zero_shot_rl.eval.latent_ranking \
  --checkpoint "$ARTIFACTS/T04-mcfb-pessimism/${CHECKPOINTS[$SEED]}" --dataset_path "$DATASET" \
  --beliefs OpTI-BFM "$P/$SHIFT.ucb_s${SEED}_pred/ucb:0.001.npz" \
  --beliefs OpTI-BFM-TS "$P/$SHIFT.ts_s${SEED}_pred/ts:0.1.npz" \
  --out "out/ranking_${SHIFT}_s${SEED}.csv" "${DYNAMICS[@]}"
