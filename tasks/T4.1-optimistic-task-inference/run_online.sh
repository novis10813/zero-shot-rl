#!/bin/bash
# Online task inference (zero_shot_rl.eval.online_inference) on one T04 FB seed.
#
#     tasks/T4.1-optimistic-task-inference/run_online.sh <seed 0|1|2> <name> [online_inference args]
#
# writes out/<name>_s<seed>.csv, out/<name>_s<seed>_z/ and out/<name>_s<seed>_pred/ under the job's working directory.
# ARTIFACTS holds the T04 checkpoints under T04-mcfb-pessimism/, DATASET the Walker RND dataset
# (tasks/T01-fb-reproduction/README.md section 1).
set -eu

SEED=$1
NAME=$2
shift 2
ARTIFACTS=${ARTIFACTS:-$HOME/zsrl-artifacts}
DATASET=${DATASET:-$HOME/zsrl-datasets/walker/rnd/dataset.npz}
CHECKPOINTS=(zsrl-32/800000.pt zsrl-41/680000.pt zsrl-42/200000.pt)

export MUJOCO_GL=egl PYTHONPATH=.
uv run python -m zero_shot_rl.eval.online_inference \
  --checkpoint "$ARTIFACTS/T04-mcfb-pessimism/${CHECKPOINTS[$SEED]}" --dataset_path "$DATASET" \
  --out "out/${NAME}_s${SEED}.csv" --z_dir "out/${NAME}_s${SEED}_z" \
  --predictions_dir "out/${NAME}_s${SEED}_pred" "$@"
