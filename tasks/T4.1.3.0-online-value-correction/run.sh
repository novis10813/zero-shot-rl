#!/bin/bash
# Online value correction (zero_shot_rl.eval.value_correction) on one T04 FB seed.
#
#     tasks/T4.1.3.0-online-value-correction/run.sh <seed 0|1|2> <name> [value_correction args]
#
# writes out/<name>_s<seed>/ (eval.csv, train.csv, q/) under the job's working directory.
# ARTIFACTS holds the T04 checkpoints under T04-mcfb-pessimism/, DATASET the Walker RND dataset
# (tasks/T01-fb-reproduction/README.md section 1).
set -eu

SEED=$1
NAME=$2
shift 2
ARTIFACTS=${ARTIFACTS:-$HOME/zsrl-artifacts}
DATASET=${DATASET:-$HOME/zsrl-datasets/walker/rnd/dataset.npz}
CHECKPOINTS=(zsrl-32/800000.pt zsrl-41/680000.pt zsrl-42/200000.pt)

export MUJOCO_GL=egl PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
uv run python -m zero_shot_rl.eval.value_correction \
  --checkpoint "$ARTIFACTS/T04-mcfb-pessimism/${CHECKPOINTS[$SEED]}" --dataset_path "$DATASET" \
  --out_dir "out/${NAME}_s${SEED}" "$@"
