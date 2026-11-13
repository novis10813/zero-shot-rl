#!/bin/bash
# zero_shot_rl.eval.value_correction on the selected FB checkpoint of T4.1.3.1 (FB seed 1,
# job zero-shot-rl-297, step 660000).
#
#     tasks/T4.1.3.1-residual-soft-q/run.sh <name> [value_correction args]
#
# writes out/<name>/ (eval.csv, train.csv, q/) under the job's working directory.
# ARTIFACTS holds the checkpoint under T4.1.3.1-residual-soft-q/, DATASET the Walker RND dataset
# (tasks/T01-fb-reproduction/README.md section 1).
set -eu

NAME=$1
shift
ARTIFACTS=${ARTIFACTS:-$HOME/zsrl-artifacts}
DATASET=${DATASET:-$HOME/zsrl-datasets/walker/rnd/dataset.npz}
CHECKPOINT=$ARTIFACTS/T4.1.3.1-residual-soft-q/zero-shot-rl-297/660000.pt

export MUJOCO_GL=egl PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
uv run python -m zero_shot_rl.eval.value_correction \
  --checkpoint "$CHECKPOINT" --dataset_path "$DATASET" --out_dir "out/$NAME" "$@"
