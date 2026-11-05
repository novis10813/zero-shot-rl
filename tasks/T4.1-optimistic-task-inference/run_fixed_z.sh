#!/bin/bash
# Data efficiency (zero_shot_rl.eval.fixed_z_returns) on one T04 FB seed: the least-squares z
# from the first n transitions of the main Random, OpTI-BFM and OpTI-BFM-TS runs and of RND
# dataset episodes.
#
#     tasks/T4.1-optimistic-task-inference/run_fixed_z.sh <seed 0|1|2>
#
# writes out/fixed_z_s<seed>.csv under the job's working directory. ARTIFACTS and DATASET as
# in run_online.sh.
set -eu

SEED=$1
ARTIFACTS=${ARTIFACTS:-$HOME/zsrl-artifacts}
DATASET=${DATASET:-$HOME/zsrl-datasets/walker/rnd/dataset.npz}
CHECKPOINTS=(zsrl-32/800000.pt zsrl-41/680000.pt zsrl-42/200000.pt)
M=tasks/T4.1-optimistic-task-inference/data/inputs/main

export MUJOCO_GL=egl PYTHONPATH=.
uv run python -m zero_shot_rl.eval.fixed_z_returns \
  --checkpoint "$ARTIFACTS/T04-mcfb-pessimism/${CHECKPOINTS[$SEED]}" --dataset_path "$DATASET" \
  --z Random "$M/base_s${SEED}_z/random.npz" --z OpTI-BFM "$M/ucb_s${SEED}_z/ucb:0.001.npz" \
  --z OpTI-BFM-TS "$M/ts_s${SEED}_z/ts:0.1.npz" --rnd RND --out "out/fixed_z_s${SEED}.csv"
