#!/bin/bash
# Evaluate saved residual Q checkpoints (omega 0.1, after 10 training episodes) in a target
# dynamics environment.
#
#     tasks/T4.1.3.1-residual-soft-q/run_cross_eval.sh <train_env> <eval_env>
#
# Reads $ARTIFACTS/T4.1.3.1-residual-soft-q/omega_w0.1/<train_env>/residual_<task>.pt and writes
# out/cross.<train_env>_to_<eval_env>/eval.csv.
set -eu

TRAIN_ENV=$1
EVAL_ENV=$2
ARTIFACTS=${ARTIFACTS:-$HOME/zsrl-artifacts}
CHECKPOINT=$ARTIFACTS/T4.1.3.1-residual-soft-q/zero-shot-rl-297/660000.pt
Q_DIR=$ARTIFACTS/T4.1.3.1-residual-soft-q/omega_w0.1/$TRAIN_ENV

DYNAMICS=()
if [ "$EVAL_ENV" != clean ]; then
  DYNAMICS=(--dynamics "tasks/T3.2-ood-robustness/configs/actuator_$EVAL_ENV.yaml")
fi

export MUJOCO_GL=egl PYTHONPATH=. OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
uv run python -m zero_shot_rl.eval.value_correction_checkpoints \
  --checkpoint "$CHECKPOINT" --q_dir "$Q_DIR" \
  --out_dir "out/cross.${TRAIN_ENV}_to_${EVAL_ENV}" \
  --final --episodes 10 --eval_episodes 20 --eval_seed_start 5 "${DYNAMICS[@]}"
