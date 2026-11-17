#!/bin/bash
# Evaluate the residual Q checkpoints of the 10 finetune seeds (omega 0.1, after 10 training
# episodes) in a target dynamics environment, with the frozen FB actor at the same z.
#
#     tasks/T4.1.3.1-residual-soft-q/run_cross_eval_seeds.sh <train_env> <eval_env>
#
# Reads $ARTIFACTS/T4.1.3.1-residual-soft-q/finetune/ft.<train_env>_<task>_s<seed>/q/residual_<task>.pt
# and writes out/crossseeds.<train_env>_to_<eval_env>/s<seed>/eval.csv for seeds 1000..1009.
set -eu

TRAIN_ENV=$1
EVAL_ENV=$2
ARTIFACTS=${ARTIFACTS:-$HOME/zsrl-artifacts}
FINETUNE=$ARTIFACTS/T4.1.3.1-residual-soft-q/finetune
CHECKPOINT=$ARTIFACTS/T4.1.3.1-residual-soft-q/zero-shot-rl-297/660000.pt

DYNAMICS=()
if [ "$EVAL_ENV" != clean ]; then
  DYNAMICS=(--dynamics "tasks/T3.2-ood-robustness/configs/actuator_$EVAL_ENV.yaml")
fi

export MUJOCO_GL=egl PYTHONPATH=. OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
for SEED in $(seq 1000 1009); do
  Q_DIR=$(mktemp -d)
  for TASK in stand walk run flip; do
    cp "$FINETUNE/ft.${TRAIN_ENV}_${TASK}_s${SEED}/q/residual_${TASK}.pt" "$Q_DIR/"
  done
  uv run python -m zero_shot_rl.eval.value_correction_checkpoints \
    --checkpoint "$CHECKPOINT" --q_dir "$Q_DIR" \
    --out_dir "out/crossseeds.${TRAIN_ENV}_to_${EVAL_ENV}/s${SEED}" \
    --final --frozen --episodes 10 --eval_episodes 20 --eval_seed_start 5 "${DYNAMICS[@]}"
  rm -rf "$Q_DIR"
done
