#!/bin/bash
#SBATCH --job-name=t32_qtrace
#SBATCH --partition=edr1-al9_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --array=0-14
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Q ranking and ensemble spread of the FB actor in the five dynamics configs (zero_shot_rl.eval.qtrace).
#
# One array task per (training seed, config): 3 seeds x 5 configs. Each task runs
# EPISODES episodes, with environment seeds 0..EPISODES-1 shared by
# all configs, and writes $OUT/qtrace_s<seed>_<config>.npz.
#
# Submit from a directory that holds logs/ and, unless REPO and ARTIFACTS are set, the
# checkout (code/) and the T04 and T06 artifacts (artifacts/):
#     mkdir -p logs && sbatch code/tasks/T3.2-ood-robustness/run_qtrace.sh
set -eu

EPISODES=${EPISODES:-30}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
REPO=${REPO:-$ROOT/code}
ARTIFACTS=${ARTIFACTS:-$ROOT/artifacts}
OUT=${OUT:-$ROOT/out}

CONFIGS=(baseline friction_mild friction_strong actuator_mild actuator_strong)
CHECKPOINTS=(T04-mcfb-pessimism/zsrl-32/800000.pt T04-mcfb-pessimism/zsrl-41/680000.pt T04-mcfb-pessimism/zsrl-42/200000.pt)
ENSEMBLES=(43 44 45)   # zero-shot-rl-<job> in T06; switch.py checks the ensemble belongs to the checkpoint
CQL_SWEEPS=(46 47 48)

SEED=$((SLURM_ARRAY_TASK_ID / ${#CONFIGS[@]}))
CONFIG=${CONFIGS[$((SLURM_ARRAY_TASK_ID % ${#CONFIGS[@]}))]}

export PATH=$HOME/.local/bin:$PATH UV_LINK_MODE=copy MUJOCO_GL=egl PYTHONPATH=.
L=/tmp/$USER-t32-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID
export UV_PROJECT_ENVIRONMENT=$L/venv UV_CACHE_DIR=$L/cache
mkdir -p "$L" "$OUT"
trap 'rm -rf "$L"' EXIT

cd "$REPO"
uv sync --no-progress
"$L/venv/bin/python" -m zero_shot_rl.eval.qtrace \
  --checkpoint "$ARTIFACTS/${CHECKPOINTS[$SEED]}" \
  --ensemble "$ARTIFACTS/T06-latent-switching/zero-shot-rl-${ENSEMBLES[$SEED]}/ensemble.pt" \
  --zs "tasks/T06-latent-switching/data/inputs/zs_cql_sweep_zero-shot-rl-${CQL_SWEEPS[$SEED]}.npz" \
  --episodes "$EPISODES" --workers "$SLURM_CPUS_PER_TASK" \
  --dynamics "tasks/T3.2-ood-robustness/configs/$CONFIG.yaml" \
  --out "$OUT/qtrace_s${SEED}_${CONFIG}.npz"
