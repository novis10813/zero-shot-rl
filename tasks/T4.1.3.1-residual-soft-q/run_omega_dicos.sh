#!/bin/bash
#SBATCH --job-name=t4131_omega
#SBATCH --partition=a100_short-al9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-35
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# omega sweep of residual soft Q-learning (zero_shot_rl.eval.value_correction) on the selected FB
# checkpoint (FB seed 1, step 660000): omega 0.01, 0.03, 0.1 at alpha 0.1, unbounded candidates,
# Q 64x2, 10 training episodes with warmup 0, tested after 0, 1, ..., 10 episodes on 20 test
# environments seeded 5..24. Same runs as configs/omega_{clean,mild,strong}.yaml, on Slurm.
#
# One array task per (environment, omega, task): 3 x 3 x 4. Each task writes
# $OUT/omega.<env>_<task>_w<omega>/ (eval.csv, train.csv, q/).
#
# Submit from a directory that holds logs/ and, unless REPO, ARTIFACTS and DATASET are set, the
# checkout (code/) and the checkpoint under artifacts/:
#     mkdir -p logs && sbatch code/tasks/T4.1.3.1-residual-soft-q/run_omega_dicos.sh
set -eu

ROOT=${SLURM_SUBMIT_DIR:-$PWD}
REPO=${REPO:-$ROOT/code}
ARTIFACTS=${ARTIFACTS:-$ROOT/artifacts}
DATASET=${DATASET:-$ROOT/../zsrl-datasets/walker/rnd/dataset.npz}
OUT=${OUT:-$ROOT/out}

ENVS=(clean mild strong)
OMEGAS=(0.01 0.03 0.1)
TASKS=(stand walk run flip)

I=$SLURM_ARRAY_TASK_ID
ENV=${ENVS[$((I / 12))]}
OMEGA=${OMEGAS[$((I / 4 % 3))]}
TASK=${TASKS[$((I % 4))]}
DYNAMICS=()
if [ "$ENV" != clean ]; then
  DYNAMICS=(--dynamics "tasks/T3.2-ood-robustness/configs/actuator_$ENV.yaml")
fi

export PATH=$HOME/.local/bin:$PATH UV_LINK_MODE=copy MUJOCO_GL=egl PYTHONPATH=.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
L=/tmp/$USER-t4131-$SLURM_JOB_ID-$I
export UV_PROJECT_ENVIRONMENT=$L/venv UV_CACHE_DIR=$L/cache
mkdir -p "$L" "$OUT"
trap 'rm -rf "$L"' EXIT

cd "$REPO"
uv sync --no-progress
"$L/venv/bin/python" -m zero_shot_rl.eval.value_correction \
  --checkpoint "$ARTIFACTS/T4.1.3.1-residual-soft-q/zero-shot-rl-297/660000.pt" \
  --dataset_path "$DATASET" --out_dir "$OUT/omega.${ENV}_${TASK}_w${OMEGA}" \
  --tasks "$TASK" --variants residual --omega "$OMEGA" --alpha 0.1 --hidden 64 \
  --episodes 10 --eval_every 1 --warmup 0 --eval_episodes 20 --eval_seed_start 5 \
  "${DYNAMICS[@]}"
