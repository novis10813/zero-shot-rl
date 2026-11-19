#!/usr/bin/env bash
# Dedicated test-time finetuning / adaptation launcher (zero_shot_rl.finetune).
#
# Examples:
#   bash scripts/finetune.sh value-correction --checkpoint <fb.pt> --dataset_path <dataset.npz> --dynamics <shift.yaml> --out_dir out/vc
#   bash scripts/finetune.sh action-transform --checkpoint <fb.pt> --dataset <dataset.npz> --g0 <g0_best.pt> --dynamics <dyn.yaml> --out-dir out/ft_aa
#   bash scripts/finetune.sh online-inference --checkpoint <fb.pt> --dataset_path <dataset.npz> --methods ucb:0.1 ts:0.01 --out out/opti.csv
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage: bash scripts/finetune.sh <mode> [args...]

Dedicated test-time finetuning launcher (separates adaptation from evaluation).

Modes:
  value-correction (vc)   Online Q_adapt / residual soft-Q finetuning (saves q/*.pt, adds --train_only by default; pass --with-eval to also run inline eval)
  action-transform (aa)   FB-AA test-time g_1 + T_theta finetuning (saves <out-dir>/<config>/transformer.pt)
  online-inference (opti) OpTI-BFM online task-vector adaptation (UCB / TS / Random / FB)
  anfis-z (anfis)         REINFORCE optimization of YAML-composed ANFIS -> FB latent z

Evaluate saved finetuned models separately with:
  bash scripts/evaluate.sh value-correction ...
  bash scripts/evaluate.sh action-transform ...
EOF
}

if [[ $# -lt 1 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
  usage
  exit 0
fi

MODE=$1
shift

export PYTHONPATH=. PYTHONUNBUFFERED=1 MUJOCO_GL=${MUJOCO_GL:-egl}

case "$MODE" in
  value-correction|value_correction|vc)
    WITH_EVAL=0
    FWD_ARGS=()
    for arg in "$@"; do
      if [[ "$arg" == "--with-eval" ]]; then
        WITH_EVAL=1
      else
        FWD_ARGS+=("$arg")
      fi
    done
    if [[ "$WITH_EVAL" -eq 0 ]]; then
      FWD_ARGS=(--train_only "${FWD_ARGS[@]}")
    fi
    echo "[FINETUNE] mode=value-correction train_only=$((1 - WITH_EVAL))"
    exec uv run python -m zero_shot_rl.finetune.value_correction "${FWD_ARGS[@]}"
    ;;
  action-transform|action_transform|aa)
    echo "[FINETUNE] mode=action-transform"
    exec uv run python -m zero_shot_rl.finetune.action_transform "$@"
    ;;
  online-inference|online_inference|opti)
    echo "[FINETUNE] mode=online-inference"
    exec uv run python -m zero_shot_rl.finetune.online_inference "$@"
    ;;
  anfis-z|anfis_z|anfis)
    echo "[FINETUNE] mode=anfis-z"
    exec uv run python -m zero_shot_rl.finetune.anfis_z "$@"
    ;;
  *)
    echo "Unknown finetune mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac
