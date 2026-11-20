#!/usr/bin/env bash
# Launch T6-anfis REINFORCE optimization for one frozen FB checkpoint.
#
# Usage (repo root):
#   CHECKPOINT=<path/to/660000.pt> DATASET=<path/to/walker/rnd/dataset.npz> OUT_DIR=<out/s0> \
#     [ANFIS_CONFIG=configs/anfis/walker.yaml] [CONFIG=tasks/T6-anfis/configs/reinforce.yaml] \
#     [DYNAMICS=<shift.yaml>] bash tasks/T6-anfis/run_anfis_z.sh [extra args...]
#
# Aggregate multi-seed outputs with Table 6:
#   PYTHONPATH=. uv run python -m analysis.table6_aggregate out/s0/table6.csv out/s1/table6.csv ...
set -euo pipefail

CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to a 660000.pt FB checkpoint}
DATASET=${DATASET:?set DATASET to <domain>/rnd/dataset.npz}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
ANFIS_CONFIG=${ANFIS_CONFIG:-configs/anfis/walker.yaml}
CONFIG=${CONFIG:-tasks/T6-anfis/configs/reinforce.yaml}

ARGS=(
  --anfis_config "$ANFIS_CONFIG"
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --dataset_path "$DATASET"
  --out_dir "$OUT_DIR"
)
if [[ -n "${DYNAMICS:-}" ]]; then
  ARGS+=(--dynamics "$DYNAMICS")
fi

exec bash scripts/finetune.sh anfis-z "${ARGS[@]}" "$@"
