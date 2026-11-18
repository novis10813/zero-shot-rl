#!/usr/bin/env bash
# Dedicated evaluation launcher (zero_shot_rl.eval).
#
# Examples:
#   bash scripts/evaluate.sh checkpoint <fb.pt> walker --dataset_path <dataset.npz> --eval_tasks stand walk run flip
#   bash scripts/evaluate.sh dynamics --checkpoint <fb.pt> --zs <zs.npz> --dynamics <a.yaml> <b.yaml> --out out/sweep.csv
#   bash scripts/evaluate.sh value-correction --checkpoint <fb.pt> --q_dir out/vc/q --final --out_dir out/vc_eval
#   bash scripts/evaluate.sh action-transform --checkpoint <fb.pt> --dataset <dataset.npz> --ft-dir out/ft_aa --dynamics <dyn.yaml> --out out/eval_aa.npz
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage: bash scripts/evaluate.sh <mode> [args...]

Dedicated evaluation launcher (no model training or finetuning).

Modes:
  checkpoint (ckpt)       Evaluate a saved FB / CFB / RL checkpoint on ExORL tasks (optional --dynamics, --out)
  dynamics                Sweep an FB checkpoint across shifted dynamics YAMLs (dynamics_sweep)
  value-correction (vc)   Evaluate saved Q_adapt checkpoints (value_correction_checkpoints)
  action-transform (aa)   Evaluate saved ActionTransformer checkpoints (eval.action_transform)
  context-score           Score test episodes with a trained context encoder
  dynamics-score          Score paired transitions with a trained dynamics model
  cvae-score              Score paired transitions with a trained CVAE ensemble
  adapt-episode           Closed-loop evaluation of context-based action adaptation
  latents                 Roll out stored FB latents
  switch                  Evaluate Rule C optimistic/pessimistic switching
  qtrace                  Evaluate ensemble Q ranking and spread
  <other>                 Any command supported by `python -m zero_shot_rl.eval`
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
  checkpoint|ckpt)
    echo "[EVAL] mode=checkpoint"
    exec uv run python -m zero_shot_rl.eval.checkpoint "$@"
    ;;
  dynamics|dynamics-sweep|dynamics_sweep)
    echo "[EVAL] mode=dynamics"
    exec uv run python -m zero_shot_rl.eval.dynamics_sweep "$@"
    ;;
  value-correction|value_correction|vc)
    echo "[EVAL] mode=value-correction"
    exec uv run python -m zero_shot_rl.eval.value_correction_checkpoints "$@"
    ;;
  action-transform|action_transform|aa)
    echo "[EVAL] mode=action-transform"
    exec uv run python -m zero_shot_rl.eval.action_transform "$@"
    ;;
  context-score|context_score)
    echo "[EVAL] mode=context-score"
    exec uv run python -m zero_shot_rl.eval.context_score "$@"
    ;;
  dynamics-score|dynamics_score)
    echo "[EVAL] mode=dynamics-score"
    exec uv run python -m zero_shot_rl.eval.dynamics_score "$@"
    ;;
  cvae-score|cvae_score)
    echo "[EVAL] mode=cvae-score"
    exec uv run python -m zero_shot_rl.eval.cvae_score "$@"
    ;;
  *)
    SUBCMD=${MODE//-/_}
    echo "[EVAL] mode=$SUBCMD"
    exec uv run python -m zero_shot_rl.eval "$SUBCMD" "$@"
    ;;
esac
