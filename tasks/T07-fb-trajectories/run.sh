#!/usr/bin/env bash
# Trajectories of the three T05 standard FB models with their original z_r on walker.
# 50 episodes (environment seeds 0..49) per task; writes under artifacts/ of the job's checkout.
# CKPT_ROOT: directory holding the config's checkpoint paths (default: the checkout).
# Usage (repo root, via lab): bash tasks/T07-fb-trajectories/run.sh
set -euo pipefail
OUT=${OUT:-artifacts/T07-fb-trajectories/trajectories}
CKPT_ROOT=${CKPT_ROOT:-.}
PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.fb_trajectories \
  --config tasks/T07-fb-trajectories/configs/sources.json --out-dir "$OUT" \
  --checkpoint-root "$CKPT_ROOT" --episodes 50 --batch 20 --workers 4 \
  --compare-trace tasks/T05-opt-to-pessi/data/inputs/behavior_fb_seed{0,1,2}.npz
