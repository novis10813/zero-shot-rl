#!/usr/bin/env bash
# Trajectories of the three T05 standard FB models with their original z_r on walker.
# 50 episodes (environment seeds 0..49) per task; writes under artifacts/ of the job's checkout.
# Usage (repo root, via lab): bash tasks/T07-fb-trajectories/run.sh
set -euo pipefail
OUT=${OUT:-artifacts/T07-fb-trajectories/trajectories}
PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.fb_trajectories \
  --config tasks/T07-fb-trajectories/configs/sources.json --out-dir "$OUT" \
  --episodes 50 --batch 20 --workers 4 \
  --compare-trace tasks/T05-opt-to-pessi/data/inputs/behavior_fb_seed{0,1,2}.npz
