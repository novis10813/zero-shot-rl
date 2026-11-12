#!/bin/bash
# Single-step stock-vs-shift classifiers of T4.1.3.2 on the paired data of T4.1.2.1 and T4.1.2.2.
#
# Usage (repo root):
#     CKPT_DIR=<dir with zsrl-32.pt zsrl-41.pt zsrl-42.pt> bash tasks/T4.1.3.2-fb-aa/run_classifier.sh
# Per FB training seed (the three T06 checkpoints), the test pairs of T4.1.2.2: episodes 0-19 in
# the stock environment replayed one step in the stock and the eight shifted dynamics
# -> out/paired_raw_s$SEED.npz (the seeds run in parallel). Then one classifier per shifted config
# on the pooled files -> out/classifier.npz (log: out/classifier.log).
set -euo pipefail

CKPT_DIR=${CKPT_DIR:?}
CKPTS=(zsrl-32 zsrl-41 zsrl-42)   # tasks/T06-latent-switching/configs/sources.json
CQL_SWEEPS=(46 47 48)
CONFIGS=(baseline friction_mild friction_strong actuator_mild actuator_strong)   # tasks/T3.2-ood-robustness/configs
HIGH=(friction_high_mild friction_high_strong actuator_high_mild actuator_high_strong)   # tasks/T4.1.2-dynamic-shift-detection/configs
CFG=tasks/T3.2-ood-robustness/configs

export PYTHONPATH=. MUJOCO_GL=egl
mkdir -p out
DYN=()
for c in "${CONFIGS[@]}"; do DYN+=("$CFG/$c.yaml"); done
for c in "${HIGH[@]}"; do DYN+=("tasks/T4.1.2-dynamic-shift-detection/configs/$c.yaml"); done

collect() {
  SEED=$1
  [ -f "out/paired_raw_s$SEED.npz" ] || uv run python -m zero_shot_rl.eval.paired_step \
    --checkpoint "$CKPT_DIR/${CKPTS[$SEED]}.pt" \
    --zs "tasks/T06-latent-switching/data/inputs/zs_cql_sweep_zero-shot-rl-${CQL_SWEEPS[$SEED]}.npz" \
    --dynamics "${DYN[@]}" --episodes 20 --first_episode 0 --out "out/paired_raw_s$SEED.npz"
}

PIDS=()
for SEED in 0 1 2; do collect "$SEED" & PIDS+=($!); done
for pid in "${PIDS[@]}"; do wait "$pid"; done
sha256sum out/paired_raw_s*.npz

PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.classifier \
  --paired out/paired_raw_s0.npz out/paired_raw_s1.npz out/paired_raw_s2.npz \
  --out out/classifier.npz 2>&1 | tee out/classifier.log
sha256sum out/classifier.npz
