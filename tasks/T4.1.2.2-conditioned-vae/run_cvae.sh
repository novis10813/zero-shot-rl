#!/bin/bash
# CVAE ensemble detection of T4.1.2.2 (same data and split as T4.1.2.1).
#
# Usage (repo root):
#     CKPT_DIR=<dir with zsrl-32.pt zsrl-41.pt zsrl-42.pt> [MEMBERS=5 LATENT=8 BETAS="0.1 1 4" HIDDEN=512 LAYERS=3 TARGET=next_state] DATASET=<walker/rnd/dataset.npz> bash tasks/T4.1.2.2-conditioned-vae/run_cvae.sh
# Per FB training seed (the three T06 checkpoints, as in T4.1.2):
#   - test pairs: episodes 0-19 in the stock environment replayed one step in the stock and the
#     eight shifted dynamics -> out/paired_raw_s$SEED.npz
#   - training rollouts: stock episodes 100-119 -> out/train_rollouts_s$SEED.npz
# Then one CVAE ensemble on the pooled training rollouts of the three seeds, and the scores of
# the pooled test pairs (the inputs of the model are standardised with the statistics of the full
# ExORL dataset) -> out/cvae_beta$BETA.pt, out/cvae_scored_beta$BETA.npz for each beta in BETAS.
set -eu

CKPT_DIR=${CKPT_DIR:?}
DATASET=${DATASET:?}
MEMBERS=${MEMBERS:-5}   # ensemble size N
LATENT=${LATENT:-8}     # dimension of z
BETAS=${BETAS:-"0.1 1 4"}   # weights of the KL term, one model per value
HIDDEN=${HIDDEN:-512}   # width and number of hidden layers of encoder and decoder
LAYERS=${LAYERS:-3}
TARGET=${TARGET:-next_state}   # next_state (as in the paper) or delta
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

for SEED in 0 1 2; do
  CKPT=$CKPT_DIR/${CKPTS[$SEED]}.pt
  ZS=tasks/T06-latent-switching/data/inputs/zs_cql_sweep_zero-shot-rl-${CQL_SWEEPS[$SEED]}.npz
  [ -f "out/paired_raw_s$SEED.npz" ] || uv run python -m zero_shot_rl.eval.paired_step --checkpoint "$CKPT" --zs "$ZS" \
    --dynamics "${DYN[@]}" --episodes 20 --first_episode 0 --out "out/paired_raw_s$SEED.npz"
  [ -f "out/train_rollouts_s$SEED.npz" ] || uv run python -m zero_shot_rl.eval.paired_step --checkpoint "$CKPT" --zs "$ZS" \
    --dynamics "$CFG/baseline.yaml" --episodes 20 --first_episode 100 --out "out/train_rollouts_s$SEED.npz"
done

for BETA in $BETAS; do
  uv run python -m zero_shot_rl.latent.cvae_model \
    --rollouts out/train_rollouts_s0.npz out/train_rollouts_s1.npz out/train_rollouts_s2.npz \
    --stats_dataset "$DATASET" --val_episodes 4 --members "$MEMBERS" --latent "$LATENT" --beta "$BETA" \
    --hidden "$HIDDEN" --layers "$LAYERS" --target "$TARGET" --epochs 1000 --seed 0 --out "out/cvae_beta$BETA.pt" \
    2>&1 | tee "out/cvae_beta$BETA.log"
  uv run python -m zero_shot_rl.latent.cvae_score --model "out/cvae_beta$BETA.pt" \
    --paired out/paired_raw_s0.npz out/paired_raw_s1.npz out/paired_raw_s2.npz --out "out/cvae_scored_beta$BETA.npz"
done
