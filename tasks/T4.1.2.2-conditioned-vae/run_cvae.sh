#!/bin/bash
# CVAE ensemble detection of T4.1.2.2 (same data and split as T4.1.2.1).
#
# Usage (repo root):
#     CKPT_DIR=<dir with zsrl-32.pt zsrl-41.pt zsrl-42.pt> [MEMBERS=5 LATENT=8 BETAS="0.1 1 4" LEARN_SIGMA=0 HIDDEN=512 LAYERS=3 TARGET=next_state] DATASET=<walker/rnd/dataset.npz> bash tasks/T4.1.2.2-conditioned-vae/run_cvae.sh
# Per FB training seed (the three T06 checkpoints, as in T4.1.2):
#   - test pairs: episodes 0-19 in the stock environment replayed one step in the stock and the
#     eight shifted dynamics -> out/paired_raw_s$SEED.npz
#   - training rollouts: stock episodes 100-119 -> out/train_rollouts_s$SEED.npz
# The seeds are collected in parallel, and the models of the values of beta are trained in parallel on one GPU (log: out/cvae_beta$BETA.log, out/cvae_learn_sigma.log).
# Then one CVAE ensemble on the pooled training rollouts of the three seeds, and the scores of
# the pooled test pairs (the inputs of the model are standardised with the statistics of the full
# ExORL dataset) -> out/cvae_beta$BETA.pt, out/cvae_scored_beta$BETA.npz for each beta in BETAS (and out/cvae_learn_sigma.* if LEARN_SIGMA=1).
set -euo pipefail

CKPT_DIR=${CKPT_DIR:?}
DATASET=${DATASET:?}
MEMBERS=${MEMBERS:-5}   # ensemble size N
LATENT=${LATENT:-8}     # dimension of z
BETAS=${BETAS:-"0.1 1 4"}   # weights of the KL term, one model per value
LEARN_SIGMA=${LEARN_SIGMA:-0}   # 1: one more model with a learned decoder std and beta 1 (out/cvae_learn_sigma.*)
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

# The three seeds, and then the values of beta, run as parallel processes; `wait` on each pid fails the script if one failed.
wait_all() { for pid in "$@"; do wait "$pid"; done; }

collect() {
  SEED=$1
  CKPT=$CKPT_DIR/${CKPTS[$SEED]}.pt
  ZS=tasks/T06-latent-switching/data/inputs/zs_cql_sweep_zero-shot-rl-${CQL_SWEEPS[$SEED]}.npz
  [ -f "out/paired_raw_s$SEED.npz" ] || uv run python -m zero_shot_rl.eval.paired_step --checkpoint "$CKPT" --zs "$ZS" \
    --dynamics "${DYN[@]}" --episodes 20 --first_episode 0 --out "out/paired_raw_s$SEED.npz"
  [ -f "out/train_rollouts_s$SEED.npz" ] || uv run python -m zero_shot_rl.eval.paired_step --checkpoint "$CKPT" --zs "$ZS" \
    --dynamics "$CFG/baseline.yaml" --episodes 20 --first_episode 100 --out "out/train_rollouts_s$SEED.npz"
}

train_and_score() {
  BETA=$1
  NAME=beta$BETA
  EXTRA=()
  if [ "$BETA" = learn_sigma ]; then BETA=1; NAME=learn_sigma; EXTRA=(--learn_sigma); fi
  uv run python -m zero_shot_rl.latent.cvae_model \
    --rollouts out/train_rollouts_s0.npz out/train_rollouts_s1.npz out/train_rollouts_s2.npz \
    --stats_dataset "$DATASET" --val_episodes 4 --members "$MEMBERS" --latent "$LATENT" --beta "$BETA" "${EXTRA[@]}" \
    --hidden "$HIDDEN" --layers "$LAYERS" --target "$TARGET" --epochs 1000 --seed 0 --out "out/cvae_$NAME.pt" \
    > "out/cvae_$NAME.log" 2>&1
  uv run python -m zero_shot_rl.latent.cvae_score --model "out/cvae_$NAME.pt" \
    --paired out/paired_raw_s0.npz out/paired_raw_s1.npz out/paired_raw_s2.npz --out "out/cvae_scored_$NAME.npz" \
    >> "out/cvae_$NAME.log" 2>&1
}

PIDS=()
for SEED in 0 1 2; do collect "$SEED" & PIDS+=($!); done
wait_all "${PIDS[@]}"

PIDS=()
for BETA in $BETAS $([ "$LEARN_SIGMA" = 1 ] && echo learn_sigma); do train_and_score "$BETA" & PIDS+=($!); done
wait_all "${PIDS[@]}"
tail -n 4 out/cvae_*.log
