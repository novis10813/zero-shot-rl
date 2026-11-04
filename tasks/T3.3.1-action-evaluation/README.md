# T3.3.1: action evaluation

## First-state action of FB and VC-FB (Walker, RND-100k)

Per-seed best checkpoints of the T04 runs, one run per seed. Every checkpoint acts at the reset
state of environment seed 0; z is inferred per task from the same 100k-transition subsample.

```bash
A=artifacts/T04-mcfb-pessimism
MUJOCO_GL=egl PYTHONPATH=. uv run python -m zero_shot_rl.eval.first_action \
  --checkpoints FB $A/zsrl-32/800000.pt $A/zsrl-41/680000.pt $A/zsrl-42/200000.pt \
  --checkpoints VC-FB $A/zero-shot-rl-27/440000.pt $A/zsrl-27/420000.pt $A/zero-shot-rl-18/500000.pt \
  --dataset_path datasets/walker/rnd/dataset.npz \
  --out tasks/T3.3.1-action-evaluation/data/first_action.csv \
  --figure tasks/T3.3.1-action-evaluation/figures/first_action.png
```
