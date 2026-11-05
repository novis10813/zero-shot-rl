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
## Distance of first-step actions to the dataset behaviour

Same checkpoints. Env seeds 0 to 49 give 50 initial states; each checkpoint's action at each
state and task is scored by `zero_shot_rl.eval.behavior_distance` (definitions in its docstring;
`--lam 1`, `--k 50`). FB seed k is paired with VC-FB seed k in the figures.

```bash
A=artifacts/T04-mcfb-pessimism
MUJOCO_GL=egl PYTHONPATH=. uv run python -m zero_shot_rl.eval.behavior_distance \
  --checkpoints FB $A/zsrl-32/800000.pt $A/zsrl-41/680000.pt $A/zsrl-42/200000.pt \
  --checkpoints VC-FB $A/zero-shot-rl-27/440000.pt $A/zsrl-27/420000.pt $A/zero-shot-rl-18/500000.pt \
  --dataset_path datasets/walker/rnd/dataset.npz \
  --out tasks/T3.3.1-action-evaluation/data/behavior_distance.csv \
  --figure_prefix tasks/T3.3.1-action-evaluation/figures/behavior_distance
```

## Distances along an episode

`zero_shot_rl.eval.behavior_rollout` runs one deterministic 1000-step episode per checkpoint and
task from env seed 0, keeps the state every 10 steps, and scores the action of every checkpoint
at those states (same distances as above). Scatter plots pair FB seed k with VC-FB seed k at the
states visited by the FB (or VC-FB) seed k episode.

```bash
A=artifacts/T04-mcfb-pessimism
MUJOCO_GL=egl PYTHONPATH=. uv run python -m zero_shot_rl.eval.behavior_rollout --episodes 1 \
  --checkpoints FB $A/zsrl-32/800000.pt $A/zsrl-41/680000.pt $A/zsrl-42/200000.pt \
  --checkpoints VC-FB $A/zero-shot-rl-27/440000.pt $A/zsrl-27/420000.pt $A/zero-shot-rl-18/500000.pt \
  --dataset_path datasets/walker/rnd/dataset.npz \
  --out tasks/T3.3.1-action-evaluation/data/rollout_distances.csv \
  --figure_prefix tasks/T3.3.1-action-evaluation/figures/rollout
```

