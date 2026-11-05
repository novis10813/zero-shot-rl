# T3.3.1: action evaluation

Every command below runs from the repo root and uses the per-seed best checkpoints of the T04
runs, one run per seed (FB seed k is paired with VC-FB seed k), on Walker RND-100k. z is
inferred per task from the same 100k-transition subsample (`zero_shot_rl.latent.task_z`).
Entries in `zero_shot_rl/latent/` only query the actor at given states; entries in
`zero_shot_rl/eval/` roll out episodes. They write the CSVs in `data/`; every figure in
`figures/` is drawn from those CSVs by `plot.py` (see the last section).

```bash
A=artifacts/T04-mcfb-pessimism
CK="--checkpoints FB $A/zsrl-32/800000.pt $A/zsrl-41/680000.pt $A/zsrl-42/200000.pt \
  --checkpoints VC-FB $A/zero-shot-rl-27/440000.pt $A/zsrl-27/420000.pt $A/zero-shot-rl-18/500000.pt \
  --dataset_path datasets/walker/rnd/dataset.npz"
T=tasks/T3.3.1-action-evaluation
```

## First-state action of FB and VC-FB

Every checkpoint acts at the reset state of environment seed 0.

```bash
MUJOCO_GL=egl PYTHONPATH=. uv run python -m zero_shot_rl.latent.first_action $CK \
  --out $T/data/first_action.csv
```

## Distance of first-step actions to the dataset behaviour

Env seeds 0 to 49 give 50 initial states; each checkpoint's action at each state and task is
scored by the distances of `zero_shot_rl.latent.behavior` (definitions in its docstring;
`--lam 1`, `--k 50`).

```bash
MUJOCO_GL=egl PYTHONPATH=. uv run python -m zero_shot_rl.latent.behavior_distance $CK \
  --out $T/data/behavior_distance.csv
```

## Distances along an episode

`zero_shot_rl.eval.behavior_rollout` runs one deterministic 1000-step episode per checkpoint and
task from env seed 0, keeps the state every 10 steps, and scores the action of every checkpoint
at those states (same distances as above).

```bash
MUJOCO_GL=egl PYTHONPATH=. uv run python -m zero_shot_rl.eval.behavior_rollout --episodes 1 $CK \
  --out $T/data/rollout_distances.csv
```

## Return difference against distance difference

`zero_shot_rl.eval.episode_returns` runs one episode from each of the 50 initial states per
checkpoint and task. The figure pairs each return with the distance of `behavior_distance.csv`
at the same initial state.

```bash
MUJOCO_GL=egl PYTHONPATH=. uv run python -m zero_shot_rl.eval.episode_returns $CK \
  --out $T/data/episode_returns.csv
```

## Reward of the action against its distance

`zero_shot_rl.eval.behavior_reward` runs one episode per checkpoint and task, keeps the
physics state every 10 steps, re-executes each checkpoint's action from that exact state and
sums the task reward over `--horizon` steps (1: the reward after taking the action). All
episodes reuse one environment seeded with 0, so only the first episode starts from the reset
state used above.

```bash
MUJOCO_GL=egl PYTHONPATH=. uv run python -m zero_shot_rl.eval.behavior_reward --horizon 1 $CK \
  --out $T/data/reward_h1.csv
```

## Action difference at dataset states

`zero_shot_rl.latent.action_difference`: 200 states sampled from the 100k-transition training
subsample; for each task and seed pair k, VC-FB seed k action minus FB seed k action
(deterministic actions, each agent with its own z).

```bash
MUJOCO_GL=egl PYTHONPATH=. uv run python -m zero_shot_rl.latent.action_difference $CK \
  --out $T/data/action_difference.csv
```

## Figures

`plot.py` draws every figure in `figures/` from `data/` (one figure per file, listed in its
docstring) and prints the Spearman correlations of the difference plots.

```bash
uv run python $T/plot.py
```

## Episode videos with actions and rewards

`zero_shot_rl.eval.record_episode` records one episode per checkpoint and task from env seed 0:
per-step actions and rewards in `data/episode_actions.npz`, one mp4 per run under `artifacts/`
(videos are not committed; only `FB_seed1_walk.mp4` and `VC-FB_seed1_walk.mp4` were kept).
`compose_video.py` puts VC-FB (left) and FB (right) side by side: video, stacked actuator
traces, per-step reward.

```bash
V=artifacts/T3.3.1-action-evaluation/videos
MUJOCO_GL=egl PYTHONPATH=. uv run python -m zero_shot_rl.eval.record_episode $CK \
  --out $T/data/episode_actions.npz --video_dir $V
uv run python $T/compose_video.py --video_dir $V \
  --task walk --seed 1 --window 100 --out $V/walk_seed1_w100.mp4
```
