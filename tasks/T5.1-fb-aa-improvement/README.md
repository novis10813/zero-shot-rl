# T5.1 FB-AA Improvement

Action Transformation ($g_0, g_1, T_\theta$) without updating the FB backbone.

## Reproduce

### 1. 主實驗：`660000.pt`（10 finetune seeds × 20 共用環境 `5..24`）

1. Train $g_0(s, a) \to B_0(s')$ on stock FB trajectories (`zero_shot_rl.latent.action_transform`):

```bash
PYTHONPATH=. uv run python -m zero_shot_rl.latent.action_transform \
    --stock <fb_trajectories_seed0.npz> <fb_trajectories_seed1.npz> <fb_trajectories_seed2.npz> \
    --checkpoint <660000.pt> --seed 0 --out-dir out/g0_660k
```

2. Run the 40-job sweep on `idlab` (`tasks/T5.1-fb-aa-improvement/configs/sweep_660k.yaml`):

```bash
lab sweep tasks/T5.1-fb-aa-improvement/configs/sweep_660k.yaml
```

### 2. T4.1.3.2 對照組（T05 3 FB seeds，5 episodes / task）

1. Train $g_0$ per seed and run `eval_action_transform` with `--episodes 5 --first-env-seed 300000 --finetune-episodes 5 --finetune-seed 320000`:

```bash
PYTHONPATH=. uv run python -m zero_shot_rl.latent.action_transform \
    --stock <fb_trajectories_seed0.npz> <fb_trajectories_seed1.npz> <fb_trajectories_seed2.npz> \
    --checkpoint <fb.pt> --seed <0|1|2> --out-dir <g0_dir>

PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.eval_action_transform \
    --checkpoint <fb.pt> --latents <zs_fb.npz> --g0 <g0_dir>/g0_best.pt \
    --dynamics tasks/T4.1.3.2-fb-aa/configs/test_dynamics.yaml \
    --episodes 5 --first-env-seed 300000 --finetune-episodes 5 --finetune-seed 320000 \
    --explore-std 0.2 --lam 0.1 --seed <0|1|2> --out <eval.npz>
```

2. Run the T4.1.3.2 FB-AA baseline on the friction configs (`zero_shot_rl.eval.adapt_episode` via `tasks/T4.1.3.2-fb-aa/run_adapt_episode.sh`):

```bash
CKPT_DIR=<ckpt_dir> MODEL_DIR=<context_model_dir> \
bash tasks/T4.1.3.2-fb-aa/run_adapt_episode.sh \
    --configs baseline friction_mild friction_strong friction_high_mild friction_high_strong
```

### 3. Regenerate and check CSVs

```bash
PYTHONPATH=.:tasks/T5.1-fb-aa-improvement uv run python -m analyze.summarize
PYTHONPATH=.:tasks/T5.1-fb-aa-improvement uv run python -m analyze.summarize --check
```
