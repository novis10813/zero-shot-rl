# T5.1 FB-AA Improvement

Action Transformation ($g_0, g_1, T_\theta$) without updating the FB backbone.

## Reproduce

1. Train $g_0(s, a) \to B_0(s')$ on stock FB trajectories (`zero_shot_rl.latent.action_transform`):

```bash
PYTHONPATH=. uv run python -m zero_shot_rl.latent.action_transform \
    --stock <fb_trajectories_seed0.npz> <fb_trajectories_seed1.npz> <fb_trajectories_seed2.npz> \
    --checkpoint <fb.pt> --seed <0|1|2> --out-dir <g0_dir>
```

2. Fine-tune $g_1$ and train $T_\theta$ on disjoint finetune seeds (`320000+`), then evaluate on evaluation seeds (`300000+`) (`zero_shot_rl.eval.eval_action_transform`):

```bash
PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.eval.eval_action_transform \
    --checkpoint <fb.pt> --latents <zs_fb.npz> --g0 <g0_dir>/g0_best.pt \
    --dynamics tasks/T4.1.3.2-fb-aa/configs/test_dynamics.yaml \
    --explore-std 0.2 --lam 0.1 --seed <0|1|2> --out <eval.npz>
```

3. Regenerate and check `data/table6_iqm.csv`:

```bash
PYTHONPATH=.:tasks/T5.1-fb-aa-improvement uv run python -m analyze.summarize
PYTHONPATH=.:tasks/T5.1-fb-aa-improvement uv run python -m analyze.summarize --check
```
