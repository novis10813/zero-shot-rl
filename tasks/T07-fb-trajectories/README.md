# T07 FB trajectories

Observation, action and reward of the three T05 standard FB models (training seeds 0, 1, 2) on
walker stand, walk, run and flip, with the latent inferred from the unmodified task reward
(`z_r`). 50 deterministic 1000-step episodes per task, environment seeds 0 to 49.

```
bash tasks/T07-fb-trajectories/run.sh
```

Runs `zero_shot_rl.eval.fb_trajectories` with `configs/sources.json` (checkpoints and `z_r`
latents are those of T05). Episodes run in batches of 20, so episodes 0 to 19 reproduce the
rewards stored by T05 exactly; the manifest records the largest difference. A different batch
size changes the floating point results and the trajectories diverge within tens of steps.

Output per model, `fb_trajectories_seed<k>.npz` with `observations [4, 50, 1000, 24]`,
`actions [4, 50, 1000, 6]`, `rewards [4, 50, 1000]`, `z`, `tasks`, `episodes`, `dt`, and a JSON
manifest. Task order is stand, walk, run, flip. Observation `t` is the one before action `t`.
