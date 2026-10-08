# T01: reproducing official FB on ExORL Walker RND-100k

How to reproduce the results in [`REPORT.md`](REPORT.md): five seeds of the authors' FB
(`main_exorl.py fb`) on the ExORL `walker` / `rnd` dataset subsampled to 100k transitions,
evaluated zero-shot on `stand`, `walk`, `run` and `flip`. The authors publish no checkpoints,
so the models are trained here.

All commands run from the repository root. `lab run` executes the pushed commit, and lab runs
`uv sync` (Python 3.9, pinned packages from `pyproject.toml`) before every job.

## 1. Dataset

Download the ExORL zip (2.6 GB) and merge it into one `.npz` outside the job's worktree, so
later jobs can reuse it. Needed once per host.

```bash
lab run --vram 0 -- 'D=$HOME/zsrl-datasets/walker/rnd; \
  uv run python -u exorl_reformatter.py walker_rnd && mkdir -p $D && \
  mv datasets/walker/rnd/dataset.npz $D/'
```

The 100k training transitions are drawn with a fixed `rng(42)` regardless of `--seed`, so
every seed trains on the same subsample.

## 2. Training

One job per seed, 1M steps. Every 20k steps the job logs a line
`Step <i> eval: eval/<task>/episode_reward_iqm=..., eval/task_reward_iqm=...`. At the end it
keeps the best model by mean IQM and the final model in
`agents/fb/saved_models/local-run-*/<step>.pt`. The T01 runs predate the final-model
change and kept only the best model (see section 5).

```bash
lab sweep tasks/T01-fb-reproduction/configs/train_seeds.yaml
```

or for a single seed:

```bash
lab run --vram 10G -- 'PYTHONUNBUFFERED=1 uv run python main_exorl.py fb walker rnd \
  --eval_tasks stand walk run flip --wandb_logging False \
  --dataset_path $HOME/zsrl-datasets/walker/rnd/dataset.npz --seed 0'
```

- Peak VRAM is about 8.4 GB, mostly from loading all 10M transitions onto the GPU before
  subsampling. It is reached after the job starts, so the lab dispatcher can place too many
  jobs on one GPU and they fail with CUDA OOM. Keep at most two jobs per 32 GB GPU.
- Use GPUs that other jobs are not saturating. Alone on an RTX 5090 a run takes about
  2.2 hours (125 it/s), two on one GPU about 3 to 5 hours each. On a GPU kept at 100% by
  other users' jobs the same run fell to about 7 it/s.

## 3. Eval data

Each `data/<job>_eval.csv` holds one run's `Step <i> eval` lines as
`step,stand,walk,run,flip,mean`. Extract it from the job log with:

```bash
lab logs -n 100000 <job> | tr '\r' '\n' | grep -E 'Step [0-9]+ eval' | python3 -c "
import re, sys
print('step,stand,walk,run,flip,mean')
for line in sys.stdin:
    step = re.search(r'Step (\d+) eval', line).group(1)
    v = dict(re.findall(r'eval/(\w+?)(?:/episode_reward_iqm|_reward_iqm)=([\d.\-]+)', line))
    print(f\"{step},{v['stand']},{v['walk']},{v['run']},{v['flip']},{v['task']}\")
" > tasks/T01-fb-reproduction/data/<job>_eval.csv
```

| Seed | Data |
|---|---|
| 42 | `data/zsrl-5_eval.csv` |
| 0 | `data/zsrl-6_eval.csv` |
| 1 | `data/zsrl-7_eval.csv` |
| 2 | `data/zsrl-12_eval.csv` |
| 3 | `data/zsrl-13_eval.csv` |

## 4. Report numbers and figures

Table 1 and both figures of the report. The step with the highest all-task IQM over the
five seeds is selected (as in the paper's Table 6), and each task's IQM over seeds is
reported with a 95% stratified-bootstrap interval (`rliable`, 10,000 resamples, seed 0).
Keep the file order below: the bootstrap intervals depend on it. The script prints, for the
selected and the final step, each task's IQM and interval and the mean of the 4 task IQMs
(the report's "4 任務平均" column), then each run's own best-step mean (the "397 到 488"
range). The paper's numbers and the fb-offline comparison are quoted from their sources.

```bash
T=tasks/T01-fb-reproduction
uv run python $T/aggregate_seeds.py $T/data/zsrl-5_eval.csv $T/data/zsrl-6_eval.csv \
  $T/data/zsrl-7_eval.csv $T/data/zsrl-12_eval.csv $T/data/zsrl-13_eval.csv \
  -o $T/figures/fb_walker_rnd_5seeds.png --comparison_output $T/figures/fb_vs_paper.png
```

## 5. Re-evaluating a checkpoint

The archived T01 checkpoints are each run's own best step, not the 900k and 1M steps of
Table 1: seed 42 at 740k, seed 0 at 800k, seed 1 at 680k, seed 2 at 200k, seed 3 at 560k.
The runs saved them as pickled agents. `convert_checkpoints.py` rewrote them in the
state_dict format of `AbstractAgent.save` and checked that every tensor and the actions on
100 random inputs are identical:

```bash
uv run python -m tasks.T01-fb-reproduction.convert_checkpoints <step>.pickle ...
```

Same protocol as the in-training evaluation: z inferred from 10k buffer transitions
relabelled with each task's reward, 10 deterministic rollouts of 1000 steps per task, IQM.
Scores differ slightly from the in-training ones, because the z-inference sample and the
environment's random state differ.

```bash
uv run python -m scripts.eval_exorl <checkpoint.pt> walker \
  --dataset_path $HOME/zsrl-datasets/walker/rnd/dataset.npz \
  --eval_tasks stand walk run flip --eval_rollouts 10
```

## Changes to the authors' code

The FB losses, the agent and the evaluation are unchanged, so training and evaluation
results are those of the original code. The two `workspaces.py` changes only affect what is
logged and kept on disk.

| Change | Why |
|---|---|
| `pyproject.toml` + `uv.lock`: Python 3.9, torch 2.7.1 (CUDA 12.8), `setuptools<70`. Other pins as in `requirements.txt`, minus the unused `gcloud` and `pylint` | torch 2.1.0 does not support Blackwell GPUs (sm_120). wandb 0.15.2 imports `pkg_resources` |
| `main_exorl.py --dataset_path` | read the dataset from outside the job's worktree |
| `agents/workspaces.py`: keep the best and the final checkpoint after training | the original deletes the best one unless it was uploaded to wandb, and never saves the final one |
| `agents/workspaces.py`: log every evaluation's per-task IQM | without wandb, per-task scores were not recorded |
| `agents/base.py`: checkpoints store class, constructor arguments and `state_dict` (`.pt`); `load_agent()` rebuilds the agent with `weights_only=True` | pickled whole agents break when the class changes and can run arbitrary code on load |
| `scripts/eval_exorl.py` | re-evaluate a saved checkpoint |
