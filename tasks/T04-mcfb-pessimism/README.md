# T04: is MC-FB more pessimistic than VC-FB?

How to reproduce the results in [`REPORT.md`](REPORT.md): FB, VC-FB, MC-FB and DVC-FB, three
seeds each, on ExORL `walker` / `rnd` subsampled to 100k transitions, then a probe that measures
how each final model values out-of-distribution actions against Monte Carlo returns.

All commands run from the repository root. The dataset is the one built in
[`tasks/T01-fb-reproduction/README.md`](../T01-fb-reproduction/README.md) section 1.

## 1. Code change: DVC-FB

`main_exorl.py dvcfb` trains Directed VC-FB (paper appendix B.1.6, eq. 19). It is
`agents/cfb/agent.py` with `dvcfb=True`: the VC-FB value penalty with every z replaced by
$$z = \sqrt{d}\,\frac{B(s_g)}{\lVert B(s_g)\rVert},\quad s_g \sim \mathcal{D}$$
where $s_g$ is a shuffled batch of next states. $z$ is computed without gradient, so the penalty
trains F and the actor but not B, as for the sampled zs of VC-FB. The FB loss is unchanged.
`dvcfb` defaults to `False`, and `vcfb` / `mcfb` behave as before.

## 2. Training

One job per algorithm and seed, 1M steps, paper hyperparameters (`agents/*/config.yaml`).
[`configs/train.yaml`](configs/train.yaml) is the grid as a `lab sweep` file. The jobs were
not launched as that sweep: each was submitted on its own with `lab run` (local GPUs) or
`dicos run` (A100 on ASGC DiCOS), because only two local GPUs were available. The command is
the same everywhere:

```bash
PYTHONUNBUFFERED=1 uv run python main_exorl.py <fb|vcfb|mcfb|dvcfb> walker rnd \
  --eval_tasks stand walk run flip --wandb_logging False \
  --dataset_path <dir>/walker/rnd/dataset.npz --seed <0|1|2>
```

Each job logs `Step <i> eval` lines every 20k steps and keeps the best model (highest mean
IQM over the four tasks) and the final model in `agents/<fb|cfb>/saved_models/local-run-*/`.

| Algorithm | Seed | Job | Where | Commit |
|---|---|---|---|---|
| FB | 0 | `zsrl-32` | lab | 30078fc |
| FB | 1 | `zsrl-41` | lab | 9df1ee9 |
| FB | 2 | `zsrl-42` | lab | 9df1ee9 |
| VC-FB | 0 | `zero-shot-rl-27` | DiCOS A100 | 9df1ee9 |
| VC-FB | 1 | `zsrl-27` | lab | 30078fc |
| VC-FB | 2 | `zero-shot-rl-18` | DiCOS A100 | 9df1ee9 |
| MC-FB | 0 | `lab-zero-shot-rl-3` | lab | 0e8075a |
| MC-FB | 1 | `zero-shot-rl-28` | DiCOS A100 | 9df1ee9 |
| MC-FB | 2 | `zero-shot-rl-29` | DiCOS A100 | 9df1ee9 |
| DVC-FB | 0 | `zsrl-43` | lab | 9df1ee9 |
| DVC-FB | 1 | `lab-zero-shot-rl-2` | lab | 9daea8c |
| DVC-FB | 2 | `zero-shot-rl-32` | DiCOS A100 | 9df1ee9 |

- Each job's checkpoints, log and seed are recorded in the tag `exp/T04-<job>`.
- Plain `zero-shot-rl-<n>` ids are `dicos` jobs. lab 0.3 started numbering its own jobs
  `zero-shot-rl-<n>` too, so those carry a `lab-` prefix in tags and artifact folders.
- The DiCOS jobs ran with `dicos run --sync -p a100-al9 --gpus 1 --cpus 4 --mem 48G -t 30h`
  from a clean tree at 9df1ee9. At about 20 it/s a run takes about 14 hours.
- Between 30078fc and 0e8075a the only change to training code is the DVC-FB branch above,
  so FB, VC-FB and MC-FB train the same at every listed commit.
- `zero-shot-rl-20` is an accidental second MC-FB seed 1 run. Its checkpoints are
  bit-identical to `zero-shot-rl-28` (same hardware), so it is not used.

## 3. Eval data

`data/eval_<algo>_s<seed>.csv` holds a run's `Step <i> eval` lines as
`step,stand,walk,run,flip,mean` (IQM over eval episodes per task, `mean` over tasks):

```bash
python tasks/T04-mcfb-pessimism/extract_eval.py <job output.log> \
  tasks/T04-mcfb-pessimism/data/eval_<algo>_s<seed>.csv
```

## 4. Pessimism probe

[`pessimism_probe.py`](pessimism_probe.py) loads one final (1M-step) checkpoint and writes
`data/probe_<algo>_s<seed>.json`. It needs a GPU and MuJoCo and takes about 7 to 20 minutes.

```bash
PYTHONPATH=. uv run python tasks/T04-mcfb-pessimism/pessimism_probe.py \
  --dataset_path <dir>/walker/rnd/dataset.npz \
  --out tasks/T04-mcfb-pessimism/data/probe_<algo>_s<seed>.json \
  --checkpoints <algo>_s<seed>=<run>/1000000.pt
```

For each eval task it infers $z_r$ from the dataset rewards and rescales the critic to reward
units,
$$\hat Q(s,a) = c_r \min(F_1,F_2)(s,a,z_r)^\top z_r,\qquad c_r = \frac{\lVert \mathbb{E}_{\mathcal{D}}[r\,B]\rVert}{\sqrt d},$$
then compares it with a Monte Carlo return $G(s,a)$: reset the simulator to $s$, take $a$,
and follow the actor at $z_r$ with the training target noise for 300 steps
($\gamma = 0.98$). Restoring a dataset state reproduces its observation to within $2\times10^{-6}$.
The 1000 training states come from probe seed 0 and are the same for every model only on the
same machine: another host drew a different set from the same seed. All reported probes ran
on one GPU, so every model is scored on the same states. Four action sources: dataset action, uniform action (`ood`), policy action, and policy
action on states the eval policy visits. It also records $Q(s,a_\text{data})$ against
$Q(s,a_\text{uniform})$ for $z$ from the prior, from $B(s)$ and $z_r$.

## 5. Report numbers and figure

All tables in the report and `figures/pessimism.png` come from `data/`:

```bash
python tasks/T04-mcfb-pessimism/summarize.py --fig tasks/T04-mcfb-pessimism/figures/pessimism.png
```
