# AGENTS.md

Baselines, modified RL functions and FB latent experiments on top of the authors' zero-shot RL
code, one report per task.

The experiment process (task ids, branches, worktrees, lifecycle, archiving and `exp/` tags,
`REPORT.md` structure, file rule, close-out, review, compute and data rules) is defined in
`/mnt/raid1/novis/research/docs/experiment-protocol.md`. Read it before any task work. This file
adds only what is specific to this repo.

## Where code lives

```
agents/ rewards/ custom_dmc_tasks/       upstream code
configs/                                 shared model/domain configs (e.g. configs/anfis/<domain>.yaml)
download.sh                              upstream dataset download, called by exorl_reformatter
scripts/                                 top-level launch scripts (train.sh, finetune.sh, evaluate.sh)
zero_shot_rl/                            shared code: upstream root modules and ours
  cli.py                                 subcommand dispatcher for train / finetune / eval
  train/                                 offline pre-training (main_exorl, main_d4rl are upstream;
                                         ensemble, g0, context_encoder, dynamics_model, cvae_model)
  finetune/                              test-time adaptation (value_correction, action_transform,
                                         online_inference)
  data/                                  dataset preparation (exorl_reformatter is upstream)
  env/                                   environments and the actors that act in them
  latent/                                FB latents, shared model definitions and geometry
  eval/                                  pure evaluation, scoring and rollouts in the simulator
  online/                                online belief, FB wrappers and value-correction models
  utils.py                               seeding and BASE_DIR (upstream)
  provenance.py                          file hashes and no-clobber publication
analysis/                                analysis tools used by two or more tasks
tests/                                   component tests for zero_shot_rl/ and agents/
tasks/T<id>-<name>/
  REPORT.md                              question, method, conclusions, results (see the protocol)
  README.md                              how to reproduce: commands, configs, code changes
  configs/                               experiment configs (sources, sweeps, seeds)
  data/                                  eval data behind every reported number
    inputs/                              stored latents, rollouts and traces the analysis reads
  figures/                               figures used in REPORT.md
  run*.sh                                launch scripts for the task's experiments
  analyze/, *.py                         analysis and visualization used only by this task
  tests/                                 tests of this task's analysis
```

Each module's docstring states what it does and how to run it.

## Running

Run everything from the repo root with `uv`, e.g.
`uv run python -m zero_shot_rl.train.main_exorl fb walker rnd` (reads `agents/<algo>/config.yaml`)
or `PYTHONPATH=. uv run python -m zero_shot_rl.latent.infer`. Task modules also need the task
directory on the path, e.g. `PYTHONPATH=.:tasks/T05-opt-to-pessi uv run python -m analyze.q1`.

## Code rules

- The authors' code is the `upstream` remote (`enjeeneer/zero-shot-rl`, branch `upstream/main`).
  Their root modules `main_exorl.py`, `main_d4rl.py`, `exorl_reformatter.py`, `dmc.py` and
  `utils.py` live in `zero_shot_rl/`. Bringing their updates into `main` is a `chore/` PR that
  merges `upstream/main`, applying changes to those modules to their `zero_shot_rl/` copies.
- New baselines go in new modules (for example `agents/<name>/`), not in edits to existing agents.
- Stricter than the protocol, for upstream code only: a change to an upstream file, including the
  modules moved to `zero_shot_rl/`, that can alter training or evaluation results goes behind a
  new argument or config key whose default keeps the original behaviour.
- For other existing entry modules, earlier results are reproduced from their `exp/` tags. A
  change that alters what such a module computes by default is stated in the commit message and
  the PR.
- For tasks after T06, code in `tasks/T<id>-<name>/` is limited to experiment configs (yaml),
  launch scripts (`.sh`) and analysis or visualization used only by that task. Any other code
  (algorithms, environments, entry modules, libraries) goes in the shared directories, e.g.
  `zero_shot_rl/`.
- A task's analysis code moves to `analysis/` or `zero_shot_rl/` when a second task needs it, and
  the same commit updates the commands in earlier tasks' READMEs.

## Verify

If only files under `tasks/T<id>-<name>/` changed:

```
uv run ruff check
PYTHONPATH=.:tasks/T<id>-<name> MUJOCO_GL=egl uv run python -m unittest discover -s tasks/T<id>-<name>/tests -t tasks/T<id>-<name>
```

If shared code changed (`zero_shot_rl/`, `agents/`, `analysis/`, `rewards/`, `custom_dmc_tasks/`,
`configs/`, `scripts/`, `tests/`), run the full block before every commit:

```
uv run ruff check
PYTHONPATH=. uv run python -m unittest discover -s tests
for d in tasks/*/tests; do
  t=${d%/tests}
  PYTHONPATH=.:$t MUJOCO_GL=egl uv run python -m unittest discover -s $d -t $t || break
done
```

Each task's tests run in their own process, because task modules such as `common.py` may share
names across tasks.

## Research rules

Scores of trained agents over seeds are aggregated with `analysis.table6_aggregate` (Table 6 of
Jeen et al. 2024). Tasks up to T3.3.1 that used other aggregations keep them.

## Data rules

- Checkpoints are saved with `AbstractAgent.save` (class, constructor arguments and
  `state_dict`) and loaded with `load_agent`. Never pickle whole objects.
- `.gitignore` covers `artifacts/`, `datasets/` and `agents/*/saved_models/`.
