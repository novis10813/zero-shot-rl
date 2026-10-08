# AGENTS.md

The repo will hold several baselines, modified RL functions and
one report per task. These rules keep `main` clean while experiments stay reproducible.

## Branches

| Branch | Purpose | Rules |
|---|---|---|
| `task/T<NN>-<name>` | one per task; experiments and trial and error | Anything goes, but never rebase, force-push or delete it (a GitHub ruleset blocks both) |
| `clean/T<NN>-<name>` | branched from `task/T<NN>-<name>` when the task ends; holds only what passes the file rule | Opened as a PR to `task/T<NN>-<name>`. Deleted once that PR is merged |
| `chore/<name>` | changes outside any task: refactors, documentation, tooling, these rules | Short-lived. Opened as a PR to `main` and deleted once merged. A refactor must not change training or evaluation results |
| `main` | runnable code and every finished task | Every new task branches from it. Receives a task only by merging its `task/` branch after the `clean/` PR |

Task numbers are two digits and never reused: `T01`, `T02`, ...

Tasks run one at a time: `task/T<NN+1>` branches from `main` after `T<NN>` is merged, so it
starts with everything earlier tasks changed.

The authors' code is the `upstream` remote (`enjeeneer/zero-shot-rl`). Always refer to it as
`upstream/main`. Bringing their updates into `main` is a `chore/` PR that merges
`upstream/main`.

## Workflow per task

1. Branch `task/T<NN>-<name>` from `main`. Run experiments with `lab run`, which executes
   the pushed commit.
2. For every job whose results go into the report:
   - copy its checkpoints (best and final) and log from the lab host to `artifacts/T<NN>-<name>/<job id>/`,
     check the sha256 against the host copy, then delete the host copy;
   - tag the commit it ran with an annotated tag `exp/T<NN>-<job id>`. The tag message
     records what the report leaves out: seed, the repo-relative paths of the checkpoints
     and log under `artifacts/`, and the checkpoints' sha256. No host names: the fork is
     public, and `lab show <job id>` gives the host.

   A tag never points to a different commit. Its message may be corrected by re-creating
   the tag on the same commit.
3. Branch `clean/T<NN>-<name>` from `task/T<NN>-<name>`. Move, rewrite or delete files until
   only what passes the file rule is left, and open a PR to `task/T<NN>-<name>`.
4. After that PR is merged, open a PR from `task/T<NN>-<name>` to `main`.
5. Keep the `task/` branch and its tags after the merge. Files deleted in step 3 stay
   reachable through them.

## File rule

A file goes to `main` only if it is needed to:

- **run** a baseline or variant: algorithms, environments, entry points, environment setup;
- **reproduce** a reported number or figure: experiment configs, eval data, analysis scripts;
- **understand** the repo: `README.md`, this file, task READMEs.

Everything else (debugging, abandoned attempts, superseded scripts and figures) is deleted on
the `clean/` branch. It stays in the `task/` branch's history and tags.

## Layout

```
agents/ rewards/ custom_dmc_tasks/       upstream code: never moved or renamed
dmc.py utils.py main_exorl.py main_d4rl.py exorl_reformatter.py
scripts/                                 our entry points (run as `python -m scripts.<name>`)
analysis/                                analysis tools used by two or more tasks
tasks/T<NN>-<name>/
  REPORT.md                              results only
  README.md                              how to reproduce: commands, configs, code changes
  configs/                               experiment configs (sweep files, seeds)
  data/                                  eval data behind every reported number
  figures/                               figures used in REPORT.md
  *.py                                   analysis used only by this task
```

- Upstream files stay where they are, so merging `upstream/main` into `main` stays easy.
- A change that can alter training or evaluation results goes behind a new argument or config
  key whose default keeps the original behaviour, so every earlier result still reproduces
  from `main`. Changes that only affect logging or which files are kept need no switch.
- New baselines go in new modules (for example `agents/<name>/`), not in edits to existing
  agents.
- A task's analysis script moves to `analysis/` only when a second task needs it. The same PR
  updates the commands in earlier tasks' READMEs.

## Reports

- `REPORT.md` presents results: setup, numbers, figures, comparisons, limitations.
- No checkpoint paths, hashes, host names or job logs in reports. Those go in the `exp/` tag
  messages.
- Every number and figure in a report must be regenerable from `data/` with a command in the
  task's `README.md`.
  Numbers quoted from other sources (papers, other repos) name their source instead.

## Data and checkpoints

Datasets, checkpoints and job logs are never committed.

- Checkpoints and job logs are kept only on this machine, in `artifacts/` at the repo root
  (git-ignored), and referenced from `exp/` tag messages. The lab hosts only run jobs and
  keep no results. `git clean -x` deletes `artifacts/`; do not run it.
- Datasets are inputs and can be re-downloaded. They may stay on the lab hosts as a cache.
- Checkpoints are saved with `AbstractAgent.save` (class, constructor arguments and
  `state_dict`) and loaded with `load_agent`. Never pickle whole objects.
- `.gitignore` covers `artifacts/`, `datasets/` and `agents/*/saved_models/`.

## Tasks

| Task | Branch | Status |
|---|---|---|
| T01 FB reproduction on ExORL Walker RND-100k | `task/T01-fb-reproduction` | Report in `tasks/T01-fb-reproduction/` |
