# AGENTS.md

Zero-shot RL experiments built on the official code of Jeen et al. (NeurIPS 2024,
`enjeeneer/zero-shot-rl`). The repo will hold several baselines, modified RL functions and
one report per task. These rules keep `main` clean while experiments stay reproducible.

## Branches

| Branch | Purpose | Rules |
|---|---|---|
| `upstream` | the authors' `main` | Only updated from `enjeeneer/zero-shot-rl`. Never commit to it |
| `task/T<NN>-<name>` | one per task; experiments and trial and error | Anything goes, but never rebase, force-push or delete it |
| `clean/T<NN>-<name>` | branched from `main` when the task ends; holds only what passes the file rule | Opened as a PR to `main` |
| `main` | runnable code and every finished task | Changes only through a reviewed `clean/` PR |

Task numbers are two digits and never reused: `T01`, `T02`, ...

## Workflow per task

1. Branch `task/T<NN>-<name>` from `main`. Run experiments with `lab run`, which executes
   the pushed commit.
2. For every job whose results go into the report, tag the commit it ran with an annotated
   tag `exp/T<NN>-<job id>`. The tag message records what the report leaves out: seed, host,
   checkpoint path, sha256 and log path. Tags are never moved or deleted.
3. Branch `clean/T<NN>-<name>` from `main`, bring over the files that pass the file rule, and
   open a PR to `main`.
4. Keep the `task/` branch and its tags after the merge. They are the only record of the
   experiment code.

## File rule

A file goes to `main` only if it is needed to:

- **run** a baseline or variant: algorithms, environments, entry points, environment setup;
- **reproduce** a reported number or figure: experiment configs, eval data, analysis scripts;
- **understand** the repo: `README.md`, this file, task READMEs.

Everything else (debugging, abandoned attempts, superseded scripts and figures) stays on the
`task/` branch.

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

- Upstream files stay where they are, so merging `upstream` into `main` stays easy.
- A change that can alter training or evaluation results goes behind a new argument or config
  key whose default keeps the original behaviour, so every earlier result still reproduces
  from `main`. Changes that only affect logging or which files are kept need no switch.
- New baselines go in new modules (for example `agents/<name>/`), not in edits to existing
  agents.
- A task's analysis script moves to `analysis/` only when a second task needs it.

## Reports

- `REPORT.md` presents results: setup, numbers, figures, comparisons, limitations.
- No checkpoint paths, hashes, host names or job logs in reports. Those go in the `exp/` tag
  messages.
- Every number and figure in a report must be regenerable from `data/` with a command in the
  task's `README.md`.

## Data and checkpoints

Datasets, checkpoints and job logs are never committed. They live on the lab hosts and are
referenced from `exp/` tag messages. `.gitignore` covers `datasets/` and
`agents/*/saved_models/`.

## Tasks

| Task | Branch | Status |
|---|---|---|
| T01 FB reproduction on ExORL Walker RND-100k | `lab-repro` (predates these rules; plays the `task/` role) | Report in `tasks/T01-fb-reproduction/` |
