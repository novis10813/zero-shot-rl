# AGENTS.md

Baselines, modified RL functions and FB latent experiments on top of the authors' zero-shot RL
code, one report per task.

## Where code lives

```
agents/ rewards/ custom_dmc_tasks/       upstream code: never moved or renamed
dmc.py utils.py main_exorl.py main_d4rl.py exorl_reformatter.py
zero_shot_rl/                            our code, used by two or more tasks
  latent/                                FB latents from stored data, no simulator
                                         solver, implicit, archive (library), infer (entry)
  eval/                                  runs the simulator
                                         env, trace_format (library), checkpoint, latents,
                                         trace, video (entry)
  provenance.py                          file hashes and no-clobber publication
analysis/                                analysis tools used by two or more tasks
tests/                                   component tests for zero_shot_rl/ and agents/
tasks/T<NN>-<name>/
  REPORT.md                              results only
  README.md                              how to reproduce: commands, configs, code changes
  configs/                               experiment configs (sources, sweeps, seeds)
  data/                                  eval data behind every reported number
    inputs/                              stored latents, rollouts and traces the analysis reads
  figures/                               figures used in REPORT.md
  analyze/, *.py                         analysis used only by this task
  tests/                                 tests of this task's analysis
```

- The authors' code is the `upstream` remote (`enjeeneer/zero-shot-rl`); refer to it as
  `upstream/main`. Upstream files stay where they are, so merging `upstream/main` into `main`
  stays easy. Bringing their updates into `main` is a `chore/` PR that merges `upstream/main`.
- Place a module in `zero_shot_rl/` by what it acts on: `latent/` for latents computed from
  data, `eval/` for anything that runs the simulator. Each module's docstring states what it does
  and how to run it.
- Run entry modules by full path from the repo root, e.g.
  `PYTHONPATH=. uv run python -m zero_shot_rl.latent.infer`. Task modules run with the task
  directory on the path, e.g. `PYTHONPATH=.:tasks/T05-opt-to-pessi uv run python -m analyze.q1`.
- New baselines go in new modules (for example `agents/<name>/`), not in edits to existing
  agents.
- A change to an upstream file that can alter training or evaluation results goes behind a new
  argument or config key whose default keeps the original behaviour.
- Code in `zero_shot_rl/` and `tasks/` may be changed. If a change alters what an existing entry
  module computes, say so in the commit message. Earlier results are reproduced from their `exp/`
  tags, not from `main`.
- A task's analysis code moves to `analysis/` or `zero_shot_rl/` when a second task needs it, and
  the same commit updates the commands in earlier tasks' READMEs.

## Verify

Run before every commit that touches code:

```
uv run ruff check
PYTHONPATH=. uv run python -m unittest discover -s tests
for d in tasks/*/tests; do
  t=${d%/tests}
  PYTHONPATH=.:$t MUJOCO_GL=egl uv run python -m unittest discover -s $d -t $t || break
done
```

Each task's tests run in their own process, because task modules such as `common.py` may share
names across tasks. A task README lists its `--check` commands, which verify that stored outputs
are reproduced byte for byte.

## Branches and tasks

| Branch | Purpose |
|---|---|
| `main` | runnable code and every finished task |
| `task/T<NN>-<name>` | one per task, branched from `main`; experiments and trial and error |
| `chore/<name>` | changes outside any task (refactors, tooling, docs, `upstream/main` merges); PR to `main`, deleted once merged |

Task numbers are two digits and never reused. Tasks run one at a time: `task/T<NN+1>` branches
from `main` after `T<NN>` is merged.

1. Branch `task/T<NN>-<name>` from `main`. Run experiments with `lab run`, which executes the
   pushed commit.
2. For every job whose results go into the report:
   - copy its checkpoints (best and final) and log from the lab host to
     `artifacts/T<NN>-<name>/<job id>/`, check the sha256 against the host copy, then delete the
     host copy;
   - tag the commit it ran with an annotated tag `exp/T<NN>-<job id>`. The message records seed,
     the repo-relative paths of the checkpoints and log under `artifacts/`, and the checkpoints'
     sha256. No host names: the fork is public, and `lab show <job id>` gives the host.
3. When the task ends, commit a cleanup on `task/T<NN>-<name>`: delete or move every file that
   fails the file rule below.
4. Open a PR from `task/T<NN>-<name>` to `main` and merge it with a merge commit, never squash, so
   the task's history and tagged commits stay in `main`.

Never rebase, force-push or delete a `task/` branch (a GitHub ruleset blocks both). A tag never
moves to a different commit; its message may be corrected by re-creating it on the same commit.

| Task | Branch | Status |
|---|---|---|
| T01 FB reproduction on ExORL Walker RND-100k | `task/T01-fb-reproduction` | Report in `tasks/T01-fb-reproduction/` |
| T04 MC-FB pessimism vs VC-FB and DVC-FB on ExORL Walker RND-100k | `task/T04-mcfb-pessimism` | Report in `tasks/T04-mcfb-pessimism/` |
| T05 Implicit-reward latents and rollouts for standard FB on ExORL Walker RND-100k | `task/T05-opt-to-pessi` | Report in `tasks/T05-opt-to-pessi/` |

## File rule

A file stays on `main` only if it is needed to:

- **run** a baseline or variant: algorithms, environments, entry points, environment setup;
- **reproduce** a reported number or figure: experiment configs, eval data, analysis scripts;
- **understand** the repo: `README.md`, this file, task READMEs.

Debugging scripts, abandoned attempts, working notes and superseded figures are deleted in the
cleanup commit. They stay reachable through the `task/` branch history and `exp/` tags.

## Reports

- `REPORT.md` presents results: setup, numbers, figures, comparisons, limitations.
- No checkpoint paths, hashes, host names or job logs in reports. Those go in `exp/` tag messages.
- Every number and figure must be regenerable from the task's `data/` with a command in its
  `README.md`. Numbers from other sources (papers, other repos) name their source.

## Data and checkpoints

- Datasets, checkpoints, job logs and videos are never committed. `.gitignore` covers
  `artifacts/`, `datasets/` and `agents/*/saved_models/`.
- Checkpoints, job logs and videos live only on this machine, in `artifacts/` at the repo root.
  Lab hosts run jobs and keep no results. `git clean -x` deletes `artifacts/`; never run it.
- Datasets are inputs and can be re-downloaded. They may stay on lab hosts as a cache.
- Checkpoints are saved with `AbstractAgent.save` (class, constructor arguments and `state_dict`)
  and loaded with `load_agent`. Never pickle whole objects.
