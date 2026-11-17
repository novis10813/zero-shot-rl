"""Tables of REPORT.md from the 10 finetune seeds in data/inputs/finetune/.

    ft.<env>_<task>_s<seed>.csv   residual soft Q-learning (omega 0.1, alpha 0.1), finetune seed
                                  1000..1009, tested on 20 episodes (evaluation seeds 5..24)
                                  after 0, 1, 3, 5 and 10 training episodes, plus the frozen FB
                                  actor on the same test episodes

Step 1: the IQM of the 20 test returns (mean of the middle 10 after sorting) of one finetune seed.
Step 2: the mean and the sample standard deviation of that IQM over the 10 finetune seeds.
The finetune seed also seeds the inference of the task latent z, so the frozen FB actor differs
between finetune seeds (same test episodes, different z). Every residual run is therefore compared
with the frozen actor of its own finetune seed: "gain" is residual IQM minus frozen IQM of the
same finetune seed, and "seeds above frozen" counts the finetune seeds with a positive gain.
The all-task rows average step 1 over the 4 tasks per finetune seed before step 2, and add a
95% bootstrap interval of the mean gain over finetune seeds (10000 resamples, numpy seed 0).

Writes data/finetune_tables.md; --check compares it with the file instead of writing.

Usage (repo root):
    PYTHONPATH=.:tasks/T4.1.3.1-residual-soft-q uv run python -m analyze.finetune_seeds [--check]
"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[1]
INPUTS = TASK / "data" / "inputs" / "finetune"
OUT = TASK / "data" / "finetune_tables.md"

ENVS = [("clean", "Clean"), ("mild", "Actuator Mild"), ("strong", "Actuator Strong")]
TASKS = ["stand", "walk", "run", "flip"]
SEEDS = list(range(1000, 1010))
EPISODES = [0, 1, 3, 5, 10]
TEST_EPISODES = 20


def iqm(returns):
    assert len(returns) == TEST_EPISODES, len(returns)
    values = sorted(returns)
    k = len(values) // 4
    return statistics.mean(values[k:len(values) - k])


def load(env, task, seed):
    """({train_episodes: IQM of the residual returns}, IQM of the frozen returns)."""
    groups = defaultdict(list)
    for row in csv.DictReader(open(INPUTS / f"ft.{env}_{task}_s{seed}.csv")):
        groups[(row["variant"], int(row["train_episodes"]))].append(float(row["return"]))
    residual = {ep: iqm(groups[("residual", ep)]) for ep in EPISODES}
    return residual, iqm(groups[("fb", 0)])


def bootstrap(values):
    rng = np.random.default_rng(0)
    means = rng.choice(values, size=(10000, len(values))).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return low, high


def ms(values):
    return f"{statistics.mean(values):.1f} ± {statistics.stdev(values):.1f}"


def build():
    out = ["每格是 10 個 finetune seed 的平均 ± 樣本標準差。finetune seed 同時決定 z 的推論，"
           "所以 frozen FB 也隨 seed 變動；增益是同一個 seed 的 residual IQM 減 frozen IQM，"
           "括號是增益為正的 seed 數。", ""]
    for env, env_name in ENVS:
        data = {(t, s): load(env, t, s) for t in TASKS for s in SEEDS}
        out += [f"### {env_name}", "",
                "| 任務 | frozen FB | " + " | ".join(f"{ep} episode 增益" for ep in EPISODES) + " |",
                "| --- | --- | " + " | ".join("---" for _ in EPISODES) + " |"]
        for t in TASKS:
            frozen = [data[(t, s)][1] for s in SEEDS]
            cells = []
            for ep in EPISODES:
                gains = [data[(t, s)][0][ep] - data[(t, s)][1] for s in SEEDS]
                cells.append(f"{ms(gains)}（{sum(g > 0 for g in gains)}/10）")
            out.append(f"| {t} | {ms(frozen)} | " + " | ".join(cells) + " |")
        frozen = [statistics.mean(data[(t, s)][1] for t in TASKS) for s in SEEDS]
        cells = []
        for ep in EPISODES:
            gains = [statistics.mean(data[(t, s)][0][ep] - data[(t, s)][1] for t in TASKS)
                     for s in SEEDS]
            low, high = bootstrap(gains)
            cells.append(f"{ms(gains)} [{low:.1f}, {high:.1f}]（{sum(g > 0 for g in gains)}/10）")
        out += [f"| 4 任務平均 | {ms(frozen)} | " + " | ".join(cells) + " |", ""]
    return "\n".join(out)


def main():
    text = build()
    if "--check" in sys.argv:
        if OUT.read_text() != text:
            sys.exit(f"{OUT} differs from the tables built from {INPUTS}")
        print(f"{OUT.relative_to(TASK.parents[1])} reproduced")
    else:
        OUT.write_text(text)
        print(f"wrote {OUT.relative_to(TASK.parents[1])}")


if __name__ == "__main__":
    main()
