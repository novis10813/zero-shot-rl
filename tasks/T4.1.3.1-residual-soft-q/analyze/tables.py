"""Tables of REPORT.md from the eval.csv of the omega sweep in data/inputs/.

    omega_<env>_<task>_w<omega>.csv   residual soft Q-learning (alpha 0.1) trained for 10 episodes
                                      on one task, tested on 20 episodes (seeds 5..24) after
                                      0, 1, ..., 10 training episodes, and the frozen FB actor on
                                      the same test episodes

Every cell is the IQM of the 20 test returns (mean of the middle 10 after sorting, the 25%
trimmed mean) +- the sample standard deviation of those 10 returns.

Writes data/tables.md; --check compares it with the file instead of writing.

Usage (repo root):
    PYTHONPATH=.:tasks/T4.1.3.1-residual-soft-q uv run python -m analyze.tables [--check]
"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
INPUTS = TASK / "data" / "inputs"
OUT = TASK / "data" / "tables.md"

ENVS = [("clean", "Clean"), ("mild", "Actuator Mild"), ("strong", "Actuator Strong")]
TASKS = ["stand", "walk", "run", "flip"]
OMEGAS = ["0.01", "0.03", "0.1"]
EPISODES = list(range(11))
TEST_EPISODES = 20


def middle(returns):
    """The returns the IQM averages: the middle half after sorting."""
    values = sorted(returns)
    k = len(values) // 4
    return values[k:len(values) - k]


def iqm_std(returns):
    assert len(returns) == TEST_EPISODES, len(returns)
    m = middle(returns)
    return statistics.mean(m), statistics.stdev(m)


def cell(returns):
    mean, std = iqm_std(returns)
    return f"{mean:.1f} ± {std:.1f}"


def load(path):
    """{(variant, train_episodes): [returns in seed order]}."""
    groups = defaultdict(list)
    for row in sorted(csv.DictReader(open(path)), key=lambda r: int(r["seed"])):
        groups[(row["variant"], int(row["train_episodes"]))].append(float(row["return"]))
    return groups


def build():
    out = []
    for env, env_name in ENVS:
        for task in TASKS:
            runs = {w: load(INPUTS / f"omega_{env}_{task}_w{w}.csv") for w in OMEGAS}
            frozen = {tuple(runs[w][("fb", 0)]) for w in OMEGAS}
            assert len(frozen) == 1, (env, task)
            frozen = list(frozen.pop())
            out += [f"### {env_name}, {task}（frozen FB {cell(frozen)}）", "",
                    "| 訓練 episode 數 | " + " | ".join(f"ω = {w}" for w in OMEGAS) + " |",
                    "| --- | " + " | ".join("---" for _ in OMEGAS) + " |"]
            for ep in EPISODES:
                out.append(f"| {ep} | " + " | ".join(cell(runs[w][("residual", ep)])
                                                     for w in OMEGAS) + " |")
            out.append("")
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
