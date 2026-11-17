"""Cross-environment tables of REPORT.md from data/inputs/crossseeds/ and data/inputs/finetune/.

    crossseeds/<train>_to_<eval>_s<seed>.csv   the residual critic of finetune seed <seed> trained
                                               for 10 episodes in <train>, tested in <eval>
                                               (omega 0.1), and the frozen FB actor at the same z
    finetune/ft.<env>_<task>_s<seed>.csv       the diagonal: trained and tested in <env>

Per finetune seed the IQM of the 20 test returns (mean of the middle 10) of the residual critic
minus the IQM of the frozen FB actor of the same finetune seed, then the mean and sample standard
deviation over the 10 finetune seeds. The 4-task rows average the gain over the tasks per seed
first and add the 95% bootstrap interval of the mean over seeds (as in analyze.finetune_seeds).

Writes data/cross_seeds_tables.md; --check compares it with the file instead of writing.

Usage (repo root):
    PYTHONPATH=.:tasks/T4.1.3.1-residual-soft-q uv run python -m analyze.cross_seeds [--check]
"""

import csv
import statistics
import sys
from collections import defaultdict

from analyze.finetune_seeds import ENVS, SEEDS, TASK, TASKS, bootstrap, iqm, load, ms

CROSS = TASK / "data" / "inputs" / "crossseeds"
OUT = TASK / "data" / "cross_seeds_tables.md"


def load_cross(train, test, seed):
    """{task: IQM gain over the frozen FB actor of the same seed}."""
    groups = defaultdict(list)
    for row in csv.DictReader(open(CROSS / f"{train}_to_{test}_s{seed}.csv")):
        groups[(row["variant"], row["task"])].append(float(row["return"]))
    return {t: iqm(groups[("residual", t)]) - iqm(groups[("fb", t)]) for t in TASKS}


def gains(train, test):
    """{seed: {task: gain}}; the diagonal comes from the finetune runs."""
    if train == test:
        return {s: {t: load(train, t, s)[0][10] - load(train, t, s)[1] for t in TASKS} for s in SEEDS}
    return {s: load_cross(train, test, s) for s in SEEDS}


def build():
    names = dict(ENVS)
    table = {(a, b): gains(a, b) for a, _ in ENVS for b, _ in ENVS}
    out = ["每格是 10 個 finetune seed 的增益（residual IQM 減同 seed 的 frozen FB IQM）平均 ± 樣本標準差，"
           "括號是增益為正的 seed 數。對角線是同環境訓練與測試。", "",
           "### 4 任務平均增益", "",
           "| 測試環境 | " + " | ".join(f"Train: {names[a]}" for a, _ in ENVS) + " |",
           "| --- | " + " | ".join("---" for _ in ENVS) + " |"]
    for test, test_name in ENVS:
        cells = []
        for train, _ in ENVS:
            g = [statistics.mean(table[(train, test)][s][t] for t in TASKS) for s in SEEDS]
            low, high = bootstrap(g)
            cells.append(f"{ms(g)} [{low:.1f}, {high:.1f}]（{sum(x > 0 for x in g)}/10）")
        out.append(f"| {test_name} | " + " | ".join(cells) + " |")
    out.append("")
    for test, test_name in ENVS:
        out += [f"### 測試環境 {test_name}，逐任務增益", "",
                "| 任務 | " + " | ".join(f"Train: {names[a]}" for a, _ in ENVS) + " |",
                "| --- | " + " | ".join("---" for _ in ENVS) + " |"]
        for t in TASKS:
            cells = []
            for train, _ in ENVS:
                g = [table[(train, test)][s][t] for s in SEEDS]
                cells.append(f"{ms(g)}（{sum(x > 0 for x in g)}/10）")
            out.append(f"| {t} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def main():
    text = build()
    if "--check" in sys.argv:
        if OUT.read_text() != text:
            sys.exit(f"{OUT} differs from the tables built from {CROSS}")
        print(f"{OUT.relative_to(TASK.parents[1])} reproduced")
    else:
        OUT.write_text(text)
        print(f"wrote {OUT.relative_to(TASK.parents[1])}")


if __name__ == "__main__":
    main()
