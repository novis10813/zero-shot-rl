"""Tables of REPORT.md from the eval.csv of the pilot jobs in data/inputs/.

    curve_<env>_<radius>.csv    Residual (hard max, Q 64x2) trained for 100 episodes, tested on 5
                                episodes after every 10, and the frozen FB on the same episodes
    search_<env>_<radius>.csv   the greedy rule on the FB value alone, no training, and the
                                frozen FB, on 20 test episodes

Writes data/tables.md; --check compares it with the file instead of writing.

Usage (repo root):
    PYTHONPATH=.:tasks/T4.1.3.0-online-value-correction uv run python -m analyze.tables [--check]
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
INPUTS = TASK / "data" / "inputs"
OUT = TASK / "data" / "tables.md"
T04_EVAL = TASK.parent / "T04-mcfb-pessimism" / "data" / "eval_fb_s0.csv"
T04_STEP = "800000"

ENVS = [("clean", "Clean"), ("mild", "Actuator Mild"), ("strong", "Actuator Strong")]
RADII = [("r0.2", "半徑 0.2"), ("unbounded", "不限半徑")]
EARLY, LATE = (0, 10, 20), (70, 80, 90, 100)


def means(path):
    """{(variant, train_episodes): mean return over the test episodes}, and the counts."""
    groups = defaultdict(list)
    for row in csv.DictReader(open(path)):
        groups[(row["variant"], int(row["train_episodes"]))].append(float(row["return"]))
    return ({k: sum(v) / len(v) for k, v in groups.items()},
            {k: len(v) for k, v in groups.items()})


def t04_clean():
    for row in csv.DictReader(open(T04_EVAL)):
        if row["step"] == T04_STEP:
            return float(row["walk"])
    raise ValueError("step not found")


def mean(values):
    return sum(values) / len(values)


def half_up(x):
    return math.floor(x + 0.5 + 1e-9)


def signed(x):
    return f"{x:+.1f}"


def build():
    curves = {(e, r): means(INPUTS / f"curve_{e}_{r}.csv") for e, _ in ENVS for r, _ in RADII}
    searches = {(e, r): means(INPUTS / f"search_{e}_{r}.csv") for e, _ in ENVS for r, _ in RADII}
    frozen5, frozen20 = {}, {}
    for e, _ in ENVS:
        values = {round(curves[(e, r)][0][("fb", 0)], 3) for r, _ in RADII}
        assert len(values) == 1, (e, values)
        frozen5[e] = curves[(e, RADII[0][0])][0][("fb", 0)]
        values = {round(searches[(e, r)][0][("fb", 0)], 3) for r, _ in RADII}
        assert len(values) == 1, (e, values)
        frozen20[e] = searches[(e, RADII[0][0])][0][("fb", 0)]
        assert curves[(e, "r0.2")][1][("fb", 0)] == 5 and searches[(e, "r0.2")][1][("fb", 0)] == 20
    out = ["## Frozen FB 的 return", "",
           "| 環境 | 5 個測試 episode（與訓練曲線同一批） | 20 個測試 episode（與 Search 同一批） | T04 的評估 |",
           "| --- | --- | --- | --- |"]
    for e, name in ENVS:
        t04 = f"{t04_clean():.1f}" if e == "clean" else "—"
        out.append(f"| {name} | {frozen5[e]:.1f} | {frozen20[e]:.1f} | {t04} |")

    out += ["", "## 訓練曲線（每個 checkpoint 的 5 個測試 episode 平均 return）", ""]
    header = ["訓練 episode 數"]
    for e, name in ENVS:
        for r, rname in RADII:
            header.append(f"{name}（frozen FB {frozen5[e]:.1f}）{rname}")
    out += ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    points = sorted({k[1] for k in curves[("clean", "r0.2")][0] if k[0] == "residual"})
    for p in points:
        cells = [str(p)]
        for e, _ in ENVS:
            for r, _ in RADII:
                cells.append(f"{curves[(e, r)][0][('residual', p)]:.1f}")
        out.append("| " + " | ".join(cells) + " |")

    out += ["", "## 訓練前期與後期的平均", "",
            "| 環境 | frozen FB | 半徑 0.2：ep0–20 / 70–100 | 不限半徑：ep0–20 / 70–100 | 後期差（不限 − 半徑） |",
            "| --- | --- | --- | --- | --- |"]
    for e, name in ENVS:
        late, cells = {}, []
        for r, _ in RADII:
            m = curves[(e, r)][0]
            early = mean([m[("residual", p)] for p in EARLY])
            late[r] = mean([m[("residual", p)] for p in LATE])
            cells.append(f"{half_up(early)} / {half_up(late[r])}")
        diff = half_up(late["unbounded"]) - half_up(late["r0.2"])
        out.append(f"| {name} | {frozen5[e]:.1f} | {cells[0]} | {cells[1]} | {diff:+d} |")

    out += ["", "## 只用 frozen FB 的動作搜尋（20 個測試 episode）", "",
            "| 環境 | Frozen FB | Search，不限半徑 | Search，半徑 0.2 |", "| --- | --- | --- | --- |"]
    for e, name in ENVS:
        cells = []
        for r in ("unbounded", "r0.2"):
            s = searches[(e, r)][0][("search", 0)]
            cells.append(f"{s:.1f} ({signed(s - frozen20[e])})")
        out.append(f"| {name} | {frozen20[e]:.1f} | {cells[0]} | {cells[1]} |")
    return "\n".join(out) + "\n"


def main():
    text = build()
    if "--check" in sys.argv[1:]:
        if OUT.read_text() != text:
            raise SystemExit(f"{OUT} differs from the tables rebuilt from data/inputs/")
        print("data/tables.md reproduced")
    else:
        OUT.write_text(text)
        print(text)


if __name__ == "__main__":
    main()
