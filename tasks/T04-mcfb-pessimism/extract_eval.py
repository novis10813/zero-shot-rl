"""Extract the periodic eval scores from a training log into a CSV.

Usage:
    python tasks/T04-mcfb-pessimism/extract_eval.py <output.log> <out.csv>

Reads the "Step <i> eval: eval/<task>/episode_reward_iqm=<x>, ..." lines written by
agents/workspaces.py and writes one row per eval step: step, one column per task, mean.
"""

import csv
import re
import sys

LINE = re.compile(r"Step (\d+) eval: (.*)")
SCORE = re.compile(r"eval/([a-z_]+)/episode_reward_iqm=([-0-9.e]+)")


def main(log_path: str, out_path: str) -> None:
    rows = []
    with open(log_path, errors="replace") as f:
        for chunk in f.read().replace("\r", "\n").split("\n"):
            m = LINE.search(chunk)
            if m is None:
                continue
            scores = {task: float(x) for task, x in SCORE.findall(m.group(2))}
            rows.append((int(m.group(1)), scores))
    if not rows:
        raise SystemExit(f"no eval lines in {log_path}")
    tasks = list(rows[0][1])
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", *tasks, "mean"])
        for step, scores in rows:
            vals = [scores[t] for t in tasks]
            w.writerow([step, *(f"{v:.3f}" for v in vals), f"{sum(vals) / len(vals):.3f}"])


if __name__ == "__main__":
    main(*sys.argv[1:])
