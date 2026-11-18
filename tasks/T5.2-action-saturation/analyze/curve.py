"""Return versus actuator gear: table and figure from the sweep CSVs.

Reads <inputs>/sweep_<task>.csv (zero_shot_rl.eval.dynamics_sweep, config names gear_<percent>)
and writes <data>/curve.csv (per task and gear: mean and std of return, return relative to
100% gear, mean speed and fall_frac) and <figures>/curve.png. Also prints, per task, the
largest gear (in percent) whose relative return falls below 0.9 and below 0.5.

Usage (repo root):
    PYTHONPATH=. uv run python tasks/T5.2-action-saturation/analyze/curve.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TASK_DIR = Path(__file__).resolve().parents[1]
TASKS = ["walk", "run", "stand", "flip"]


def load(inputs):
    frames = []
    for task in TASKS:
        path = inputs / f"sweep_{task}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["task"] = task
            df["gear"] = df["dynamics"].str.removeprefix("gear_").astype(int)
            frames.append(df)
    return pd.concat(frames)


def summarize(df):
    g = df.groupby(["task", "gear"])
    out = g.agg(return_mean=("return", "mean"), return_std=("return", "std"),
                speed=("speed", "mean"), fall_frac=("fall_frac", "mean"), n=("return", "size"))
    out = out.reset_index()
    base = out[out.gear == 100].set_index("task")["return_mean"]
    out["relative"] = out["return_mean"] / out["task"].map(base)
    return out.sort_values(["task", "gear"], ascending=[True, False])


def main():
    summary = summarize(load(TASK_DIR / "data" / "inputs"))
    summary.to_csv(TASK_DIR / "data" / "curve.csv", index=False, float_format="%.4g")
    tasks = list(summary.task.unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for task in tasks:
        d = summary[summary.task == task].sort_values("gear")
        axes[0].errorbar(d.gear, d.return_mean, d.return_std, marker="o", label=task, capsize=2)
        axes[1].plot(d.gear, d.relative, marker="o", label=task)
        axes[2].plot(d.gear, d.fall_frac, marker="o", label=task)
        for level in (0.9, 0.5):
            above = d[d.relative >= level].gear
            below = d[d.relative < level].gear
            print(f"{task}: relative return < {level} at gear <= "
                  f"{below.max() if len(below) else 'none'}% (still >= at {above.min()}%)")
    for ax, ylabel in zip(axes, ["return", "return / return at 100% gear", "fraction of steps fallen"]):
        ax.set_xlabel("actuator gear (% of XML)")
        ax.set_ylabel(ylabel)
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend()
    axes[1].axhline(0.9, color="gray", ls="--", lw=0.8)
    fig.tight_layout()
    (TASK_DIR / "figures").mkdir(exist_ok=True)
    fig.savefig(TASK_DIR / "figures" / "curve.png", dpi=150)


if __name__ == "__main__":
    main()
