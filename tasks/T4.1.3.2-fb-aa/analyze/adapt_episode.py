"""Closed-loop return of the action adaptation under gear shifts.

pack: from the adapt_episode_s{0,1,2}.npz of run_adapt_episode.sh (zero_shot_rl.eval.adapt_episode),
writes data/inputs/adapt_episode.npz with the three FB seeds stacked on the episode axis (N = 60,
the same environment seeds in every config and method, so returns pair across them):

    configs, gear [C], methods ["none", "star", "adapt"]
    task [N], seed [N]            task and FB training seed (0, 1, 2) of each episode
    ret [C, 3, N]                 return
    da  [C, 3, N]                 mean over the episode of |a_executed - a|

Tables, one row per config and method:

data/adapt_episode.csv
    mean         mean return over the 60 episodes
    d_none       mean paired difference to the method none in the same config, d_none_se its standard error
    d_stock      mean paired difference to none in the stock walker (config 0), d_stock_se its standard error
    da           mean |a_executed - a|

data/adapt_episode_task.csv
    mean         mean return over the 15 episodes of the task

Figure figures/adapt_episode.png: mean return per config and method, with the stock walker's return.

Usage (repo root):
    PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.adapt_episode pack --dir <dir with adapt_episode_s*.npz>
    PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.adapt_episode [--check]
    PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.adapt_episode figure
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[1]
TASKS = ["stand", "walk", "run", "flip"]
FIELDS = ["config", "gear", "method", "mean", "d_none", "d_none_se", "d_stock", "d_stock_se", "da"]
TASK_FIELDS = ["config", "gear", "method", "task", "mean"]


def paired(x, y):
    """Mean and standard error of the paired difference x - y."""
    d = np.asarray(x, float) - np.asarray(y, float)
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def pack(directory):
    parts = [np.load(Path(directory) / f"adapt_episode_s{s}.npz") for s in range(3)]
    for p in parts[1:]:
        if list(p["configs"]) != list(parts[0]["configs"]) or list(p["methods"]) != list(parts[0]["methods"]):
            raise ValueError("the seeds ran different configs or methods")
    out = dict(
        configs=parts[0]["configs"], gear=parts[0]["gear"], methods=parts[0]["methods"],
        task=np.concatenate([p["task"] for p in parts]), seed=np.repeat(np.arange(3), len(parts[0]["task"])),
        ret=np.concatenate([p["ret"] for p in parts], axis=2),
        da=np.concatenate([p["da"].mean(-1) for p in parts], axis=2))
    path = TASK / "data/inputs/adapt_episode.npz"
    np.savez_compressed(path, **out)
    print(f"wrote {path}")


def rows():
    data = np.load(TASK / "data/inputs/adapt_episode.npz")
    ret, da, task = data["ret"], data["da"], data["task"]
    out, by_task = [], []
    for c, name in enumerate(data["configs"]):
        for m, method in enumerate(data["methods"]):
            d_none, d_none_se = paired(ret[c, m], ret[c, 0]) if m else (0.0, 0.0)
            d_stock, d_stock_se = paired(ret[c, m], ret[0, 0]) if (c or m) else (0.0, 0.0)
            out.append(dict(config=str(name), gear=float(data["gear"][c]), method=str(method),
                            mean=ret[c, m].mean(), d_none=d_none, d_none_se=d_none_se,
                            d_stock=d_stock, d_stock_se=d_stock_se, da=da[c, m].mean()))
            for t in TASKS:
                by_task.append(dict(config=str(name), gear=float(data["gear"][c]), method=str(method),
                                    task=t, mean=ret[c, m][task == t].mean()))
    return out, by_task


def render(records, fields):
    lines = [",".join(fields)]
    for r in records:
        lines.append(",".join(r[k] if isinstance(r[k], str) else f"{r[k]:.6g}" for k in fields))
    return "\n".join(lines) + "\n"


def figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(TASK / "data/inputs/adapt_episode.npz")
    ret = data["ret"]
    order = np.argsort(data["gear"])  # lowest gear first
    plt.rcParams.update({"font.size": 9, "savefig.dpi": 130})
    fig, ax = plt.subplots(figsize=(8, 3.8))
    labels = {"none": "FB, no correction", "star": "ideal correction (known gear)", "adapt": "FB-AA (window context)"}
    colors = {"none": "0.6", "star": "tab:green", "adapt": "tab:blue"}
    width = 0.26
    for m, method in enumerate(data["methods"]):
        mean = ret[order, m].mean(-1)
        se = ret[order, m].std(-1, ddof=1) / np.sqrt(ret.shape[-1])
        ax.bar(np.arange(len(order)) + (m - 1) * width, mean, width, yerr=se, color=colors[str(method)],
               label=labels[str(method)], capsize=2)
    ax.axhline(ret[0, 0].mean(), color="k", ls="--", lw=0.8, label="stock walker, no correction")
    ax.set_xticks(np.arange(len(order)), [f"gear {data['gear'][i]:.2f}" for i in order])
    ax.set_ylabel("return (60 episodes)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    path = TASK / "figures/adapt_episode.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "pack":
        parser = ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--dir", required=True)
        return pack(parser.parse_args().dir)
    if len(sys.argv) > 1 and sys.argv[1] == "figure":
        return figure()
    records, by_task = rows()
    texts = {TASK / "data/adapt_episode.csv": render(records, FIELDS),
             TASK / "data/adapt_episode_task.csv": render(by_task, TASK_FIELDS)}
    for path, text in texts.items():
        if "--check" in sys.argv:
            if path.read_text() != text:
                sys.exit(f"{path} differs from the regenerated table")
        else:
            path.write_text(text)
    print("ok" if "--check" in sys.argv else "written")


if __name__ == "__main__":
    main()
