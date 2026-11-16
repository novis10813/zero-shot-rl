"""How much of a gear shift an action correction can undo: the torque the stock actions need against what the shifted motors can give.

Walker motors apply gear_i * clip(a_i, -1, 1). Under a gear multiplier m the torque limit is
m * gear_i, so a stock torque |tau_i| = gear_i * |a_i| above it cannot be reproduced by any action.

pack: from the action_bound_s{0,1,2}.npz of run_action_bound.sh (zero_shot_rl.eval.action_bound,
every step of 20 stock episodes per seed) and the adapt_step_s{0,1,2}.npz of run_adapt_step.sh
(zero_shot_rl.eval.adapt_step, every 8th step from step 32), writes data/inputs/action_bound.npz,
the three seeds stacked on the sample axis:

    a [60000, 6]                         the stock actions
    configs, gear [4]                    the gear-shifted dynamics
    b_none, b_star [4, 60000]            |B(s'_k(a)) - B(s'_0(a))| and the same with a* = clip(a / m)
    rows [21780]                         the rows of the 60000 that adapt_step sampled
    b_adapt [4, 2, 6, 21780]             |B(s'_k(a')) - B(s'_0(a))| of the learned correction, kinds
                                         window and oracle, grid ks x lrs
    kinds, ks, lrs

Table data/action_bound.csv, one row per gear-shifted config:

    infeasible   share of the 360,000 stock action entries with |a_i| > m (all steps)
    shortfall    sum_i max(|tau_i| - m gear_i, 0) / sum_i |tau_i| over all steps: the share of the stock
                 torque the shifted motors cannot give
    b_none, b_star, b_window, b_oracle   median |B(s') - B(s'_stock)| without correction, with a*, and
                 with the learned correction (K = 20, lr = 0.05) from the window and the oracle context,
                 on the 21,780 steps adapt_step sampled
    rec_star, rec_window, rec_oracle     1 - b_* / b_none, the share of the gap undone

Figure figures/action_bound.png: the stock torque distribution per joint type with the torque
limits of the two gear-reduced settings, and the infeasible share and the shortfall as functions
of m.

Usage (repo root):
    PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.action_bound pack \
        --bound <dir with action_bound_s*.npz> --adapt <dir with adapt_step_s*.npz>
    PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.action_bound [--check]
    PYTHONPATH=.:tasks/T4.1.3.2-fb-aa uv run python -m analyze.action_bound figure
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[1]
GEAR = np.array([100.0, 50.0, 20.0, 100.0, 50.0, 20.0])  # right and left hip, knee, ankle (walker.xml)
JOINTS = ["hip", "knee", "ankle"]
LEARNED = (20, 0.05)  # K and lr of the learned correction in the table
FIELDS = ["config", "gear", "infeasible", "shortfall", "b_none", "b_star", "b_window", "b_oracle",
          "rec_star", "rec_window", "rec_oracle"]


def infeasible(a, m):
    """Share of the action entries whose stock torque a gear multiplier m cannot give."""
    return (np.abs(np.clip(a, -1, 1)) > m).mean()


def shortfall(a, m, gear=GEAR):
    """Share of the stock torque above the limit m * gear of the joint."""
    tau = np.abs(np.clip(a, -1, 1)) * gear
    return np.maximum(tau - m * gear, 0).sum() / tau.sum()


def sampled_rows(bound, adapt):
    """Rows of the action_bound samples (episode-major, all steps) that adapt_step sampled."""
    steps = int(bound["t"].max()) + 1
    position = {int(e): i for i, e in enumerate(bound["episode"][::steps])}
    rows = np.array([position[int(e)] * steps + int(t) for e, t in zip(adapt["episode"], adapt["t"])])
    if not (np.array_equal(bound["episode"][rows], adapt["episode"]) and np.array_equal(bound["t"][rows], adapt["t"])):
        raise ValueError("the samples of action_bound and adapt_step do not match")
    return rows


def pack(bound_dir, adapt_dir):
    parts = {k: [] for k in ["a", "b_none", "b_star", "rows", "b_adapt"]}
    offset = 0
    for seed in range(3):
        bound = np.load(Path(bound_dir) / f"action_bound_s{seed}.npz")
        adapt = np.load(Path(adapt_dir) / f"adapt_step_s{seed}.npz")
        if list(bound["configs"]) != list(adapt["configs"]):
            raise ValueError("the dynamics of the two jobs differ")
        shifted = np.flatnonzero(bound["gear"] != 1.0)
        rows = sampled_rows(bound, adapt)
        if not np.allclose(bound["b_none"][:, rows], adapt["b_none"], atol=1e-4):
            raise ValueError("the uncorrected B gaps of the two jobs differ")
        parts["a"].append(bound["a"])
        parts["b_none"].append(bound["b_none"][shifted])
        parts["b_star"].append(bound["b_star"][shifted])
        parts["rows"].append(rows + offset)
        parts["b_adapt"].append(adapt["b_adapt"][shifted])
        offset += len(bound["a"])
    out = {k: np.concatenate(v, axis=-1 if k != "a" and k != "rows" else 0) for k, v in parts.items()}
    out.update(configs=bound["configs"][shifted], gear=bound["gear"][shifted],
               kinds=adapt["kinds"], ks=adapt["ks"], lrs=adapt["lrs"])
    path = TASK / "data/inputs/action_bound.npz"
    np.savez_compressed(path, **out)
    print(f"wrote {path}")


def rows():
    data = np.load(TASK / "data/inputs/action_bound.npz")
    a, sample = data["a"], data["rows"]
    g = [i for i, (k, lr) in enumerate(zip(data["ks"], data["lrs"])) if (k, lr) == LEARNED][0]
    out = []
    for c, name in enumerate(data["configs"]):
        m = float(data["gear"][c])
        med = {"none": np.median(data["b_none"][c, sample]), "star": np.median(data["b_star"][c, sample]),
               "window": np.median(data["b_adapt"][c, 0, g]), "oracle": np.median(data["b_adapt"][c, 1, g])}
        out.append(dict(config=str(name), gear=m, infeasible=infeasible(a, m), shortfall=shortfall(a, m),
                        b_none=med["none"], b_star=med["star"], b_window=med["window"], b_oracle=med["oracle"],
                        rec_star=1 - med["star"] / med["none"], rec_window=1 - med["window"] / med["none"],
                        rec_oracle=1 - med["oracle"] / med["none"]))
    return out


def render(records):
    lines = [",".join(FIELDS)]
    for r in records:
        lines.append(",".join(r[k] if k == "config" else f"{r[k]:.6g}" for k in FIELDS))
    return "\n".join(lines) + "\n"


def figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(TASK / "data/inputs/action_bound.npz")
    a = np.abs(np.clip(data["a"], -1, 1))
    reduced = [(str(n), round(float(m), 2)) for n, m in zip(data["configs"], data["gear"]) if m < 1]
    plt.rcParams.update({"font.size": 9, "savefig.dpi": 130})
    fig = plt.figure(figsize=(10, 6.2))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.45, wspace=0.3)
    for j, joint in enumerate(JOINTS):
        ax = fig.add_subplot(grid[0, j])
        tau = np.concatenate([a[:, j], a[:, j + 3]]) * GEAR[j]
        ax.hist(tau, bins=60, range=(0, GEAR[j]), color="0.55")
        for (name, m), color in zip(reduced, ["tab:orange", "tab:red"]):
            ax.axvline(m * GEAR[j], color=color, ls="--", label=f"limit at m={m}")
        ax.set_yscale("log")
        ax.set_xlabel(f"stock |torque| of the {joint} (N m)")
        ax.set_title(joint)
        if j == 0:
            ax.set_ylabel("steps (log)")
            ax.legend(frameon=False, loc="upper center")
    ax = fig.add_subplot(grid[1, :])
    ms = np.linspace(0.5, 1.6, 111)
    ax.plot(ms, [infeasible(data["a"], m) for m in ms], label="share of entries with |a| > m")
    ax.plot(ms, [shortfall(data["a"], m) for m in ms], label="share of stock torque above the limit")
    for name, m in zip(data["configs"], data["gear"]):
        ax.axvline(m, color="0.8", lw=0.8, zorder=0)
        ax.text(m, 1.02, str(name).replace("actuator_", "").replace("_", " "), ha="center", fontsize=7, rotation=0)
    ax.set_xlabel("gear multiplier m")
    ax.set_ylabel("cannot be undone by any action")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, loc="center right")
    path = TASK / "figures/action_bound.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path)
    print(f"wrote {path}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "pack":
        parser = ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--bound", required=True)
        parser.add_argument("--adapt", required=True)
        args = parser.parse_args()
        return pack(args.bound, args.adapt)
    if len(sys.argv) > 1 and sys.argv[1] == "figure":
        return figure()
    text = render(rows())
    path = TASK / "data/action_bound.csv"
    if "--check" in sys.argv:
        if path.read_text() != text:
            sys.exit(f"{path} differs from the regenerated table")
    else:
        path.write_text(text)
    print("ok" if "--check" in sys.argv else "written")


if __name__ == "__main__":
    main()
