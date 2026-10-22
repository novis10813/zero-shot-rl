"""Two checks of the standard FB latent shifts (implicit_z_fb.py).

representation (needs the checkpoint and the dataset, no simulator beyond the dataset
action step used for z inference): how much of a reward can the linear map B(s')^T w
express? For each target y on the inference transitions it reports

    mean share   ybar^2 / E[y^2], the part of y that is a constant on the data
    R2 ls        held-out R^2 of the centred target fitted by least squares on B(s')
    R2 proj      R^2 of the centred target against c B(s')^T E_D[(y - ybar) B(s')], the
                 reward FB actually uses after projecting, best scale c

Targets: the task rewards, and the ReBRAC penalty p(s') = ||pi_{z_r}(s') - a'_D||^2 at the
task's z_r. CQL needs the behaviour density pi_beta at data actions and is not included.

residual (from data/zs_fb_<job>.npz only): removes from each family's direction away from
z_r its component along u_1 = E_D[B(s')], the direction of a constant reward on the data,
and compares what is left. Per family it reports the residual norm (0 = the direction is
u_1 alone) and the cosine between residuals of different families; C1 random directions
give the chance level.

Usage (repo root):
    PYTHONPATH=. uv run python tasks/T05-opt-to-pessi/checks_fb.py representation \
        --checkpoint <fb.pt> --dataset_path <dataset.npz> --out <representation.csv>
    uv run python tasks/T05-opt-to-pessi/checks_fb.py residual
writes data/fb_checks.md (residual, plus representation from data/representation_fb_<job>.csv
when present) and prints it.
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
JOBS = ["zero-shot-rl-30", "zero-shot-rl-31", "zero-shot-rl-32"]
TASKS = ["stand", "walk", "run", "flip"]
FAMILIES = ["CQL", "ReBRAC", "E2E", "E2E-bi"]


def representation(args):
    import torch

    sys.path.insert(0, str(ROOT))
    from agents.base import load_agent
    from agents.onestep_fb.replay_buffer import OneStepFBReplayBuffer
    from implicit_z import normalized
    from rewards import RewardFunctionConstructor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = load_agent(Path(args.checkpoint), device)
    rc = RewardFunctionConstructor(domain_name="walker", task_names=TASKS, seed=args.seed,
                                   device=device)
    buffer = OneStepFBReplayBuffer(reward_constructor=rc, dataset_path=Path(args.dataset_path),
                                   discount=0.98, transitions=args.transitions, device=device)
    # same inference set as implicit_z_fb.py
    rng = np.random.default_rng(args.seed)
    valid = buffer.valid_indices.cpu().numpy()
    idx = rng.choice(valid, min(args.n_inference, len(valid)), replace=False)
    st = buffer.storage
    next_obs, rewards = rc.process_timesteps(batch_physics=st["physics"][idx],
                                             batch_actions=st["actions"][idx].cpu().numpy())
    next_actions = st["next_actions"][torch.as_tensor(idx, device=device)]
    with torch.no_grad():
        B = torch.cat([agent.FB.backward_representation(o) for o in next_obs.split(20000)])
    Bn = B.double().cpu().numpy()
    half = len(Bn) // 2

    def stats(y):
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        yc = y - y.mean()
        w, *_ = np.linalg.lstsq(Bn[:half], yc[:half], rcond=None)
        res = yc[half:] - Bn[half:] @ w
        r2_ls = 1 - (res ** 2).sum() / ((yc[half:] - yc[half:].mean()) ** 2).sum()
        proj = Bn @ (Bn * yc[:, None]).mean(0)
        c = (proj @ yc) / (proj @ proj)
        r2_proj = 1 - ((yc - c * proj) ** 2).sum() / (yc ** 2).sum()
        return {"mean_share": y.mean() ** 2 / (y ** 2).mean(), "r2_ls": r2_ls, "r2_proj": r2_proj}

    rows = []
    for task in TASKS:
        r = rewards[task].reshape(-1, 1)
        rows.append({"task": task, "target": "task reward", **stats(r.cpu().numpy())})
        z = normalized((r * B).mean(0), agent._z_dimension)
        with torch.no_grad():
            a = torch.cat([agent.actor(o, z.expand(len(o), -1), 0.2, sample=False)[0]
                           for o in next_obs.split(20000)]).clamp(-1.0, 1.0)
        p = ((a - next_actions) ** 2).sum(1)
        rows.append({"task": task, "target": "ReBRAC penalty", **stats(p.cpu().numpy())})
        print(rows[-2], rows[-1], flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6g")


def unit(v):
    return v / np.linalg.norm(v)


def perp(v, *basis):
    for b in basis:
        v = v - (v @ b) * b
    return v


def residual(args):
    rows, pairs = [], []
    for seed, job in enumerate(JOBS):
        d = np.load(ROOT / "data" / f"zs_fb_{job}.npz")
        for task in TASKS:
            i = np.flatnonzero(d["task"] == task)
            fam, tgt = d["family"][i], d["target_angle"][i]
            z_r = unit(d["z"][i[fam == "z_r"][0]])
            u = unit(perp(d["u_1"], z_r))  # u_1 direction, orthogonal to z_r
            res = {}
            for f in FAMILIES + ["C1"]:
                sel = i[(fam == f) & (tgt <= 10)]
                if f == "C1":
                    sel = i[(fam == f) & (tgt == 10)]
                for k, j in enumerate(sel[-1:] if f != "C1" else sel):
                    t = unit(perp(d["z"][j], z_r))
                    r = perp(t, u)
                    name = f if f != "C1" else f"C1_{k}"
                    res[name] = r
                    rows.append({"seed": seed, "task": task, "family": f,
                                 "cos_u1": t @ u, "residual_norm": np.linalg.norm(r)})
            names = list(res)
            for a in range(len(names)):
                for b in range(a + 1, len(names)):
                    fa, fb = names[a].split("_")[0], names[b].split("_")[0]
                    if fa == fb or "E2E" in (fa, fb):  # E2E practical has no residual
                        continue
                    pairs.append({"seed": seed, "task": task, "pair": " / ".join(sorted((fa, fb))),
                                  "cos": unit(res[names[a]]) @ unit(res[names[b]])})
    rows, pairs = pd.DataFrame(rows), pd.DataFrame(pairs)

    lines = ["Residual check: each family's direction away from z_r (10 deg, 5 deg if 10 is not "
             "reached) with its u_1 component removed. Mean over the 4 tasks per seed.", "",
             "| Family | cos to u_1 (seed 0 / 1 / 2) | residual norm (seed 0 / 1 / 2) |", "|---|---|---|"]
    by = rows.groupby(["family", "seed"]).mean(numeric_only=True)
    for f in FAMILIES + ["C1"]:
        c = " / ".join(f"{by.loc[(f, s), 'cos_u1']:+.2f}" for s in range(len(JOBS)))
        n = " / ".join(f"{by.loc[(f, s), 'residual_norm']:.2f}" for s in range(len(JOBS)))
        lines.append(f"| {f} | {c} | {n} |")
    lines += ["", "| Residual pair | cos (seed 0 / 1 / 2), mean over tasks | range over all seeds and tasks |",
              "|---|---|---|"]
    pb = pairs.groupby(["pair", "seed"]).cos.mean()
    for pair, g in pairs.groupby("pair"):
        c = " / ".join(f"{pb.loc[(pair, s)]:+.2f}" for s in range(len(JOBS)))
        lines.append(f"| {pair} | {c} | {g.cos.min():+.2f} to {g.cos.max():+.2f} |")
    lines += ["", "C1 rows are three random directions per task: their cosines with a family give the "
              "chance level (about +-0.14 for 48 free dimensions)."]

    rep = [ROOT / "data" / f"representation_fb_{j}.csv" for j in JOBS]
    if all(p.exists() for p in rep):
        r = pd.concat([pd.read_csv(p).assign(seed=s) for s, p in enumerate(rep)])
        lines += ["", "Representation check: how much of each reward the 50-d B(s') expresses on the "
                  "inference transitions. Mean over 4 tasks and 3 seeds (min to max).", "",
                  "| Target | mean share | R2 ls (centred, held out) | R2 proj (centred, as FB uses it) |",
                  "|---|---|---|---|"]
        for target, g in r.groupby("target", sort=False):
            cells = [f"{g[m].mean():.2f} ({g[m].min():.2f} to {g[m].max():.2f})"
                     for m in ("mean_share", "r2_ls", "r2_proj")]
            lines.append(f"| {target} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    (ROOT / "data" / "fb_checks.md").write_text(text + "\n")
    print(text)


def main():
    parser = ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    rep = sub.add_parser("representation")
    rep.add_argument("--checkpoint", type=str, required=True)
    rep.add_argument("--dataset_path", type=str, required=True)
    rep.add_argument("--out", type=str, required=True)
    rep.add_argument("--transitions", type=int, default=100000)
    rep.add_argument("--n_inference", type=int, default=100000)
    rep.add_argument("--seed", type=int, default=0)
    sub.add_parser("residual")
    args = parser.parse_args()
    representation(args) if args.mode == "representation" else residual(args)


if __name__ == "__main__":
    main()
