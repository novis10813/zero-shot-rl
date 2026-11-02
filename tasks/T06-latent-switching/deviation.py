"""Action deviation of the walk policies of a fixed FB on dataset states, no simulator.

On --n_states states s drawn from the training buffer (100k transitions, seed-42
subsampling of OfflineReplayBuffer), with their dataset actions a_D, it writes per policy

    dev_data  mean ||a(s) - a_D||^2   distance to the behaviour data (ReBRAC's BC term)
    dev_opt   mean ||a(s) - pi(s, z_r)||^2   distance to the FB policy without correction
    pess_frac fraction of states that take a_pess (rule C only)

for these policies:
    candidate   pi(s, z) for every row of the --zs NPZ (cql_sweep.py), named as in
                analyze.selection ("z_r 0", "CQL 7.5", "T05 z_r", ...)
    switch      rule C of switch_eval.py at each kappa of --kappas between z_r and the
                CQL latent at --angle, with the --ensemble of the same checkpoint
    random      a_pess with probability p: the deviation is the expectation
                (1 - p) dev(a_opt) + p dev(a_pess)

Usage (repo root):
    PYTHONPATH=.:tasks/T06-latent-switching uv run python -m deviation \
        --checkpoint <fb.pt> --ensemble <ensemble.pt> --zs <zs_sweep.npz> \
        --dataset_path <dir>/walker/rnd/dataset.npz --out <deviation.csv>
"""

import csv
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from agents.base import OfflineReplayBuffer, load_agent
from ensemble import load_ensemble
from switch_eval import KAPPAS, choose_pess, latents, values
from zero_shot_rl.eval.env import policy

FIELDS = ["kind", "name", "kappa", "p", "dev_data", "dev_opt", "pess_frac"]


def candidate_name(family, target_angle):
    return family if np.isnan(target_angle) else f"{family} {target_angle:g}"


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ensemble", type=str, required=True)
    parser.add_argument("--zs", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--task", type=str, default="walk")
    parser.add_argument("--angle", type=float, default=7.5)
    parser.add_argument("--kappas", nargs="*", type=float, default=KAPPAS)
    parser.add_argument("--random_ps", nargs="*", type=float, default=[0.5])
    parser.add_argument("--n_states", type=int, default=10000)
    parser.add_argument("--transitions", type=int, default=100000)
    args = parser.parse_args()

    cpu = torch.device("cpu")
    agent = load_agent(Path(args.checkpoint), cpu).eval()
    ensemble = load_ensemble(Path(args.ensemble), cpu).eval()
    buffer = OfflineReplayBuffer(
        reward_constructor=None, dataset_path=Path(args.dataset_path), discount=0.98,
        device=cpu, task=args.task, relabel=False, transitions=args.transitions,
    )
    idx = np.random.default_rng(0).choice(
        len(buffer.storage["observations"]), args.n_states, replace=False
    )
    s = buffer.storage["observations"][idx].float()
    a_data = buffer.storage["actions"][idx].float()
    n = len(s)

    def act(z):
        return policy(agent, s, torch.as_tensor(z, dtype=torch.float32).expand(n, -1))

    def dev(a, ref):
        return ((a - ref) ** 2).sum(1)

    zs = np.load(args.zs)
    z_r, z_cql = latents(zs, args.task, args.angle)
    rows = []
    with torch.no_grad():
        a_opt, a_pess = act(z_r), act(z_cql)
        for i in np.flatnonzero(zs["task"] == args.task):
            a = act(zs["z"][i])
            rows.append({"kind": "candidate",
                         "name": candidate_name(zs["family"][i], zs["target_angle"][i]),
                         "dev_data": dev(a, a_data).mean().item(),
                         "dev_opt": dev(a, a_opt).mean().item()})

        z_r_n = torch.as_tensor(z_r, dtype=torch.float32).expand(n, -1)
        q_opt, u = values(ensemble, s, a_opt, z_r_n)
        q_pess, _ = values(ensemble, s, a_pess, z_r_n)
        for kappa in args.kappas:
            take = torch.as_tensor(choose_pess(q_opt, u, q_pess, kappa))[:, None]
            a = torch.where(take, a_pess, a_opt)
            rows.append({"kind": "switch", "name": f"switch {kappa:g}", "kappa": kappa,
                         "dev_data": dev(a, a_data).mean().item(),
                         "dev_opt": dev(a, a_opt).mean().item(),
                         "pess_frac": take.float().mean().item()})

        d_opt, d_pess = dev(a_opt, a_data).mean().item(), dev(a_pess, a_data).mean().item()
        d_pess_opt = dev(a_pess, a_opt).mean().item()
        for p in args.random_ps:
            rows.append({"kind": "random", "name": f"random {p:g}", "p": p,
                         "dev_data": (1 - p) * d_opt + p * d_pess, "dev_opt": p * d_pess_opt,
                         "pess_frac": p})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{row[k]:.6g}" if isinstance(row.get(k), float) else row.get(k, "")
                             for k in FIELDS})


if __name__ == "__main__":
    main()
