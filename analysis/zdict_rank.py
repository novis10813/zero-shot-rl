"""Singular values of the ANFIS rule dictionary Z_dict = B^T Psi / N for several sensor sets.

With membership parameters fixed, an ANFIS reward with identity output activation is linear in
its consequent parameters:

    r(s) = sum_k w_k(s) (W_k x(s) + b_k)

where w_k are normalised rule firing strengths and x are the normalised sensors. The basis
functions are Psi(s) = [w_k(s), w_k(s) x_i(s)] for every rule k (first_order) or [w_k(s)]
(zero_order). Every task latent the ANFIS can produce is then Z_dict @ theta (before the
sqrt(d_z) normalisation), so the column space of Z_dict is the set of reachable latents.

For each ANFIS config the script reports the numerical rank, entropy effective rank and
condition number of Z_dict, and the relative least-squares residual of each task's z_r when
projected onto the column space. Membership parameters are taken from `init_from_data`, the
same initialisation `zero_shot_rl.finetune.anfis_z` uses.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m analysis.zdict_rank \\
        --checkpoint <fb.pt> --dataset_path datasets/walker/rnd/dataset.npz \\
        --anfis_configs configs/anfis/walker.yaml tasks/T6.3-sensors/configs/<set>.yaml \\
        --out_dir tasks/T6.3-sensors/data
"""

import csv
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from agents.base import load_agent
from zero_shot_rl.env.sensors import DOMAIN_TASKS
from zero_shot_rl.finetune.anfis_z import collect_reference_set, load_anfis_config, resolve_anfis_sensors
from zero_shot_rl.latent.anfis import ANFIS, FirstOrderConsequent, ZeroOrderConsequent, weighted_z
from zero_shot_rl.provenance import no_clobber


@torch.no_grad()
def rule_basis(model: ANFIS, features: torch.Tensor) -> torch.Tensor:
    """Basis functions Psi `[N, P]` such that `model(features) == Psi @ theta` for consequent params."""
    if model.output_activation != "identity":
        raise ValueError(f"output_activation must be identity, got {model.output_activation!r}")
    xn = model.normalize(features)
    w = model.firing_weights(features)
    if isinstance(model.consequent, ZeroOrderConsequent):
        return w
    if isinstance(model.consequent, FirstOrderConsequent):
        x1 = torch.cat([xn, torch.ones_like(xn[:, :1])], dim=1)
        return (w.unsqueeze(2) * x1.unsqueeze(1)).reshape(len(features), -1)
    raise ValueError(f"unsupported consequent {type(model.consequent).__name__}")


def spectrum_stats(sigma: np.ndarray, tol: float) -> Dict[str, float]:
    """Numerical rank (sigma_i > tol * sigma_1), entropy effective rank and condition number."""
    p = sigma / sigma.sum()
    p = p[p > 0]
    return {
        "numeric_rank": int((sigma > tol * sigma[0]).sum()),
        "effective_rank": float(np.exp(-(p * np.log(p)).sum())),
        "condition_number": float(sigma[0] / sigma[-1]) if sigma[-1] > 0 else float("inf"),
    }


def projection_residual(z_dict: np.ndarray, z: np.ndarray) -> float:
    """Relative residual ||Z theta* - z|| / ||z|| of the least-squares fit of z on the columns of Z."""
    theta, *_ = np.linalg.lstsq(z_dict, z, rcond=None)
    return float(np.linalg.norm(z_dict @ theta - z) / np.linalg.norm(z))


def analyse(
    model: ANFIS, features: torch.Tensor, backward_repr: torch.Tensor, z_tasks: Dict[str, torch.Tensor], tol: float
) -> Dict[str, object]:
    psi = rule_basis(model, features)
    z_dict = (backward_repr.T @ psi / len(features)).double().cpu().numpy()
    sigma = np.linalg.svd(z_dict, compute_uv=False)
    row: Dict[str, object] = {
        "num_sensors": model.in_dim,
        "num_rules": model.num_rules,
        "num_columns": z_dict.shape[1],
        **spectrum_stats(sigma, tol),
    }
    for task, z in z_tasks.items():
        row[f"residual_{task}"] = projection_residual(z_dict, z.double().cpu().numpy())
    row["sigma"] = sigma
    return row


def main(argv: Sequence[str] = None) -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--anfis_configs", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset_transitions", type=int, default=100000)
    parser.add_argument("--z_inference_steps", type=int, default=10000)
    parser.add_argument("--reference_seed", type=int, default=42)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)

    configs = [(Path(p).stem, load_anfis_config(p)) for p in args.anfis_configs]
    domains = {resolve_anfis_sensors(cfg)[0] for _, cfg in configs}
    if len(domains) != 1:
        raise ValueError(f"all ANFIS configs must share one domain, got {sorted(domains)}")
    domain = domains.pop()
    names_per_config = {name: resolve_anfis_sensors(cfg)[2] for name, cfg in configs}
    union: List[str] = []
    for names in names_per_config.values():
        union.extend(n for n in names if n not in union)

    tasks = list(DOMAIN_TASKS[domain])
    observations, features, rewards, _ = collect_reference_set(
        dataset_path=Path(args.dataset_path),
        domain=domain,
        sensor_names=union,
        tasks=tasks,
        dataset_transitions=args.dataset_transitions,
        z_inference_steps=args.z_inference_steps,
        seed=args.reference_seed,
        discount=args.discount,
        device=device,
    )
    agent = load_agent(Path(args.checkpoint), device)
    with torch.no_grad():
        backward_repr = agent.FB.backward_representation(observations).detach()
    z_tasks = {task: weighted_z(rewards[task], backward_repr) for task in tasks}

    rows, spectra = [], {}
    for name, cfg in configs:
        cols = [union.index(n) for n in names_per_config[name]]
        model = ANFIS.from_config(cfg, out_dim=1).to(device)
        model.init_from_data(features[:, cols])
        row = analyse(model, features[:, cols], backward_repr, z_tasks, args.tol)
        spectra[name] = row.pop("sigma")
        rows.append({"config": name, **row})
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with no_clobber(out_dir / "zdict_rank.csv") as tmp:
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with no_clobber(out_dir / "zdict_spectra.npz") as tmp:
        with open(tmp, "wb") as f:
            np.savez(f, **spectra)


if __name__ == "__main__":
    main()
