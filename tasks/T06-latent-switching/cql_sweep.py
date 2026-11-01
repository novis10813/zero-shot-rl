"""CQL latents of a fixed standard FB on a finer angle grid, in the NPZ format of
zero_shot_rl.latent.infer, for zero_shot_rl.eval.latents.

For every --tasks task, z_r and the CQL latent of zero_shot_rl.latent.implicit at each
--angles degrees from z_r are solved as in zero_shot_rl.latent.infer. A re-solve on
another copy of the dataset differs from the T05 latents by about 2 degrees, the size of
the small angles here, so the swept latents are compared only with the z_r of the same
solve. The rows of --zs (a T05 NPZ) for z_r and CQL at --stored_angles are appended with
family "T05 z_r" and "T05 CQL", to tie the rollouts to the T05 ones.

Usage (repo root):
    PYTHONPATH=.:tasks/T06-latent-switching uv run python -m cql_sweep \
        --checkpoint <fb.pt> --dataset_path <dataset.npz> --zs <zs.npz> --out <zs_sweep.npz>
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from zero_shot_rl.latent.implicit import Simulator
from zero_shot_rl.latent.infer import build_solver, solver_arguments
from zero_shot_rl.latent.solver import angle, normalized

ANGLES = [1, 2.5, 5, 7.5, 10, 15, 20]
STORED_ANGLES = [5, 10]
FIELDS = ("task", "family", "strength", "target_angle", "angle", "converged", "iters",
          "multi", "z")


def stored_rows(stored, tasks, angles):
    """Indices of the z_r and CQL rows of a T05 NPZ for the tasks and target angles."""
    rows = []
    for task in tasks:
        in_task = stored["task"] == task
        rows += list(np.flatnonzero(in_task & (stored["family"] == "z_r")))
        for target in angles:
            rows += list(np.flatnonzero(in_task & (stored["family"] == "CQL")
                                        & (stored["target_angle"] == target)))
    return rows


def main():
    parser = ArgumentParser()
    solver_arguments(parser)
    parser.add_argument("--zs", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--tasks", nargs="+", default=["walk"])
    parser.add_argument("--angles", nargs="+", type=float, default=ANGLES)
    parser.add_argument("--stored_angles", nargs="+", type=float, default=STORED_ANGLES)
    args = parser.parse_args()

    sim = Simulator(args.workers)  # spawned before CUDA is initialised
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent, _, _, _, iz = build_solver(args, sim, device)
    d = iz.d

    rows = []

    def add(task, family, z_hat, z_hat_r, strength=0.0, target=0.0, converged=True, iters=0,
            multi=False):
        z = normalized(z_hat, d)
        rows.append({
            "task": task, "family": family, "strength": float(strength),
            "target_angle": float(target), "angle": angle(z, normalized(z_hat_r, d)),
            "converged": converged, "iters": iters, "multi": multi, "z": z.cpu().numpy(),
        })
        print(f"{task:5s} {family:8s} target {target:5.1f} angle {rows[-1]['angle']:6.2f} "
              f"strength {strength:+.4g} iters {iters} converged {converged}", flush=True)

    for task in args.tasks:
        z_hat_r = iz.z_hat_r(task)
        iz.set_task(z_hat_r)
        add(task, "z_r", z_hat_r, z_hat_r)
        grid = iz.angle_grid("CQL", 1, z_hat_r)
        for target in args.angles:
            match = iz.match_angle("CQL", 1, target, z_hat_r, grid)
            if match is None or isinstance(match, str):
                raise RuntimeError(f"{task} CQL {target} degrees: {match or 'unreachable'}")
            s, init = match
            z_hat, converged, iters = iz.solve("CQL", s, z_hat_r)
            multi = abs(angle(normalized(z_hat, d), normalized(z_hat_r, d)) - target) > 1.0
            if multi:
                z_hat, converged, _ = iz.solve("CQL", s, z_hat_r, init=init)
            add(task, "CQL", z_hat, z_hat_r, s, target, converged, iters, multi)

    stored = np.load(args.zs)
    solved_z_r = {r["task"]: torch.as_tensor(r["z"]) for r in rows if r["family"] == "z_r"}
    for i in stored_rows(stored, args.tasks, args.stored_angles):
        rows.append({k: stored[k][i] for k in FIELDS} | {"family": f"T05 {stored['family'][i]}"})
        task = str(stored["task"][i])
        print(f"{task:5s} {rows[-1]['family']:8s} target {stored['target_angle'][i]:5.1f} "
              f"angle to the solved z_r "
              f"{angle(torch.as_tensor(stored['z'][i]), solved_z_r[task]):6.2f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: (np.stack if k == "z" else np.array)([r[k] for r in rows])
                     for k in FIELDS},
             u_1=iz.u_1.cpu().numpy(), checkpoint=args.checkpoint)
    print(f"wrote {len(rows)} latents to {out}")
    sim.pool.close()


if __name__ == "__main__":
    main()
