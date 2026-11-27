"""Collect the final task latents z found by T6.2 runs into one npz for the z-reward analysis.

Each run directory `<artifacts_dir>/zero-shot-rl-<id>/` holds `run.log` (or `output.log`) and
`models/<task>_final.pt`, whose `meta["z"]` is the latent after 800k env steps. The run's
condition, OOD environment and finetune seed are read from the `Wrote out/<cond>_<env>_ft<seed>`
line of its log.

Output `data/inputs/found_z.npz` holds `z` [conditions, envs, tasks, seeds, d_z] and the label
arrays `conditions`, `envs`, `tasks`, `seeds`.

Usage (repo root):
    PYTHONPATH=.:tasks/T6.3-sensors uv run python -m analyze.extract_found_z \\
        --artifacts_dir <T6.2 artifacts dir> --out tasks/T6.3-sensors/data/inputs/found_z.npz
"""

import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from zero_shot_rl.provenance import no_clobber

CONDITIONS = ("cmaes_anfis", "cmaes_z")
ENVS = ("friction_mild", "friction_high_mild", "actuator_mild", "actuator_high_mild")
TASKS = ("stand", "walk", "run", "flip")


def discover(artifacts_dir: Path, conditions: Sequence[str], envs: Sequence[str]) -> Dict[Tuple[str, str, int], Path]:
    """Map (condition, env, seed) -> models dir, from the `Wrote out/...` line of each run log."""
    pattern = re.compile(r"Wrote out/(" + "|".join(conditions) + r")_(" + "|".join(envs) + r")_ft(\d+)\b")
    found: Dict[Tuple[str, str, int], Path] = {}
    for run in sorted(artifacts_dir.glob("zero-shot-rl-*")):
        for log in (run / "run.log", run / "output.log"):
            if not log.exists():
                continue
            m = pattern.search(log.read_text(errors="ignore"))
            if m:
                key = (m.group(1), m.group(2), int(m.group(3)))
                if key in found:
                    raise ValueError(f"duplicate run for {key}: {found[key].parent} and {run}")
                found[key] = run / "models"
            break
    return found


def collect(found: Dict[Tuple[str, str, int], Path], conditions, envs, tasks) -> Tuple[np.ndarray, np.ndarray]:
    seeds = sorted({s for (_, _, s) in found})
    z = np.full((len(conditions), len(envs), len(tasks), len(seeds), 0), np.nan)
    for ci, c in enumerate(conditions):
        for ei, e in enumerate(envs):
            for si, s in enumerate(seeds):
                models = found.get((c, e, s))
                if models is None:
                    raise ValueError(f"missing run for {(c, e, s)}")
                for ti, t in enumerate(tasks):
                    ckpt = torch.load(models / f"{t}_final.pt", map_location="cpu", weights_only=False)
                    vec = ckpt["meta"]["z"].double().numpy()
                    if z.shape[-1] == 0:
                        z = np.full(z.shape[:-1] + (vec.shape[0],), np.nan)
                    z[ci, ei, ti, si] = vec
    return z, np.array(seeds)


def main(argv: Sequence[str] = None) -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("tasks/T6.3-sensors/data/inputs/found_z.npz"))
    args = parser.parse_args(argv)
    found = discover(args.artifacts_dir, CONDITIONS, ENVS)
    z, seeds = collect(found, CONDITIONS, ENVS, TASKS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with no_clobber(args.out) as tmp:
        with open(tmp, "wb") as f:
            np.savez(f, z=z, conditions=np.array(CONDITIONS), envs=np.array(ENVS), tasks=np.array(TASKS), seeds=seeds)
    print(f"wrote {args.out}: z {z.shape}")


if __name__ == "__main__":
    main()
