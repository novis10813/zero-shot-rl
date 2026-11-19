"""Optimize an ANFIS state-weighting function to infer better FB latents z via REINFORCE.

Given a frozen FB checkpoint and a YAML config that composes an ANFIS architecture
(zero_shot_rl.latent.anfis), each task runs:

    1. Reference pool: sample N transitions from the offline ExORL dataset, extract the
       configured Walker physics features Phi [N, D] (default: torso_height, torso_upright,
       horizontal_velocity, angmomentum), frozen backward representations B [N, d_z], and true
       task rewards r [N, 1].
    2. Optional warm-start: regress r from Phi so the initial ANFIS latent mu_z matches z_r.
    3. Episodic REINFORCE: perturb mu_z = weighted_z(ANFIS(Phi), B) with (antithetic) Gaussian
       noise projected onto the sqrt(d_z)-sphere, evaluate each z_k with the frozen FB policy in
       the simulator, and update ANFIS parameters from the normalised episode-return advantages.
    4. Periodic Table-6 evaluation: evaluate deterministic mu_z on fixed test seeds and write
       `<out_dir>/table6.csv` (step, stand, walk, run, flip; each cell is the IQM over
       `eval_episodes` rollouts), directly consumable by `analysis/table6_aggregate.py`.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.finetune.anfis_z \\
        --config configs/anfis_z.yaml \\
        --checkpoint artifacts/T4.1.3.1-residual-soft-q/zero-shot-rl-296/local-run-Y-11-12-09-01-22/660000.pt \\
        --dataset_path datasets/walker/rnd/dataset.npz \\
        --out_dir out/anfis_z_s0
"""

import copy
import csv
import math
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy import stats

from agents.base import load_agent
from agents.fb.agent import FB
from agents.fb.replay_buffer import FBReplayBuffer
from rewards import RewardFunctionConstructor
from zero_shot_rl.env.dynamics import load_dynamics
from zero_shot_rl.env.walker import STEPS, make_env, policy
from zero_shot_rl.latent.anfis import ANFIS, weighted_z
from zero_shot_rl.provenance import no_clobber, sha256
from zero_shot_rl.utils import set_seed_everywhere

ALL_TASKS = ["stand", "walk", "run", "flip"]

FEATURE_EXTRACTORS = {
    "torso_height": lambda physics: float(physics.torso_height()),
    "torso_upright": lambda physics: float(physics.torso_upright()),
    "horizontal_velocity": lambda physics: float(physics.horizontal_velocity()),
    "com_vx": lambda physics: float(physics.horizontal_velocity()),
    "CoM_vx": lambda physics: float(physics.horizontal_velocity()),
    "angmomentum": lambda physics: float(physics.angmomentum()),
    "angmom_y": lambda physics: float(physics.angmomentum()),
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "anfis": {
        "features": ["torso_height", "torso_upright", "horizontal_velocity", "angmomentum"],
        "num_rules": 8,
        "normalize_inputs": True,
        "temperature": None,
        "t_norm": "product",
        "normalize_firing": True,
        "output_activation": "identity",
        "membership": {
            "type": "gaussian",
            "min_sigma": 1e-3,
            "init_sigma": 1.0,
        },
        "consequent": {
            "type": "first_order",
        },
        "init_from_data": True,
    },
    "reference": {
        "dataset_transitions": 100000,
        "z_inference_steps": 10000,
        "seed": 42,
        "discount": 0.98,
    },
    "warmstart": {
        "enabled": True,
        "steps": 500,
        "lr": 1e-2,
        "weight_decay": 0.0,
    },
    "reinforce": {
        "iters": 50,
        "pop_size": 16,
        "antithetic": True,
        "sigma": 0.5,
        "lr": 1e-2,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "train_episodes_per_z": 1,
        "train_seed_offset": 10000,
        "seed": 0,
    },
    "eval": {
        "tasks": list(ALL_TASKS),
        "eval_every": 10,
        "eval_episodes": 10,
        "eval_seed_start": 0,
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    with open(path) as f:
        user_cfg = yaml.safe_load(f) or {}
    return deep_merge(DEFAULT_CONFIG, user_cfg)


def feature_dim(feature_names: Sequence[str], obs_dim: int = 24) -> int:
    dim = 0
    for name in feature_names:
        if name == "observations":
            dim += obs_dim
        elif name in FEATURE_EXTRACTORS:
            dim += 1
        else:
            known = sorted([*FEATURE_EXTRACTORS, "observations"])
            raise ValueError(f"unknown feature {name!r}; choose from {known}")
    if dim == 0:
        raise ValueError("anfis.features must be non-empty")
    return dim


def extract_physics_features(
    physics, feature_names: Sequence[str], observation: Optional[np.ndarray] = None
) -> np.ndarray:
    """Extract the configured feature vector from a MuJoCo Walker `physics` instance."""
    parts = []
    for name in feature_names:
        if name == "observations":
            if observation is None:
                raise ValueError("observation array is required when 'observations' is in features")
            parts.append(np.asarray(observation, dtype=np.float32).ravel())
        elif name in FEATURE_EXTRACTORS:
            parts.append(np.array([FEATURE_EXTRACTORS[name](physics)], dtype=np.float32))
        else:
            raise ValueError(f"unknown feature {name!r}")
    return np.concatenate(parts, axis=0)


def iqm(values: Sequence[float]) -> float:
    return float(stats.trim_mean(np.asarray(values, dtype=np.float64), 0.25))


def eval_points(iters: int, every: int) -> List[int]:
    if every <= 0:
        return sorted({0, iters})
    return sorted({*range(0, iters + 1, every), iters})


def collect_reference_set(
    dataset_path: Path,
    feature_names: Sequence[str],
    tasks: Sequence[str],
    dataset_transitions: int,
    z_inference_steps: int,
    seed: int,
    discount: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
    """Sample reference transitions from ExORL dataset and extract obs, physics features, and task rewards."""
    set_seed_everywhere(seed)
    rc = RewardFunctionConstructor(domain_name="walker", task_names=list(tasks), seed=seed, device=device)
    buffer = FBReplayBuffer(
        reward_constructor=rc,
        dataset_path=dataset_path,
        transitions=dataset_transitions,
        relabel=False,
        task=None,
        device=device,
        discount=discount,
        action_condition=None,
    )
    if z_inference_steps > len(buffer.storage["observations"]):
        raise ValueError(
            f"z_inference_steps ({z_inference_steps}) exceeds loaded transitions "
            f"({len(buffer.storage['observations'])})"
        )
    idx = torch.randint(0, len(buffer.storage["observations"]), (z_inference_steps,))
    batch_physics = buffer.storage["physics"][idx]
    batch_actions = buffer.storage["actions"][idx].detach().cpu().numpy()

    env = rc._env  # pylint: disable=protected-access
    obs_list, feat_list, rew_list = [], [], []
    for phys, act in zip(batch_physics, batch_actions):
        env.physics.set_state(phys)
        ts = env.step(act)
        obs = ts.observation["observations"]
        obs_list.append(obs)
        feat_list.append(extract_physics_features(env.physics, feature_names, obs))
        rew_list.append(rc(env.physics))

    observations = torch.as_tensor(np.asarray(obs_list), dtype=torch.float32, device=device)
    features = torch.as_tensor(np.asarray(feat_list), dtype=torch.float32, device=device)
    rew_arr = np.asarray(rew_list, dtype=np.float32)
    rewards = {
        task: torch.as_tensor(rew_arr[:, i : i + 1], dtype=torch.float32, device=device)
        for i, task in enumerate(tasks)
    }
    return observations, features, rewards, rc.reward_functions


def warmstart_anfis(
    model: ANFIS,
    features: torch.Tensor,
    target_rewards: torch.Tensor,
    backward_repr: torch.Tensor,
    z_r: torch.Tensor,
    steps: int,
    lr: float,
    weight_decay: float = 0.0,
) -> Dict[str, float]:
    """Supervised warm-start fitting ANFIS(features) to true task rewards."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = float("nan")
    for _ in range(steps):
        opt.zero_grad()
        pred = model(features)
        loss = F.mse_loss(pred, target_rewards)
        loss.backward()
        opt.step()
        mse = float(loss.item())
    model.eval()
    with torch.no_grad():
        mu_z = weighted_z(model(features), backward_repr)
        cos = float(F.cosine_similarity(mu_z.unsqueeze(0), z_r.unsqueeze(0)).item())
    return {"warmstart_mse": mse, "warmstart_cos": cos}


def sample_perturbations(
    mu_z: torch.Tensor, pop_size: int, sigma: float, antithetic: bool, generator: torch.Generator
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample K perturbations z_tilde around mu_z and their projections onto the sqrt(d_z)-sphere."""
    d_z = mu_z.shape[0]
    if antithetic:
        if pop_size % 2 != 0:
            raise ValueError(f"antithetic sampling requires even pop_size, got {pop_size}")
        half = torch.randn(pop_size // 2, d_z, generator=generator, device=mu_z.device, dtype=mu_z.dtype)
        eps = torch.cat([half, -half], dim=0)
    else:
        eps = torch.randn(pop_size, d_z, generator=generator, device=mu_z.device, dtype=mu_z.dtype)
    z_tilde = mu_z.detach().unsqueeze(0) + sigma * eps
    z_sphere = math.sqrt(d_z) * F.normalize(z_tilde, dim=-1, eps=1e-8)
    return z_tilde, z_sphere


def evaluate_zs(
    agent: FB,
    zs: torch.Tensor,
    env_seeds: Sequence[int],
    reward_fn,
    dynamics=None,
    steps: int = STEPS,
) -> np.ndarray:
    """Evaluate M latents `zs` [M, d_z] across `env_seeds` in lockstep with one batched actor call per step.

    Returns episode returns of shape [M, len(env_seeds)].
    """
    m, e = zs.shape[0], len(env_seeds)
    total = m * e
    z_batch = zs.repeat_interleave(e, dim=0)
    envs = [make_env(int(seed), dynamics) for _ in range(m) for seed in env_seeds]
    try:
        obs = np.stack([env.reset().observation["observations"] for env in envs])
        returns = np.zeros(total, dtype=np.float64)
        for _ in range(steps):
            s = torch.as_tensor(obs, dtype=torch.float32, device=z_batch.device)
            with torch.no_grad():
                actions = policy(agent, s, z_batch).cpu().numpy()
            for i, env in enumerate(envs):
                obs[i] = env.step(actions[i]).observation["observations"]
                returns[i] += reward_fn(env.physics)
        return returns.reshape(m, e)
    finally:
        for env in envs:
            env.close()


def reinforce_step(
    model: ANFIS,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    backward_repr: torch.Tensor,
    z_tilde: torch.Tensor,
    candidate_returns: np.ndarray,
    sigma: float,
    grad_clip: Optional[float] = 1.0,
) -> Dict[str, float]:
    """One episodic REINFORCE gradient step on ANFIS given perturbed latents `z_tilde` and their returns."""
    returns_t = torch.as_tensor(candidate_returns, dtype=features.dtype, device=features.device)
    mean_r = returns_t.mean()
    std_r = returns_t.std(unbiased=False)
    if float(std_r.item()) < 1e-8:
        return {
            "loss": 0.0,
            "mean_return": float(mean_r.item()),
            "std_return": float(std_r.item()),
            "max_return": float(returns_t.max().item()),
            "skipped": 1.0,
        }
    advantages = (returns_t - mean_r) / (std_r + 1e-8)

    model.train()
    optimizer.zero_grad()
    mu_z = weighted_z(model(features), backward_repr)
    # Minimising +A_k * ||z_tilde_k - mu_z||^2 / (2 sigma^2) moves mu_z toward high-advantage z_tilde_k.
    sq_dist = ((z_tilde.detach() - mu_z.unsqueeze(0)) ** 2).sum(dim=-1)
    loss = (advantages * sq_dist / (2.0 * (sigma ** 2))).mean()
    loss.backward()
    if grad_clip is not None and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    model.eval()
    return {
        "loss": float(loss.item()),
        "mean_return": float(mean_r.item()),
        "std_return": float(std_r.item()),
        "max_return": float(returns_t.max().item()),
        "skipped": 0.0,
    }


def train_task(
    task: str,
    task_index: int,
    agent: FB,
    features: torch.Tensor,
    backward_repr: torch.Tensor,
    target_rewards: torch.Tensor,
    reward_fn,
    cfg: Dict[str, Any],
    dynamics,
    out_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Warm-start and REINFORCE-optimize one ANFIS model for `task`."""
    anfis_cfg = cfg["anfis"]
    ws_cfg = cfg["warmstart"]
    rf_cfg = cfg["reinforce"]
    ev_cfg = cfg["eval"]

    seed = int(rf_cfg["seed"]) + task_index * 100
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    z_r = weighted_z(target_rewards, backward_repr).detach()
    eval_seeds = [int(ev_cfg["eval_seed_start"]) + i for i in range(int(ev_cfg["eval_episodes"]))]

    baseline_returns = evaluate_zs(agent, z_r.unsqueeze(0), eval_seeds, reward_fn, dynamics)[0]
    baseline_iqm = iqm(baseline_returns)
    print(f"[{task}] baseline z_r  eval IQM={baseline_iqm:.1f}  mean={baseline_returns.mean():.1f}", flush=True)

    model = ANFIS.from_config(anfis_cfg, in_dim=features.shape[1], out_dim=1).to(device)
    if anfis_cfg.get("init_from_data", True):
        model.init_from_data(features, seed=seed)

    ws_info = {"warmstart_mse": float("nan"), "warmstart_cos": float("nan")}
    if ws_cfg.get("enabled", True) and int(ws_cfg.get("steps", 0)) > 0:
        ws_info = warmstart_anfis(
            model,
            features,
            target_rewards,
            backward_repr,
            z_r,
            steps=int(ws_cfg["steps"]),
            lr=float(ws_cfg["lr"]),
            weight_decay=float(ws_cfg.get("weight_decay", 0.0)),
        )
        print(
            f"[{task}] warmstart ({ws_cfg['steps']} steps)  mse={ws_info['warmstart_mse']:.4g}  "
            f"cos(mu_z, z_r)={ws_info['warmstart_cos']:.4f}",
            flush=True,
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(rf_cfg["lr"]),
        weight_decay=float(rf_cfg.get("weight_decay", 0.0)),
    )

    iters = int(rf_cfg["iters"])
    eval_steps = set(eval_points(iters, int(ev_cfg["eval_every"])))
    pop_size = int(rf_cfg["pop_size"])
    sigma = float(rf_cfg["sigma"])
    antithetic = bool(rf_cfg.get("antithetic", True))
    grad_clip = rf_cfg.get("grad_clip", 1.0)
    train_eps = int(rf_cfg.get("train_episodes_per_z", 1))
    train_offset = int(rf_cfg.get("train_seed_offset", 10000))

    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    iqm_by_step: Dict[int, float] = {}
    best_iqm = -float("inf")
    best_state = None
    best_z = None

    def run_eval(step: int):
        nonlocal best_iqm, best_state, best_z
        model.eval()
        with torch.no_grad():
            mu_z = weighted_z(model(features), backward_repr)
            cos = float(F.cosine_similarity(mu_z.unsqueeze(0), z_r.unsqueeze(0)).item())
        rets = evaluate_zs(agent, mu_z.unsqueeze(0), eval_seeds, reward_fn, dynamics)[0]
        score = iqm(rets)
        iqm_by_step[step] = score
        for s_id, r in zip(eval_seeds, rets):
            eval_rows.append({"task": task, "step": step, "episode_seed": s_id, "return": float(r)})
        if score > best_iqm:
            best_iqm = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_z = mu_z.detach().cpu().clone()
        print(
            f"[{task}] eval step {step:4d}  IQM={score:6.1f}  mean={rets.mean():6.1f}  "
            f"delta_vs_zr={score - baseline_iqm:+6.1f}  cos={cos:.4f}",
            flush=True,
        )

    if 0 in eval_steps:
        run_eval(0)

    t0 = time.time()
    for step in range(1, iters + 1):
        model.eval()
        with torch.no_grad():
            mu_z = weighted_z(model(features), backward_repr)
        z_tilde, z_sphere = sample_perturbations(mu_z, pop_size, sigma, antithetic, gen)
        train_seeds = [train_offset + (step - 1) * train_eps + j for j in range(train_eps)]
        cand_returns = evaluate_zs(agent, z_sphere, train_seeds, reward_fn, dynamics).mean(axis=1)
        stats_step = reinforce_step(
            model, optimizer, features, backward_repr, z_tilde, cand_returns, sigma, grad_clip
        )
        with torch.no_grad():
            new_mu = weighted_z(model(features), backward_repr)
            cos = float(F.cosine_similarity(new_mu.unsqueeze(0), z_r.unsqueeze(0)).item())
        train_rows.append({
            "task": task,
            "step": step,
            "mean_return": stats_step["mean_return"],
            "std_return": stats_step["std_return"],
            "max_return": stats_step["max_return"],
            "cos_to_zr": cos,
            "loss": stats_step["loss"],
        })
        if step in eval_steps:
            run_eval(step)

    elapsed = time.time() - t0
    print(f"[{task}] done {iters} iters in {elapsed:.1f}s  best IQM={best_iqm:.1f} (baseline {baseline_iqm:.1f})", flush=True)

    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        final_z = weighted_z(model(features), backward_repr).detach().cpu()
    model.save(
        models_dir / f"{task}_final.pt",
        task=task,
        features=list(anfis_cfg["features"]),
        z=final_z,
        z_r=z_r.detach().cpu(),
        baseline_iqm=baseline_iqm,
        warmstart=ws_info,
    )
    if best_state is not None:
        best_model = ANFIS.from_config(anfis_cfg, in_dim=features.shape[1], out_dim=1)
        best_model.load_state_dict(best_state)
        best_model.save(
            models_dir / f"{task}_best.pt",
            task=task,
            features=list(anfis_cfg["features"]),
            z=best_z,
            z_r=z_r.detach().cpu(),
            best_iqm=best_iqm,
            baseline_iqm=baseline_iqm,
            warmstart=ws_info,
        )

    return {
        "task": task,
        "baseline_iqm": baseline_iqm,
        "iqm_by_step": iqm_by_step,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
    }


def write_table6_csv(path: Path, tasks: Sequence[str], iqm_per_task: Dict[str, Dict[int, float]]) -> None:
    """Write `step,<tasks...>` CSV matching `analysis/table6_aggregate.py`."""
    steps = sorted(iqm_per_task[tasks[0]].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", *tasks])
        writer.writeheader()
        for step in steps:
            row = {"step": step}
            for task in tasks:
                row[task] = f"{iqm_per_task[task][step]:.6f}"
            writer.writerow(row)


def main():
    parser = ArgumentParser(description="REINFORCE optimization of composable ANFIS -> FB latent z")
    parser.add_argument("--config", type=str, default=None, help="YAML config specifying ANFIS and REINFORCE setup")
    parser.add_argument("--checkpoint", type=str, required=True, help="Frozen FB checkpoint (.pt)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Offline ExORL Walker dataset (.npz)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory (refuses to overwrite existing table6.csv)")
    parser.add_argument("--dynamics", type=str, default=None, help="Optional shifted dynamics YAML (default: clean Walker)")
    parser.add_argument("--tasks", nargs="+", default=None, help="Override eval.tasks from config")
    parser.add_argument("--seed", type=int, default=None, help="Override reinforce.seed from config")
    parser.add_argument("--device", type=str, default=None, help="torch device (default: cuda if available else cpu)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.tasks is not None:
        cfg["eval"]["tasks"] = list(args.tasks)
    if args.seed is not None:
        cfg["reinforce"]["seed"] = int(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table6_path = out_dir / "table6.csv"
    if table6_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {table6_path}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    tasks = list(cfg["eval"]["tasks"])
    feature_names = list(cfg["anfis"]["features"])
    feature_dim(feature_names)  # validate early
    dynamics = load_dynamics(args.dynamics) if args.dynamics else None

    print(f"Loading FB checkpoint {args.checkpoint} on {device}", flush=True)
    agent = load_agent(Path(args.checkpoint), device)
    if not isinstance(agent, FB):
        raise ValueError(f"Expected FB checkpoint, got {type(agent).__name__}")
    agent.eval()

    ref_cfg = cfg["reference"]
    print(
        f"Collecting {ref_cfg['z_inference_steps']} reference transitions from {args.dataset_path} "
        f"(features={feature_names})",
        flush=True,
    )
    observations, features, rewards, reward_fns = collect_reference_set(
        dataset_path=Path(args.dataset_path),
        feature_names=feature_names,
        tasks=tasks,
        dataset_transitions=int(ref_cfg["dataset_transitions"]),
        z_inference_steps=int(ref_cfg["z_inference_steps"]),
        seed=int(ref_cfg["seed"]),
        discount=float(ref_cfg["discount"]),
        device=device,
    )
    with torch.no_grad():
        backward_repr = agent.FB.backward_representation(observations).detach()

    resolved = copy.deepcopy(cfg)
    resolved["provenance"] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(Path(args.checkpoint)),
        "dataset_path": str(args.dataset_path),
        "dynamics": args.dynamics,
    }
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    iqm_per_task: Dict[str, Dict[int, float]] = {}
    baseline_per_task: Dict[str, Dict[int, float]] = {}
    all_train_rows: List[Dict[str, Any]] = []
    all_eval_rows: List[Dict[str, Any]] = []

    for idx, task in enumerate(tasks):
        res = train_task(
            task=task,
            task_index=idx,
            agent=agent,
            features=features,
            backward_repr=backward_repr,
            target_rewards=rewards[task],
            reward_fn=reward_fns[task],
            cfg=cfg,
            dynamics=dynamics,
            out_dir=out_dir,
            device=device,
        )
        iqm_per_task[task] = res["iqm_by_step"]
        baseline_per_task[task] = {0: res["baseline_iqm"]}
        all_train_rows.extend(res["train_rows"])
        all_eval_rows.extend(res["eval_rows"])

    with no_clobber(table6_path) as tmp:
        write_table6_csv(tmp, tasks, iqm_per_task)
    write_table6_csv(out_dir / "baseline_table6.csv", tasks, baseline_per_task)

    with open(out_dir / "train.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["task", "step", "mean_return", "std_return", "max_return", "cos_to_zr", "loss"]
        )
        writer.writeheader()
        writer.writerows(all_train_rows)

    with open(out_dir / "eval_episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "step", "episode_seed", "return"])
        writer.writeheader()
        writer.writerows(all_eval_rows)

    print(f"Wrote {table6_path} (compatible with analysis.table6_aggregate)", flush=True)


if __name__ == "__main__":
    main()
