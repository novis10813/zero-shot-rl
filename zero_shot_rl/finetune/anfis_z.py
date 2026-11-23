"""Optimize FB task latents z via REINFORCE, LoLA, ReLA, CEM, or CMA-ES (ANFIS or direct z).

Given a frozen FB checkpoint and a YAML config:

    1. Reference pool: sample N transitions from the offline ExORL dataset, extract frozen
       backward representations B [N, d_z], true task rewards r [N, 1], and (when ``param=anfis``)
       the enabled scalar sensors Phi [N, D]. Compute both normalized ``z_r`` and reward-scaled
       ``w_r = E[r(s) B(s)]``.
    2. Initialization:
         - ``param=anfis``: initialize ANFIS from data and (optionally) supervised warm-start
           ANFIS(Phi) -> r so initial mu_z matches z_r.
         - ``param=z``: initialize free latent vector z on the sqrt(d_z)-sphere at z_r
           (or randomly when ``zero_shot_init=False``).
    3. Adaptation via ``algo``:
         - ``reinforce``: episodic REINFORCE with Gaussian perturbations on z.
         - ``lola``: Lookahead Latent Adaptation (Sikchi et al., 2025, arXiv:2504.07896, Alg. 2).
         - ``rela``: Residual Latent Adaptation (Sikchi et al., 2025, arXiv:2504.07896, Alg. 1).
         - ``cem``: Cross-Entropy Method in parameter space (full-episode or lookahead bootstrap).
         - ``cmaes``: CMA-ES / sep-CMA-ES in parameter space (full-episode or lookahead bootstrap).
       Stopping and periodic Table-6 evaluation are controlled by cumulative environment steps
       (``budget.max_env_steps`` and ``budget.eval_every_env_steps``). For backward compatibility,
       ``algo=reinforce`` without ``budget.max_env_steps`` retains the legacy ``reinforce.iters``
       / ``eval.eval_every`` schedule.
    4. Outputs: ``<out_dir>/table6.csv`` (directly consumable by ``analysis/table6_aggregate.py``),
       ``baseline_table6.csv``, ``train.csv``, ``eval_episodes.csv``, ``config_resolved.yaml``,
       and ``models/<task>_{best,final}.pt``.

Usage (repo root):
    PYTHONPATH=. MUJOCO_GL=egl uv run python -m zero_shot_rl.finetune.anfis_z \\
        --algo lola --param anfis \\
        --max_env_steps 200000 --eval_every_env_steps 10000 \\
        --anfis_config configs/anfis/walker.yaml \\
        --checkpoint <fb.pt> --dataset_path datasets/walker/rnd/dataset.npz \\
        --out_dir out/anfis_lola_s0
"""

import copy
import csv
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
from zero_shot_rl.env.sensors import (
    DOMAIN_TASKS,
    extract_sensors,
    validate_sensors,
)
from zero_shot_rl.finetune.anfis_updaters import (
    ALGO_TYPES,
    TRAIN_FIELDS,
    ANFISLatent,
    DirectZLatent,
    LatentParam,
    build_updater,
    evaluate_zs,
    load_latent_checkpoint,
    normalize_algo_type,
    normalize_param_type,
    reinforce_step,
    sample_perturbations,
    unnormalized_w_r,
)
from zero_shot_rl.latent.anfis import ANFIS, parse_sensor_specs, weighted_z
from zero_shot_rl.provenance import no_clobber, sha256
from zero_shot_rl.utils import BASE_DIR, set_seed_everywhere

# Re-exported for tests and callers importing directly from zero_shot_rl.finetune.anfis_z.
__all__ = [
    "DEFAULT_CONFIG",
    "collect_reference_set",
    "eval_points",
    "evaluate_zs",
    "iqm",
    "load_anfis_config",
    "load_config",
    "reinforce_step",
    "resolve_anfis_sensors",
    "sample_perturbations",
    "train_task",
    "warmstart_anfis",
    "write_table6_csv",
]

# Keys whose nested dicts should be replaced atomically rather than merged with defaults.
_REPLACE_KEYS = frozenset({"sensors", "default_membership", "consequent"})

DEFAULT_ENV_STEPS_BUDGET = 200000
DEFAULT_EVAL_EVERY_ENV_STEPS = 10000

DEFAULT_CONFIG: Dict[str, Any] = {
    "algo": "reinforce",
    "param": "anfis",
    "zero_shot_init": True,
    "budget": {
        "max_env_steps": None,
        "eval_every_env_steps": None,
    },
    "anfis": {
        "domain": "walker",
        "normalize_inputs": True,
        "temperature": None,
        "t_norm": "product",
        "normalize_firing": True,
        "max_rules": 256,
        "default_membership": {
            "type": "gaussian",
            "min_sigma": 1e-3,
            "init_sigma": 1.0,
        },
        "sensors": {
            "torso_height": {"enabled": True, "num_terms": 3},
            "torso_upright": {"enabled": True, "num_terms": 2},
            "horizontal_velocity": {"enabled": True, "num_terms": 3},
            "angmomentum": {"enabled": True, "num_terms": 2},
        },
        "consequent": {
            "type": "first_order",
        },
        "output_activation": "identity",
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
    "lola": {
        "horizon": 100,
        "num_start_states": 2,
        "num_latents": 5,
        "init_state_prob": 0.2,
        "bootstrap": True,
        "fb_q_head": "min",
        "discount": 0.98,
        "init_sigma": 0.5,
        "learn_std": True,
        "min_sigma": 1e-2,
        "max_sigma": 2.0,
        "lr": 0.05,
        "std_lr": None,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "normalize_advantages": False,
        "buffer_size": 100000,
        "train_seed_offset": 10000,
        "seed": 0,
    },
    "rela": {
        "use_residual": True,
        "hidden_dim": None,
        "residual_hidden_dim": 64,
        "scratch_hidden_dim": 1024,
        "critic_layers": 2,
        "fb_q_head": "min",
        "discount": 0.98,
        "tau": 0.005,
        "utd": 4,
        "actor_update_freq": 4,
        "batch_size": 256,
        "buffer_size": 300000,
        "exploration_std": 0.1,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.3,
        "warmup_steps": 256,
        "actor_warmup_steps": 0,
        "critic_lr": 1e-4,
        "actor_lr": 1e-4,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "log_every_env_steps": 1000,
        "train_seed_offset": 10000,
        "seed": 0,
    },
    "cem": {
        "pop_size": 16,
        "elite_frac": 0.25,
        "num_elites": None,
        "init_sigma": 0.3,
        "min_sigma": 1e-2,
        "max_sigma": 2.0,
        "alpha": 0.7,
        "antithetic": True,
        "param_scope": "all",
        "train_episodes_per_z": 1,
        "episode_steps": None,
        "horizon": None,
        "num_start_states": 2,
        "init_state_prob": 0.2,
        "bootstrap": True,
        "fb_q_head": "min",
        "discount": 0.98,
        "buffer_size": 100000,
        "train_seed_offset": 10000,
        "seed": 0,
    },
    "cmaes": {
        "pop_size": None,
        "init_sigma": 0.3,
        "min_sigma": 1e-4,
        "max_sigma": 5.0,
        "diagonal": False,
        "param_scope": "all",
        "train_episodes_per_z": 1,
        "episode_steps": None,
        "horizon": None,
        "num_start_states": 2,
        "init_state_prob": 0.2,
        "bootstrap": True,
        "fb_q_head": "min",
        "discount": 0.98,
        "buffer_size": 100000,
        "train_seed_offset": 10000,
        "seed": 0,
    },
    "eval": {
        "tasks": None,
        "eval_every": 10,
        "eval_episodes": 10,
        "eval_seed_start": 0,
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if k not in _REPLACE_KEYS and isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve_config_path(raw_path: Union[str, Path], relative_to: Optional[Path] = None) -> Path:
    p = Path(raw_path)
    if p.exists():
        return p
    if not p.is_absolute():
        if relative_to is not None and (relative_to / p).exists():
            return relative_to / p
        repo_candidate = Path(BASE_DIR) / p
        if repo_candidate.exists():
            return repo_candidate
    return p


def load_anfis_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a standalone ANFIS architecture/sensor YAML (``configs/anfis/<domain>.yaml``)."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"ANFIS config {path} must be a YAML mapping")
    if "anfis" in raw:
        anfis_block = raw["anfis"]
        if not isinstance(anfis_block, dict):
            raise ValueError(f"'anfis' section in {path} must be a mapping")
        return copy.deepcopy(anfis_block)
    return copy.deepcopy(raw)


def load_config(path: Optional[str] = None, anfis_config: Optional[str] = None) -> Dict[str, Any]:
    """Load and merge experiment config (``--config``) with ANFIS config (``--anfis_config``).

    Resolution order (later overrides earlier):
      1. DEFAULT_CONFIG
      2. External ANFIS YAML (``anfis_config`` argument, else ``anfis_config:`` key in ``path``)
      3. Experiment YAML (``path``, including optional partial ``anfis:`` overrides)
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    user_cfg: Dict[str, Any] = {}
    config_parent: Optional[Path] = None
    if path is not None:
        cfg_path = _resolve_config_path(path)
        config_parent = cfg_path.parent
        with open(cfg_path) as f:
            user_cfg = yaml.safe_load(f) or {}

    ext_anfis_raw = anfis_config if anfis_config is not None else user_cfg.get("anfis_config")
    if ext_anfis_raw is not None:
        ext_path = _resolve_config_path(ext_anfis_raw, relative_to=config_parent)
        cfg["anfis"] = deep_merge(cfg["anfis"], load_anfis_config(ext_path))

    user_overrides = {k: v for k, v in user_cfg.items() if k != "anfis_config"}
    cfg = deep_merge(cfg, user_overrides)

    cfg["algo"] = normalize_algo_type(cfg.get("algo", "reinforce"))
    cfg["param"] = normalize_param_type(cfg.get("param", "anfis"))

    domain = cfg["anfis"].get("domain", "walker")
    if cfg["eval"].get("tasks") is None:
        if domain in DOMAIN_TASKS:
            cfg["eval"]["tasks"] = list(DOMAIN_TASKS[domain])
    return cfg


def resolve_anfis_sensors(anfis_cfg: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    """Validate and resolve ``(domain, enabled_sensor_specs, enabled_sensor_names)`` from ``anfis:`` config."""
    domain = str(anfis_cfg.get("domain", "walker"))
    sensor_specs = parse_sensor_specs(anfis_cfg)
    sensor_names = validate_sensors(domain, [s["name"] for s in sensor_specs])
    return domain, sensor_specs, sensor_names


def iqm(values: Sequence[float]) -> float:
    return float(stats.trim_mean(np.asarray(values, dtype=np.float64), 0.25))


def eval_points(iters: int, every: int) -> List[int]:
    if every <= 0:
        return sorted({0, iters})
    return sorted({*range(0, iters + 1, every), iters})


def collect_reference_set(
    dataset_path: Path,
    domain: str,
    sensor_names: Optional[Sequence[str]],
    tasks: Sequence[str],
    dataset_transitions: int,
    z_inference_steps: int,
    seed: int,
    discount: float,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
    """Sample reference transitions from an ExORL dataset and extract obs, optional sensors, and rewards."""
    extract_feats = bool(sensor_names)
    if extract_feats:
        validate_sensors(domain, sensor_names or ())
    set_seed_everywhere(seed)
    rc = RewardFunctionConstructor(domain_name=domain, task_names=list(tasks), seed=seed, device=device)
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
        obs_list.append(ts.observation["observations"])
        if extract_feats:
            feat_list.append(extract_sensors(env.physics, domain, sensor_names or ()))
        rew_list.append(rc(env.physics))

    observations = torch.as_tensor(np.asarray(obs_list), dtype=torch.float32, device=device)
    features = (
        torch.as_tensor(np.asarray(feat_list), dtype=torch.float32, device=device) if extract_feats else None
    )
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


def _resolve_budget(cfg: Dict[str, Any], algo: str) -> Dict[str, Any]:
    budget = dict(cfg.get("budget") or {})
    max_env_steps = budget.get("max_env_steps")
    eval_every_env_steps = budget.get("eval_every_env_steps")
    if algo in ("lola", "rela", "cem", "cmaes"):
        if max_env_steps is None:
            max_env_steps = DEFAULT_ENV_STEPS_BUDGET
        if eval_every_env_steps is None:
            eval_every_env_steps = DEFAULT_EVAL_EVERY_ENV_STEPS
    return {
        "max_env_steps": int(max_env_steps) if max_env_steps is not None else None,
        "eval_every_env_steps": int(eval_every_env_steps) if eval_every_env_steps is not None else None,
        "eval_every_iters": int(cfg["eval"]["eval_every"]),
    }


def _build_latent_param(
    *,
    param_type: str,
    task: str,
    seed: int,
    resume_dir: Optional[str],
    resume_step: int,
    cfg: Dict[str, Any],
    features: Optional[torch.Tensor],
    backward_repr: torch.Tensor,
    target_rewards: torch.Tensor,
    z_r: torch.Tensor,
    device: torch.device,
) -> Tuple[LatentParam, Dict[str, float]]:
    ws_info = {"warmstart_mse": float("nan"), "warmstart_cos": float("nan")}
    zero_shot_init = bool(cfg.get("zero_shot_init", True))

    if param_type == "anfis":
        if features is None:
            raise ValueError("param='anfis' requires reference sensor features")
        anfis_cfg = cfg["anfis"]
        ws_cfg = cfg["warmstart"]
        model = ANFIS.from_config(anfis_cfg, out_dim=1).to(device)
        if resume_dir:
            loaded, loaded_meta = load_latent_checkpoint(Path(resume_dir) / f"{task}_final.pt", device=device)
            if not isinstance(loaded, ANFIS):
                raise ValueError(f"Expected ANFIS checkpoint in {resume_dir}, got {type(loaded).__name__}")
            model.load_state_dict(loaded.state_dict())
            ws_info = loaded_meta.get("warmstart", ws_info)
            print(f"[{task}] resumed ANFIS from {resume_dir} at step {resume_step}", flush=True)
        elif anfis_cfg.get("init_from_data", True):
            model.init_from_data(features, seed=seed)

        if (
            not resume_dir
            and zero_shot_init
            and ws_cfg.get("enabled", True)
            and int(ws_cfg.get("steps", 0)) > 0
        ):
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
        return ANFISLatent(model, features, backward_repr), ws_info

    # param_type == "z"
    z_dim = int(backward_repr.shape[-1])
    if resume_dir:
        loaded, loaded_meta = load_latent_checkpoint(Path(resume_dir) / f"{task}_final.pt", device=device)
        if isinstance(loaded, DirectZLatent):
            param = DirectZLatent(z_dim=z_dim, init_z=loaded.compute_z(), device=device)
        elif "z" in loaded_meta and loaded_meta["z"] is not None:
            param = DirectZLatent(z_dim=z_dim, init_z=loaded_meta["z"], device=device)
        else:
            raise ValueError(f"Cannot resume direct z from {resume_dir}/{task}_final.pt")
        print(f"[{task}] resumed direct-z from {resume_dir} at step {resume_step}", flush=True)
        return param, ws_info

    init_z = z_r if zero_shot_init else None
    param = DirectZLatent(z_dim=z_dim, init_z=init_z, seed=seed, device=device)
    with torch.no_grad():
        cos = float(F.cosine_similarity(param.compute_z().unsqueeze(0), z_r.unsqueeze(0)).item())
    ws_info["warmstart_cos"] = cos
    print(f"[{task}] direct-z init (zero_shot_init={zero_shot_init})  cos(z, z_r)={cos:.4f}", flush=True)
    return param, ws_info


def train_task(
    task: str,
    task_index: int,
    domain: str,
    agent: FB,
    features: Optional[torch.Tensor],
    backward_repr: torch.Tensor,
    target_rewards: torch.Tensor,
    reward_fn,
    cfg: Dict[str, Any],
    dynamics,
    out_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Initialize and optimize one task latent (ANFIS or direct z) with the configured algo."""
    algo = normalize_algo_type(cfg.get("algo", "reinforce"))
    param_type = normalize_param_type(cfg.get("param", "anfis"))
    algo_cfg = cfg[algo]
    ev_cfg = cfg["eval"]
    budget_cfg = _resolve_budget(cfg, algo)

    seed = int(algo_cfg.get("seed", 0)) + task_index * 100
    resume_dir = algo_cfg.get("resume_models_dir") or cfg["reinforce"].get("resume_models_dir")
    resume_step = int(algo_cfg.get("resume_step", cfg["reinforce"].get("resume_step", 0))) if resume_dir else 0
    torch.manual_seed(seed + resume_step)

    z_r = weighted_z(target_rewards, backward_repr).detach()
    w_r = unnormalized_w_r(target_rewards, backward_repr).detach()
    eval_seeds = [int(ev_cfg["eval_seed_start"]) + i for i in range(int(ev_cfg["eval_episodes"]))]

    baseline_returns = evaluate_zs(agent, z_r.unsqueeze(0), eval_seeds, reward_fn, domain=domain, dynamics=dynamics)[0]
    baseline_iqm = iqm(baseline_returns)
    print(
        f"[{task}] ({algo}/{param_type}) baseline z_r  eval IQM={baseline_iqm:.1f}  "
        f"mean={baseline_returns.mean():.1f}",
        flush=True,
    )

    latent_param, ws_info = _build_latent_param(
        param_type=param_type,
        task=task,
        seed=seed,
        resume_dir=resume_dir,
        resume_step=resume_step,
        cfg=cfg,
        features=features,
        backward_repr=backward_repr,
        target_rewards=target_rewards,
        z_r=z_r,
        device=device,
    )

    eval_rows: List[Dict[str, Any]] = []
    iqm_by_step: Dict[int, float] = {}
    best_iqm = -float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_z: Optional[torch.Tensor] = None

    def run_eval(step_labels: Sequence[int]) -> None:
        nonlocal best_iqm, best_state, best_z
        latent_param.eval()
        with torch.no_grad():
            mu_z = latent_param.compute_z()
            cos = float(F.cosine_similarity(mu_z.unsqueeze(0), z_r.unsqueeze(0)).item())
        rets = evaluate_zs(agent, mu_z.unsqueeze(0), eval_seeds, reward_fn, domain=domain, dynamics=dynamics)[0]
        score = iqm(rets)
        for label in step_labels:
            iqm_by_step[label] = score
            for s_id, r in zip(eval_seeds, rets):
                eval_rows.append({"task": task, "step": label, "episode_seed": s_id, "return": float(r)})
        if score > best_iqm:
            best_iqm = score
            best_state = latent_param.snapshot_state()
            best_z = mu_z.detach().cpu().clone()
        unit = "env_steps" if budget_cfg["max_env_steps"] is not None else "step"
        print(
            f"[{task}] eval {unit} {step_labels[-1]:6d}  IQM={score:6.1f}  mean={rets.mean():6.1f}  "
            f"delta_vs_zr={score - baseline_iqm:+6.1f}  cos={cos:.4f}",
            flush=True,
        )

    updater = build_updater(algo, cfg)
    t0 = time.time()
    train_rows = updater.run(
        task=task,
        domain=domain,
        agent=agent,
        latent_param=latent_param,
        z_r=z_r,
        w_r=w_r,
        reward_fn=reward_fn,
        dynamics=dynamics,
        seed=seed,
        eval_callback=run_eval,
        budget_cfg=budget_cfg,
        device=device,
    )
    elapsed = time.time() - t0
    final_env_steps = train_rows[-1]["env_steps"] if train_rows else 0
    print(
        f"[{task}] done ({algo}/{param_type}) env_steps={final_env_steps} in {elapsed:.1f}s  "
        f"best IQM={best_iqm:.1f} (baseline {baseline_iqm:.1f})",
        flush=True,
    )

    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    latent_param.eval()
    with torch.no_grad():
        final_z = latent_param.compute_z().detach().cpu()

    common_meta: Dict[str, Any] = {
        "domain": domain,
        "task": task,
        "algo": algo,
        "z_r": z_r.detach().cpu(),
        "w_r": w_r.detach().cpu(),
        "baseline_iqm": baseline_iqm,
        "warmstart": ws_info,
    }
    latent_param.save_checkpoint(models_dir / f"{task}_final.pt", z=final_z, **common_meta)

    if best_state is not None:
        if param_type == "anfis":
            best_anfis = ANFIS.from_config(cfg["anfis"], out_dim=1)
            best_anfis.load_state_dict(best_state)
            best_wrapper: LatentParam = ANFISLatent(best_anfis, features.cpu(), backward_repr.cpu())  # type: ignore[union-attr]
        else:
            best_wrapper = DirectZLatent(z_dim=int(backward_repr.shape[-1]))
            best_wrapper.load_snapshot(best_state)
        best_wrapper.save_checkpoint(
            models_dir / f"{task}_best.pt",
            z=best_z,
            best_iqm=best_iqm,
            **common_meta,
        )

    return {
        "task": task,
        "baseline_iqm": baseline_iqm,
        "iqm_by_step": iqm_by_step,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
    }


def write_table6_csv(path: Path, tasks: Sequence[str], iqm_per_task: Dict[str, Dict[int, float]]) -> None:
    """Write ``step,<tasks...>`` CSV matching ``analysis/table6_aggregate.py``."""
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
    parser = ArgumentParser(description="Optimize FB latent z via REINFORCE, LoLA, ReLA, CEM, or CMA-ES (ANFIS or direct z)")
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        choices=[*ALGO_TYPES, "cma-es"],
        help="Update algorithm: reinforce, lola, rela, cem, or cmaes (overrides config)",
    )
    parser.add_argument(
        "--param",
        type=str,
        default=None,
        choices=["anfis", "z", "direct_z"],
        help="Latent parameterization: anfis or z/direct_z (overrides config)",
    )
    parser.add_argument(
        "--max_env_steps",
        type=int,
        default=None,
        help="Total environment steps budget per task (overrides budget.max_env_steps)",
    )
    parser.add_argument(
        "--eval_every_env_steps",
        type=int,
        default=None,
        help="Evaluate deterministic policy every N environment steps (overrides budget.eval_every_env_steps)",
    )
    parser.add_argument(
        "--no_zero_shot_init",
        action="store_true",
        help="Disable zero-shot z_r initialization / ANFIS warmstart (ablation)",
    )
    parser.add_argument(
        "--anfis_config",
        type=str,
        default=None,
        help="Standalone ANFIS architecture/sensor YAML (e.g. configs/anfis/walker.yaml)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Experiment/finetune/eval YAML (e.g. tasks/T6-anfis/configs/reinforce.yaml)",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Frozen FB checkpoint (.pt)")
    parser.add_argument("--dataset_path", type=str, default=None, help="Offline ExORL dataset (.npz)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory (refuses to overwrite existing table6.csv)")
    parser.add_argument("--dynamics", type=str, default=None, help="Optional shifted dynamics YAML (Walker only)")
    parser.add_argument("--tasks", nargs="+", default=None, help="Override eval.tasks from config")
    parser.add_argument("--seed", type=int, default=None, help="Override algorithm seed from config")
    parser.add_argument(
        "--train_seed_offset",
        type=int,
        default=None,
        help="Override algorithm train_seed_offset from config",
    )
    parser.add_argument(
        "--eval_models_dir",
        type=str,
        default=None,
        help="Evaluate pre-trained checkpoints (<dir>/<task>_<tag>.pt) without finetuning",
    )
    parser.add_argument(
        "--eval_model_tag",
        type=str,
        default="final",
        choices=["final", "best"],
        help="Which saved checkpoint tag to evaluate with --eval_models_dir",
    )
    parser.add_argument(
        "--resume_models_dir",
        type=str,
        default=None,
        help="Continue from <dir>/<task>_final.pt (no warm-start), starting after --resume_step",
    )
    parser.add_argument(
        "--resume_step",
        type=int,
        default=None,
        help="Iterations already done in --resume_models_dir; steps resume_step+1..reinforce.iters are run",
    )
    parser.add_argument(
        "--eval_zr_only",
        action="store_true",
        help="Evaluate frozen FB baseline z_r only (no finetuning)",
    )
    parser.add_argument("--device", type=str, default=None, help="torch device (default: cuda if available else cpu)")
    args = parser.parse_args()

    if args.eval_models_dir is None and not args.dataset_path:
        parser.error("--dataset_path is required unless --eval_models_dir is specified")

    cfg = load_config(args.config, anfis_config=args.anfis_config)
    if args.algo is not None:
        cfg["algo"] = normalize_algo_type(args.algo)
    if args.param is not None:
        cfg["param"] = normalize_param_type(args.param)
    if args.max_env_steps is not None:
        cfg["budget"]["max_env_steps"] = int(args.max_env_steps)
    if args.eval_every_env_steps is not None:
        cfg["budget"]["eval_every_env_steps"] = int(args.eval_every_env_steps)
    if args.no_zero_shot_init:
        cfg["zero_shot_init"] = False
        cfg["warmstart"]["enabled"] = False
    if args.tasks is not None:
        cfg["eval"]["tasks"] = list(args.tasks)

    algo = cfg["algo"]
    param_type = cfg["param"]

    if args.seed is not None:
        cfg[algo]["seed"] = int(args.seed)
        cfg["reinforce"]["seed"] = int(args.seed)
    if args.train_seed_offset is not None:
        cfg[algo]["train_seed_offset"] = int(args.train_seed_offset)
        cfg["reinforce"]["train_seed_offset"] = int(args.train_seed_offset)
    if (args.resume_models_dir is None) != (args.resume_step is None):
        parser.error("--resume_models_dir and --resume_step must be given together")
    if args.resume_models_dir is not None:
        if algo == "reinforce" and cfg["budget"].get("max_env_steps") is None:
            if args.resume_step < 1 or args.resume_step >= int(cfg["reinforce"]["iters"]):
                parser.error("--resume_step must satisfy 1 <= resume_step < reinforce.iters")
        cfg[algo]["resume_models_dir"] = args.resume_models_dir
        cfg[algo]["resume_step"] = int(args.resume_step)

    # Validate domain and (if using ANFIS) sensor/architecture specs before loading checkpoints or datasets.
    if param_type == "anfis":
        domain, _, sensor_names = resolve_anfis_sensors(cfg["anfis"])
        dry_model = ANFIS.from_config(cfg["anfis"], out_dim=1)
    else:
        domain = str(cfg["anfis"].get("domain", "walker"))
        sensor_names = []
        dry_model = None

    if args.dynamics and domain != "walker":
        raise ValueError(f"--dynamics is only supported for domain 'walker', got {domain!r}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table6_path = out_dir / "table6.csv"
    if table6_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {table6_path}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    tasks = list(cfg["eval"]["tasks"])
    dynamics = load_dynamics(args.dynamics) if args.dynamics else None

    if dry_model is not None:
        print(
            f"Algo={algo}  Param={param_type}  Domain={domain}  sensors={sensor_names}  "
            f"terms={list(dry_model.term_counts)}  rules={dry_model.num_rules}",
            flush=True,
        )
    else:
        print(f"Algo={algo}  Param={param_type}  Domain={domain}", flush=True)
    print(f"Loading FB checkpoint {args.checkpoint} on {device}", flush=True)
    agent = load_agent(Path(args.checkpoint), device)
    if not isinstance(agent, FB):
        raise ValueError(f"Expected FB checkpoint, got {type(agent).__name__}")
    agent.eval()

    ref_cfg = cfg["reference"]
    resolved = copy.deepcopy(cfg)
    resolved["provenance"] = {
        "anfis_config": args.anfis_config,
        "config": args.config,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(Path(args.checkpoint)),
        "dataset_path": str(args.dataset_path) if args.dataset_path else None,
        "dynamics": args.dynamics,
        "eval_models_dir": args.eval_models_dir,
        "eval_model_tag": args.eval_model_tag if args.eval_models_dir else None,
    }
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    iqm_per_task: Dict[str, Dict[int, float]] = {}
    baseline_per_task: Dict[str, Dict[int, float]] = {}
    all_train_rows: List[Dict[str, Any]] = []
    all_eval_rows: List[Dict[str, Any]] = []

    if args.eval_models_dir is not None:
        models_dir = Path(args.eval_models_dir)
        ev_cfg = cfg["eval"]
        eval_seeds = [int(ev_cfg["eval_seed_start"]) + i for i in range(int(ev_cfg["eval_episodes"]))]
        rc = RewardFunctionConstructor(
            domain_name=domain,
            task_names=list(tasks),
            seed=int(ref_cfg["seed"]),
            device=device,
        )
        reward_fns = rc.reward_functions
        for task in tasks:
            ckpt_path = models_dir / f"{task}_{args.eval_model_tag}.pt"
            _, meta = load_latent_checkpoint(ckpt_path, device=device)
            z = meta["z"].to(device=device, dtype=torch.float32)
            rets = evaluate_zs(agent, z.unsqueeze(0), eval_seeds, reward_fns[task], domain=domain, dynamics=dynamics)[0]
            score = iqm(rets)
            iqm_per_task[task] = {0: score}
            for s_id, r in zip(eval_seeds, rets):
                all_eval_rows.append({"task": task, "step": 0, "episode_seed": s_id, "return": float(r)})

            if "z_r" in meta and meta["z_r"] is not None:
                z_r = meta["z_r"].to(device=device, dtype=torch.float32)
                base_rets = evaluate_zs(
                    agent, z_r.unsqueeze(0), eval_seeds, reward_fns[task], domain=domain, dynamics=dynamics
                )[0]
                base_iqm = iqm(base_rets)
                baseline_per_task[task] = {0: base_iqm}
                print(
                    f"[{task}] eval-only ({args.eval_model_tag})  IQM={score:6.1f}  "
                    f"baseline z_r IQM={base_iqm:6.1f}  delta={score - base_iqm:+6.1f}",
                    flush=True,
                )
            else:
                baseline_per_task[task] = {0: float("nan")}
                print(f"[{task}] eval-only ({args.eval_model_tag})  IQM={score:6.1f}", flush=True)
    else:
        print(
            f"Collecting {ref_cfg['z_inference_steps']} reference transitions from {args.dataset_path}",
            flush=True,
        )
        observations, features, rewards, reward_fns = collect_reference_set(
            dataset_path=Path(args.dataset_path),
            domain=domain,
            sensor_names=sensor_names if param_type == "anfis" else None,
            tasks=tasks,
            dataset_transitions=int(ref_cfg["dataset_transitions"]),
            z_inference_steps=int(ref_cfg["z_inference_steps"]),
            seed=int(ref_cfg["seed"]),
            discount=float(ref_cfg["discount"]),
            device=device,
        )
        with torch.no_grad():
            backward_repr = agent.FB.backward_representation(observations).detach()

        if args.eval_zr_only:
            ev_cfg = cfg["eval"]
            eval_seeds = [int(ev_cfg["eval_seed_start"]) + i for i in range(int(ev_cfg["eval_episodes"]))]
            for task in tasks:
                z_r = weighted_z(rewards[task], backward_repr).detach()
                rets = evaluate_zs(
                    agent, z_r.unsqueeze(0), eval_seeds, reward_fns[task], domain=domain, dynamics=dynamics
                )[0]
                score = iqm(rets)
                iqm_per_task[task] = {0: score}
                baseline_per_task[task] = {0: score}
                for s_id, r in zip(eval_seeds, rets):
                    all_eval_rows.append({"task": task, "step": 0, "episode_seed": s_id, "return": float(r)})
                print(f"[{task}] frozen FB z_r  eval IQM={score:6.1f}  mean={rets.mean():.1f}", flush=True)
        else:
            for idx, task in enumerate(tasks):
                res = train_task(
                    task=task,
                    task_index=idx,
                    domain=domain,
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
        writer = csv.DictWriter(f, fieldnames=list(TRAIN_FIELDS))
        writer.writeheader()
        writer.writerows(all_train_rows)

    with open(out_dir / "eval_episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "step", "episode_seed", "return"])
        writer.writeheader()
        writer.writerows(all_eval_rows)

    print(f"Wrote {table6_path} (compatible with analysis.table6_aggregate)", flush=True)


if __name__ == "__main__":
    main()
