"""Latent parameterizations and online/episodic updaters for FB latent adaptation.

Supports two latent parameterizations:
  - ``anfis``: Cartesian-grid ANFIS state-weighting function mapped through frozen B(s)
    into the sqrt(d_z)-sphere via ``weighted_z(ANFIS(Phi), B)``.
  - ``z`` (alias ``direct_z``): direct optimization of a free latent vector z on the
    sqrt(d_z)-sphere, as in Sikchi et al. (2025).

Supports three update algorithms:
  - ``reinforce``: episodic REINFORCE with (antithetic) Gaussian perturbations on z.
  - ``lola``: Lookahead Latent Adaptation (Sikchi et al., 2025, arXiv:2504.07896, Alg. 2) —
    short-horizon on-policy rollouts from arbitrary reset states with frozen FB terminal
    bootstrapping, RLOO baseline, and Gaussian latent policy N(mu, diag(sigma^2)).
  - ``rela``: Residual Latent Adaptation (Sikchi et al., 2025, arXiv:2504.07896, Alg. 1) —
    TD3-based off-policy actor-critic with small twin residual critics correcting the
    frozen FB base Q-function, backpropagating deterministic policy gradients into z / ANFIS.

All updaters can be budgeted and evaluated by cumulative environment steps (``max_env_steps``
and ``eval_every_env_steps``) for apples-to-apples sample-efficiency comparison.
"""

import copy
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.fb.agent import FB
from zero_shot_rl.env.sensors import make_domain_env
from zero_shot_rl.env.walker import policy
from zero_shot_rl.latent.anfis import ANFIS, load_anfis, weighted_z

PARAM_TYPES: Tuple[str, ...] = ("anfis", "z")
PARAM_SCOPES: Tuple[str, ...] = ("all", "consequent")
ALGO_TYPES: Tuple[str, ...] = ("reinforce", "lola", "rela")

TRAIN_FIELDS: Tuple[str, ...] = (
    "task",
    "step",
    "env_steps",
    "mean_return",
    "std_return",
    "max_return",
    "cos_to_zr",
    "loss",
    "critic_loss",
    "sigma_mean",
)


def normalize_param_type(name: str) -> str:
    """Canonicalize latent parameterization name to ``'anfis'`` or ``'z'``."""
    key = str(name).strip().lower()
    if key in ("z", "direct_z", "direct", "latent"):
        return "z"
    if key == "anfis":
        return "anfis"
    raise ValueError(f"Unknown param type {name!r}; expected one of {PARAM_TYPES}")


def normalize_algo_type(name: str) -> str:
    """Canonicalize update algorithm name to ``'reinforce'``, ``'lola'``, or ``'rela'``."""
    key = str(name).strip().lower()
    if key in ALGO_TYPES:
        return key
    raise ValueError(f"Unknown algo {name!r}; expected one of {ALGO_TYPES}")


def project_sphere(z: torch.Tensor) -> torch.Tensor:
    """Project vectors along the last dimension onto the sphere of radius sqrt(d_z)."""
    d_z = z.shape[-1]
    return math.sqrt(d_z) * F.normalize(z, dim=-1, eps=1e-8)


def unnormalized_w_r(target_rewards: torch.Tensor, backward_repr: torch.Tensor) -> torch.Tensor:
    """Reward-scaled task vector w_r = E[r(s) B(s)] matching F(s, a, z)^T w_r to true reward units."""
    weights = target_rewards.reshape(-1)
    return (weights.unsqueeze(-1) * backward_repr).mean(dim=0)


def fb_q_base(
    agent: FB,
    obs: torch.Tensor,
    actions: torch.Tensor,
    z_cond: torch.Tensor,
    w_r: torch.Tensor,
    head: str = "min",
) -> torch.Tensor:
    """Frozen FB base Q-value F(s, a, z_cond)^T w_r in true reward units.

    Differentiable with respect to ``actions`` (and ``z_cond``) when called outside
    ``torch.no_grad()``.
    """
    if z_cond.ndim == 1:
        z_batch = z_cond.unsqueeze(0).expand(obs.shape[0], -1)
    else:
        z_batch = z_cond
    f1, f2 = agent.FB.forward_representation(observation=obs, action=actions, z=z_batch)
    q1 = f1 @ w_r
    q2 = f2 @ w_r
    if head == "min":
        return torch.minimum(q1, q2)
    if head == "mean":
        return 0.5 * (q1 + q2)
    raise ValueError(f"Unknown fb_q_head {head!r}; choose 'min' or 'mean'")


def set_env_physics(env: Any, physics_state: np.ndarray) -> None:
    """Reset env (preserving shifted dynamics base) and restore an exact MuJoCo physics state."""
    env.reset()
    with env.physics.reset_context():
        env.physics.set_state(physics_state)


class LatentParam(nn.Module):
    """Base interface mapping trainable parameters to a latent z on the sqrt(d_z)-sphere."""

    param_type: str
    z_dim: int

    def compute_z(self) -> torch.Tensor:
        raise NotImplementedError

    def post_step(self) -> None:
        """Optional projection applied after an optimizer step."""

    def param_tensors(self, scope: str = "all") -> List[nn.Parameter]:
        """Return trainable parameter tensors selected by ``scope`` ('all' or 'consequent')."""
        if scope not in PARAM_SCOPES:
            raise ValueError(f"Unknown param_scope {scope!r}; expected one of {PARAM_SCOPES}")
        return list(self.parameters())

    @torch.no_grad()
    def get_flat_params(self, scope: str = "all") -> torch.Tensor:
        """Return a 1D detached clone of parameters in ``scope``."""
        tensors = self.param_tensors(scope)
        return torch.cat([p.detach().reshape(-1) for p in tensors]).clone()

    @torch.no_grad()
    def set_flat_params(self, flat: torch.Tensor, scope: str = "all") -> None:
        """Write a 1D parameter vector back into ``scope`` tensors and apply ``post_step()``."""
        tensors = self.param_tensors(scope)
        expected = sum(p.numel() for p in tensors)
        flat_1d = flat.reshape(-1)
        if flat_1d.numel() != expected:
            raise ValueError(f"Expected flat param length {expected} for scope={scope!r}, got {flat_1d.numel()}")
        offset = 0
        for p in tensors:
            n = p.numel()
            p.data.copy_(flat_1d[offset : offset + n].to(device=p.device, dtype=p.dtype).view_as(p))
            offset += n
        self.post_step()

    @torch.no_grad()
    def eval_candidates_to_zs(self, candidates: torch.Tensor, scope: str = "all") -> torch.Tensor:
        """Map ``candidates [K, D]`` to latents ``[K, d_z]`` without mutating current parameters."""
        saved = self.get_flat_params(scope)
        zs: List[torch.Tensor] = []
        self.eval()
        try:
            for k in range(candidates.shape[0]):
                self.set_flat_params(candidates[k], scope)
                zs.append(self.compute_z().detach())
        finally:
            self.set_flat_params(saved, scope)
        return torch.stack(zs, dim=0)

    def snapshot_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def load_snapshot(self, state: Dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state)

    def save_checkpoint(self, path: Path, **meta: Any) -> None:
        raise NotImplementedError


class ANFISLatent(LatentParam):
    """ANFIS state-weighting parameterization: z = weighted_z(ANFIS(Phi), B)."""

    param_type = "anfis"

    def __init__(self, model: ANFIS, features: torch.Tensor, backward_repr: torch.Tensor):
        super().__init__()
        self.model = model
        self.z_dim = int(backward_repr.shape[-1])
        self.register_buffer("features", features)
        self.register_buffer("backward_repr", backward_repr)

    def compute_z(self) -> torch.Tensor:
        return weighted_z(self.model(self.features), self.backward_repr)

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        return self.model.parameters(recurse=recurse)

    def param_tensors(self, scope: str = "all") -> List[nn.Parameter]:
        if scope == "all":
            return list(self.model.parameters())
        if scope == "consequent":
            return list(self.model.consequent.parameters())
        raise ValueError(f"Unknown param_scope {scope!r}; expected one of {PARAM_SCOPES}")

    def snapshot_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def load_snapshot(self, state: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state)

    def save_checkpoint(self, path: Path, **meta: Any) -> None:
        self.model.save(
            path,
            param_type="anfis",
            sensors=list(self.model.sensor_names),
            term_counts=list(self.model.term_counts),
            num_rules=self.model.num_rules,
            **meta,
        )


class DirectZLatent(LatentParam):
    """Direct latent parameterization: free vector z projected onto the sqrt(d_z)-sphere."""

    param_type = "z"

    def __init__(
        self,
        z_dim: int,
        init_z: Optional[torch.Tensor] = None,
        seed: int = 0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        if init_z is not None:
            vec = project_sphere(init_z.detach().to(device=device, dtype=dtype).reshape(self.z_dim))
        else:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            vec = project_sphere(torch.randn(self.z_dim, generator=gen, dtype=dtype)).to(device=device)
        self.raw_z = nn.Parameter(vec.clone())

    def compute_z(self) -> torch.Tensor:
        return project_sphere(self.raw_z)

    @torch.no_grad()
    def post_step(self) -> None:
        self.raw_z.data.copy_(project_sphere(self.raw_z.data))

    def save_checkpoint(self, path: Path, **meta: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "param_type": "z",
            "z_dim": self.z_dim,
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "meta": meta,
        }
        torch.save(payload, path)


def load_latent_checkpoint(
    path: Path, device: torch.device = torch.device("cpu")
) -> Tuple[Optional[nn.Module], Dict[str, Any]]:
    """Load either an ANFIS checkpoint or a DirectZLatent checkpoint, returning (module, meta)."""
    blob = torch.load(Path(path), map_location=device)
    if blob.get("param_type") == "z" and "config" not in blob:
        param = DirectZLatent(z_dim=int(blob["z_dim"]), device=device)
        param.load_state_dict(blob["state_dict"])
        param.eval()
        return param, blob.get("meta", {})
    model, meta = load_anfis(Path(path), device=device)
    return model, meta


class EvalScheduler:
    """Schedules periodic deterministic evaluations by milestone thresholds (env_steps or iters)."""

    def __init__(self, milestones: Sequence[int], run_eval: Callable[[Sequence[int]], None]):
        self.milestones: List[int] = sorted(set(int(m) for m in milestones))
        self._run_eval = run_eval
        self._next_idx = 0
        self.evaluated: List[int] = []

    @classmethod
    def from_budget(
        cls,
        max_steps: int,
        eval_every: int,
        run_eval: Callable[[Sequence[int]], None],
        start_step: int = 0,
    ) -> "EvalScheduler":
        if eval_every <= 0:
            milestones = [start_step, max_steps]
        else:
            milestones = [*range(start_step, max_steps + 1, eval_every), max_steps]
        return cls(milestones, run_eval)

    def trigger_due(self, current: int) -> None:
        """Evaluate all milestones <= ``current`` with one rollout per crossed batch."""
        due: List[int] = []
        while self._next_idx < len(self.milestones) and self.milestones[self._next_idx] <= current:
            due.append(self.milestones[self._next_idx])
            self._next_idx += 1
        if due:
            self._run_eval(due)
            self.evaluated.extend(due)

    def finalize(self, current: int) -> None:
        """Ensure any remaining milestones are populated once training ends."""
        self.trigger_due(current)
        if self._next_idx < len(self.milestones):
            remaining = self.milestones[self._next_idx :]
            self._run_eval(remaining)
            self.evaluated.extend(remaining)
            self._next_idx = len(self.milestones)


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
    z_sphere = project_sphere(z_tilde)
    return z_tilde, z_sphere


def evaluate_zs_with_steps(
    agent: FB,
    zs: torch.Tensor,
    env_seeds: Sequence[int],
    reward_fn,
    domain: str = "walker",
    dynamics=None,
    steps: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Evaluate M latents across ``env_seeds`` in lockstep; return ``(returns [M, E], total_env_steps)``."""
    m, e = zs.shape[0], len(env_seeds)
    total = m * e
    z_batch = zs.repeat_interleave(e, dim=0)
    envs = [make_domain_env(domain, int(seed), dynamics=dynamics) for _ in range(m) for seed in env_seeds]
    try:
        timesteps = [env.reset() for env in envs]
        obs = np.stack([ts.observation["observations"] for ts in timesteps])
        done = np.zeros(total, dtype=bool)
        returns = np.zeros(total, dtype=np.float64)
        step = 0
        total_env_steps = 0
        while not done.all():
            if steps is not None and step >= steps:
                break
            s = torch.as_tensor(obs, dtype=torch.float32, device=z_batch.device)
            with torch.no_grad():
                actions = policy(agent, s, z_batch).cpu().numpy()
            for i, env in enumerate(envs):
                if done[i]:
                    continue
                ts = env.step(actions[i])
                total_env_steps += 1
                obs[i] = ts.observation["observations"]
                returns[i] += float(reward_fn(env.physics))
                if ts.last():
                    done[i] = True
            step += 1
        return returns.reshape(m, e), total_env_steps
    finally:
        for env in envs:
            env.close()


def evaluate_zs(
    agent: FB,
    zs: torch.Tensor,
    env_seeds: Sequence[int],
    reward_fn,
    domain: str = "walker",
    dynamics=None,
    steps: Optional[int] = None,
) -> np.ndarray:
    """Evaluate M latents ``zs`` [M, d_z] across ``env_seeds`` in lockstep. Returns ``[M, len(env_seeds)]``."""
    returns, _ = evaluate_zs_with_steps(
        agent, zs, env_seeds, reward_fn, domain=domain, dynamics=dynamics, steps=steps
    )
    return returns


def reinforce_param_step(
    latent_param: LatentParam,
    optimizer: torch.optim.Optimizer,
    z_tilde: torch.Tensor,
    candidate_returns: np.ndarray,
    sigma: float,
    grad_clip: Optional[float] = 1.0,
) -> Dict[str, float]:
    """One episodic REINFORCE gradient step on any ``LatentParam`` given perturbed latents and returns."""
    device = z_tilde.device
    dtype = z_tilde.dtype
    returns_t = torch.as_tensor(candidate_returns, dtype=dtype, device=device)
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

    latent_param.train()
    optimizer.zero_grad()
    mu_z = latent_param.compute_z()
    sq_dist = ((z_tilde.detach() - mu_z.unsqueeze(0)) ** 2).sum(dim=-1)
    loss = (advantages * sq_dist / (2.0 * (sigma ** 2))).mean()
    loss.backward()
    if grad_clip is not None and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(latent_param.parameters(), grad_clip)
    optimizer.step()
    latent_param.post_step()
    latent_param.eval()
    return {
        "loss": float(loss.item()),
        "mean_return": float(mean_r.item()),
        "std_return": float(std_r.item()),
        "max_return": float(returns_t.max().item()),
        "skipped": 0.0,
    }


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
    """Backward-compatible wrapper performing one REINFORCE step directly on an ``ANFIS`` model."""
    wrapper = ANFISLatent(model, features, backward_repr)
    return reinforce_param_step(wrapper, optimizer, z_tilde, candidate_returns, sigma, grad_clip)


class PhysicsStateBuffer:
    """Fixed-capacity ring buffer of ``(physics_state, flat_obs)`` pairs for LoLA resets."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._physics: Optional[np.ndarray] = None
        self._obs: Optional[np.ndarray] = None
        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, physics_state: np.ndarray, obs: np.ndarray) -> None:
        if self._physics is None:
            self._physics = np.empty((self.capacity, physics_state.shape[0]), dtype=physics_state.dtype)
            self._obs = np.empty((self.capacity, obs.shape[0]), dtype=np.float32)
        self._physics[self._ptr] = physics_state
        self._obs[self._ptr] = obs
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        assert self._physics is not None and self._obs is not None and self._size > 0
        idx = int(rng.integers(0, self._size))
        return self._physics[idx].copy(), self._obs[idx].copy()


class TransitionBuffer:
    """Fixed-capacity ring buffer of ``(s, a, r, s')`` transitions for ReLA."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.obs = np.empty((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((self.capacity,), dtype=np.float32)
        self.next_obs = np.empty((self.capacity, obs_dim), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray) -> None:
        self.obs[self._ptr] = obs
        self.actions[self._ptr] = action
        self.rewards[self._ptr] = float(reward)
        self.next_obs[self._ptr] = next_obs
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int, rng: np.random.Generator, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = rng.integers(0, self._size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.actions[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=device),
        )


class CandidateEvaluator:
    """Evaluates a batch of candidate latents ``zs [K, d_z]`` via full episodes or lookahead bootstrap.

    - ``horizon is None``: runs ``episodes_per_z`` full episodes (capped at ``episode_steps`` if set)
      per candidate using common episode seeds within each call.
    - ``horizon > 0``: samples ``num_start_states`` shared start states (initial reset with prob
      ``init_state_prob``, else from an online ``PhysicsStateBuffer``), rolls out all ``K``
      candidates in lockstep for ``horizon`` steps, optionally adds frozen FB terminal bootstrap
      ``gamma^horizon * F(s_n, pi_z(s_n), z)^T w_r``, and averages over the start states.
    """

    def __init__(
        self,
        *,
        agent: FB,
        domain: str,
        reward_fn,
        w_r: torch.Tensor,
        dynamics=None,
        episodes_per_z: int = 1,
        episode_steps: Optional[int] = None,
        horizon: Optional[int] = None,
        num_start_states: int = 2,
        init_state_prob: float = 0.2,
        bootstrap: bool = True,
        fb_q_head: str = "min",
        discount: float = 0.98,
        buffer_size: int = 100000,
        seed: int = 0,
        train_seed_offset: int = 10000,
    ):
        self.agent = agent
        self.domain = domain
        self.reward_fn = reward_fn
        self.w_r = w_r
        self.dynamics = dynamics
        self.episodes_per_z = int(episodes_per_z)
        self.episode_steps = int(episode_steps) if episode_steps is not None else None
        self.horizon = int(horizon) if horizon is not None else None
        self.num_start_states = int(num_start_states)
        self.init_state_prob = float(init_state_prob)
        self.bootstrap = bool(bootstrap)
        self.fb_q_head = str(fb_q_head)
        self.discount = float(discount)
        self.seed = int(seed)
        self.train_seed_offset = int(train_seed_offset)
        self._call_idx = 0
        self._rng = np.random.default_rng(self.seed + self.train_seed_offset)
        self.state_buffer = PhysicsStateBuffer(int(buffer_size))
        self._envs: List[Any] = []
        self._reset_env: Any = None
        self._env_k: int = 0

    def _ensure_lookahead_envs(self, k: int) -> None:
        if self._envs and self._env_k == k:
            return
        self.close()
        total = self.num_start_states * k
        base_seed = self.seed + self.train_seed_offset
        self._reset_env = make_domain_env(self.domain, base_seed, dynamics=self.dynamics)
        self._envs = [make_domain_env(self.domain, base_seed + 1 + i, dynamics=self.dynamics) for i in range(total)]
        self._env_k = k

    def evaluate(self, zs: torch.Tensor) -> Tuple[np.ndarray, int]:
        """Return ``(fitness [K], env_steps_consumed)`` for ``zs [K, d_z]``."""
        k = int(zs.shape[0])
        self._call_idx += 1
        if self.horizon is None:
            base = self.train_seed_offset + self.seed * 1000 + self._call_idx * self.episodes_per_z
            ep_seeds = [base + e for e in range(self.episodes_per_z)]
            rets, steps = evaluate_zs_with_steps(
                self.agent,
                zs,
                ep_seeds,
                self.reward_fn,
                domain=self.domain,
                dynamics=self.dynamics,
                steps=self.episode_steps,
            )
            return rets.mean(axis=1), steps

        m = self.num_start_states
        horizon = self.horizon
        self._ensure_lookahead_envs(k)
        assert self._reset_env is not None

        start_states: List[Tuple[np.ndarray, np.ndarray]] = []
        for _ in range(m):
            use_init = (len(self.state_buffer) == 0) or (float(self._rng.random()) < self.init_state_prob)
            if use_init:
                ts = self._reset_env.reset()
                phys = self._reset_env.physics.get_state().copy()
                obs0 = ts.observation["observations"].copy()
                self.state_buffer.add(phys, obs0)
                start_states.append((phys, obs0))
            else:
                start_states.append(self.state_buffer.sample(self._rng))

        total = m * k
        obs_batch = np.empty((total, start_states[0][1].shape[0]), dtype=np.float32)
        for i, (phys, obs0) in enumerate(start_states):
            for j in range(k):
                idx = i * k + j
                set_env_physics(self._envs[idx], phys)
                obs_batch[idx] = obs0

        # Candidate z_j is evaluated from every start state i -> repeat [k, d_z] across m blocks.
        z_flat = zs.unsqueeze(0).expand(m, k, -1).reshape(total, -1)
        returns = np.zeros(total, dtype=np.float64)
        done = np.zeros(total, dtype=bool)
        gamma_pow = 1.0
        steps_consumed = 0

        for _ in range(horizon):
            s_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=z_flat.device)
            with torch.no_grad():
                actions = policy(self.agent, s_t, z_flat).cpu().numpy()
            for idx, env in enumerate(self._envs):
                if done[idx]:
                    continue
                ts = env.step(actions[idx])
                steps_consumed += 1
                next_obs = ts.observation["observations"]
                obs_batch[idx] = next_obs
                returns[idx] += gamma_pow * float(self.reward_fn(env.physics))
                self.state_buffer.add(env.physics.get_state().copy(), next_obs.copy())
                if ts.last():
                    done[idx] = True
            gamma_pow *= self.discount
            if done.all():
                break

        if self.bootstrap:
            alive = ~done
            if alive.any():
                s_term = torch.as_tensor(obs_batch[alive], dtype=torch.float32, device=z_flat.device)
                z_term = z_flat[torch.as_tensor(alive, dtype=torch.bool, device=z_flat.device)]
                with torch.no_grad():
                    a_term = policy(self.agent, s_term, z_term)
                    v_term = fb_q_base(
                        self.agent, s_term, a_term, z_term, self.w_r, head=self.fb_q_head
                    ).cpu().numpy()
                returns[alive] += gamma_pow * v_term

        return returns.reshape(m, k).mean(axis=0), steps_consumed

    def close(self) -> None:
        for env in self._envs:
            env.close()
        self._envs = []
        self._env_k = 0
        if self._reset_env is not None:
            self._reset_env.close()
            self._reset_env = None


def _build_q_mlp(obs_dim: int, action_dim: int, hidden_dim: int, layers: int, zero_init_last: bool) -> nn.Sequential:
    mods: List[nn.Module] = []
    in_dim = obs_dim + action_dim
    for _ in range(layers):
        mods.append(nn.Linear(in_dim, hidden_dim))
        mods.append(nn.ReLU())
        in_dim = hidden_dim
    out_layer = nn.Linear(in_dim, 1)
    if zero_init_last:
        nn.init.zeros_(out_layer.weight)
        nn.init.zeros_(out_layer.bias)
    mods.append(out_layer)
    return nn.Sequential(*mods)


class ResidualCritic(nn.Module):
    """Twin MLP critics Q_1(s, a), Q_2(s, a) for ReLA (residual or from-scratch)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        layers: int = 2,
        zero_init_last: bool = True,
    ):
        super().__init__()
        self.q1 = _build_q_mlp(obs_dim, action_dim, hidden_dim, layers, zero_init_last)
        self.q2 = _build_q_mlp(obs_dim, action_dim, hidden_dim, layers, zero_init_last)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([obs, actions], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)


class ReinforceUpdater:
    """Episodic REINFORCE updater supporting both legacy iter mode and env_steps budget mode."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def run(
        self,
        *,
        task: str,
        domain: str,
        agent: FB,
        latent_param: LatentParam,
        z_r: torch.Tensor,
        w_r: torch.Tensor,  # unused by REINFORCE, kept for uniform updater signature
        reward_fn,
        dynamics,
        seed: int,
        eval_callback: Callable[[Sequence[int]], None],
        budget_cfg: Dict[str, Any],
        device: torch.device,
    ) -> List[Dict[str, Any]]:
        del w_r
        rf_cfg = self.cfg
        resume_step = int(rf_cfg.get("resume_step", 0)) if rf_cfg.get("resume_models_dir") else 0
        gen = torch.Generator(device=device).manual_seed(seed + resume_step)

        optimizer = torch.optim.Adam(
            latent_param.parameters(),
            lr=float(rf_cfg["lr"]),
            weight_decay=float(rf_cfg.get("weight_decay", 0.0)),
        )

        pop_size = int(rf_cfg["pop_size"])
        sigma = float(rf_cfg["sigma"])
        antithetic = bool(rf_cfg.get("antithetic", True))
        grad_clip = rf_cfg.get("grad_clip", 1.0)
        train_eps = int(rf_cfg.get("train_episodes_per_z", 1))
        train_offset = int(rf_cfg.get("train_seed_offset", 10000))
        max_ep_steps = rf_cfg.get("episode_steps")

        max_env_steps = budget_cfg.get("max_env_steps")
        eval_every_env_steps = budget_cfg.get("eval_every_env_steps")
        use_env_budget = max_env_steps is not None

        if use_env_budget:
            scheduler = EvalScheduler.from_budget(
                max_steps=int(max_env_steps),
                eval_every=int(eval_every_env_steps or 0),
                run_eval=eval_callback,
                start_step=0,
            )
            scheduler.trigger_due(0)
        else:
            iters = int(rf_cfg["iters"])
            eval_every_iters = int(budget_cfg["eval_every_iters"])
            if eval_every_iters <= 0:
                milestones = sorted({0, iters})
            else:
                milestones = sorted({*range(0, iters + 1, eval_every_iters), iters})
            if rf_cfg.get("resume_models_dir"):
                milestones = sorted({resume_step} | {m for m in milestones if m > resume_step})
            scheduler = EvalScheduler(milestones, eval_callback)
            if resume_step in scheduler.milestones:
                scheduler.trigger_due(resume_step)

        train_rows: List[Dict[str, Any]] = []
        env_steps = 0
        step = resume_step

        while True:
            if use_env_budget:
                if env_steps >= int(max_env_steps):
                    break
            else:
                if step >= int(rf_cfg["iters"]):
                    break
            step += 1

            latent_param.eval()
            with torch.no_grad():
                mu_z = latent_param.compute_z()
            z_tilde, z_sphere = sample_perturbations(mu_z, pop_size, sigma, antithetic, gen)
            train_seeds = [train_offset + (step - 1) * train_eps + j for j in range(train_eps)]
            ret_matrix, batch_env_steps = evaluate_zs_with_steps(
                agent,
                z_sphere,
                train_seeds,
                reward_fn,
                domain=domain,
                dynamics=dynamics,
                steps=max_ep_steps,
            )
            env_steps += batch_env_steps
            cand_returns = ret_matrix.mean(axis=1)
            stats_step = reinforce_param_step(
                latent_param, optimizer, z_tilde, cand_returns, sigma, grad_clip
            )
            with torch.no_grad():
                new_mu = latent_param.compute_z()
                cos = float(F.cosine_similarity(new_mu.unsqueeze(0), z_r.unsqueeze(0)).item())
            train_rows.append({
                "task": task,
                "step": step,
                "env_steps": env_steps,
                "mean_return": stats_step["mean_return"],
                "std_return": stats_step["std_return"],
                "max_return": stats_step["max_return"],
                "cos_to_zr": cos,
                "loss": stats_step["loss"],
                "critic_loss": float("nan"),
                "sigma_mean": sigma,
            })
            scheduler.trigger_due(env_steps if use_env_budget else step)

        if use_env_budget:
            scheduler.finalize(env_steps)
        return train_rows


def lola_rloo_advantages(returns: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """Compute Leave-One-Out advantages for ``returns`` of shape ``[m, k]`` (Sikchi et al., Eq. 7)."""
    k = returns.shape[-1]
    if k < 2:
        raise ValueError(f"LoLA RLOO requires num_latents k >= 2, got {k}")
    mean_r = returns.mean(dim=-1, keepdim=True)
    adv = (k / float(k - 1)) * (returns - mean_r)
    if normalize:
        std = adv.std(unbiased=False)
        if float(std.item()) > 1e-8:
            adv = adv / (std + 1e-8)
    return adv


def lola_update_step(
    latent_param: LatentParam,
    log_std: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    z_samples: torch.Tensor,
    returns: torch.Tensor,
    *,
    learn_std: bool = True,
    min_sigma: float = 1e-2,
    max_sigma: float = 2.0,
    grad_clip: Optional[float] = 1.0,
    normalize_advantages: bool = False,
) -> Dict[str, float]:
    """One LoLA RLOO score-function update on ``latent_param`` (and optionally ``log_std``).

    Args:
        z_samples: unprojected Gaussian draws ``[m, k, d_z]``.
        returns: lookahead returns ``[m, k]``.
    """
    adv = lola_rloo_advantages(returns.detach(), normalize=normalize_advantages)
    latent_param.train()
    optimizer.zero_grad()
    mu_z = latent_param.compute_z()
    log_s = log_std if learn_std else log_std.detach()
    log_s_clamped = log_s.clamp(math.log(min_sigma), math.log(max_sigma))
    var = torch.exp(2.0 * log_s_clamped)
    # Exact log N(z | mu, diag(sigma^2)) up to additive constant -0.5 * d_z * log(2 pi).
    diff_sq = (z_samples.detach() - mu_z.view(1, 1, -1)) ** 2
    log_prob = -0.5 * (diff_sq / var.view(1, 1, -1) + 2.0 * log_s_clamped.view(1, 1, -1)).sum(dim=-1)
    loss = -(adv * log_prob).mean()
    loss.backward()
    if grad_clip is not None and grad_clip > 0:
        params: List[nn.Parameter] = list(latent_param.parameters())
        if learn_std and isinstance(log_std, nn.Parameter):
            params.append(log_std)
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
    optimizer.step()
    latent_param.post_step()
    if learn_std:
        with torch.no_grad():
            log_std.data.clamp_(math.log(min_sigma), math.log(max_sigma))
    latent_param.eval()
    return {
        "loss": float(loss.item()),
        "mean_return": float(returns.mean().item()),
        "std_return": float(returns.std(unbiased=False).item()),
        "max_return": float(returns.max().item()),
        "sigma_mean": float(torch.exp(log_std.detach()).mean().item()),
    }


class LolaUpdater:
    """Lookahead Latent Adaptation (Sikchi et al., 2025, Algorithm 2)."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def run(
        self,
        *,
        task: str,
        domain: str,
        agent: FB,
        latent_param: LatentParam,
        z_r: torch.Tensor,
        w_r: torch.Tensor,
        reward_fn,
        dynamics,
        seed: int,
        eval_callback: Callable[[Sequence[int]], None],
        budget_cfg: Dict[str, Any],
        device: torch.device,
    ) -> List[Dict[str, Any]]:
        c = self.cfg
        max_env_steps = int(budget_cfg["max_env_steps"])
        eval_every_env_steps = int(budget_cfg.get("eval_every_env_steps") or 0)

        m = int(c.get("num_start_states", 2))
        k = int(c.get("num_latents", 5))
        if k < 2:
            raise ValueError(f"LoLA requires num_latents >= 2 for RLOO baseline, got {k}")
        horizon = int(c.get("horizon", 100))
        init_state_prob = float(c.get("init_state_prob", 0.2))
        bootstrap = bool(c.get("bootstrap", True))
        fb_q_head = str(c.get("fb_q_head", "min"))
        discount = float(c.get("discount", 0.98))
        init_sigma = float(c.get("init_sigma", 0.5))
        learn_std = bool(c.get("learn_std", True))
        min_sigma = float(c.get("min_sigma", 1e-2))
        max_sigma = float(c.get("max_sigma", 2.0))
        lr = float(c.get("lr", 0.05))
        std_lr = float(c["std_lr"]) if c.get("std_lr") is not None else lr
        weight_decay = float(c.get("weight_decay", 0.0))
        grad_clip = c.get("grad_clip", 1.0)
        normalize_adv = bool(c.get("normalize_advantages", False))
        buffer_size = int(c.get("buffer_size", 100000))
        train_offset = int(c.get("train_seed_offset", 10000))

        rng = np.random.default_rng(seed)
        gen = torch.Generator(device=device).manual_seed(seed)

        log_std = nn.Parameter(
            torch.full((latent_param.z_dim,), math.log(init_sigma), dtype=torch.float32, device=device),
            requires_grad=learn_std,
        )
        param_groups: List[Dict[str, Any]] = [
            {"params": list(latent_param.parameters()), "lr": lr, "weight_decay": weight_decay}
        ]
        if learn_std:
            param_groups.append({"params": [log_std], "lr": std_lr, "weight_decay": 0.0})
        optimizer = torch.optim.Adam(param_groups)

        state_buffer = PhysicsStateBuffer(buffer_size)
        scheduler = EvalScheduler.from_budget(
            max_steps=max_env_steps,
            eval_every=eval_every_env_steps,
            run_eval=eval_callback,
            start_step=0,
        )
        scheduler.trigger_due(0)

        d0_env = make_domain_env(domain, train_offset + seed, dynamics=dynamics)
        rollout_envs = [
            make_domain_env(domain, train_offset + seed + 1 + idx, dynamics=dynamics)
            for idx in range(m * k)
        ]

        train_rows: List[Dict[str, Any]] = []
        env_steps = 0
        grad_step = 0

        try:
            while env_steps < max_env_steps:
                grad_step += 1
                latent_param.eval()
                with torch.no_grad():
                    mu_z = latent_param.compute_z()
                    sigma_vec = torch.exp(log_std.clamp(math.log(min_sigma), math.log(max_sigma)))
                    eps = torch.randn(m, k, latent_param.z_dim, generator=gen, device=device, dtype=mu_z.dtype)
                    z_samples = mu_z.view(1, 1, -1) + sigma_vec.view(1, 1, -1) * eps
                    z_sphere = project_sphere(z_samples)

                # Sample m starting states from mixture nu(D_online, d_0) and reset each k-group identically.
                start_obs: List[np.ndarray] = []
                for b in range(m):
                    if len(state_buffer) == 0 or float(rng.random()) < init_state_prob:
                        ts0 = d0_env.reset()
                        phys0 = d0_env.physics.get_state().copy()
                        obs0 = np.asarray(ts0.observation["observations"], dtype=np.float32).copy()
                    else:
                        phys0, obs0 = state_buffer.sample(rng)
                    for i in range(k):
                        env_idx = b * k + i
                        set_env_physics(rollout_envs[env_idx], phys0)
                        start_obs.append(obs0.copy())

                obs_batch = np.stack(start_obs, axis=0)
                z_flat = z_sphere.reshape(m * k, -1)
                done = np.zeros(m * k, dtype=bool)
                returns = np.zeros(m * k, dtype=np.float64)
                gammas = np.ones(m * k, dtype=np.float64)

                for _ in range(horizon):
                    if done.all():
                        break
                    s_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        actions = policy(agent, s_t, z_flat).cpu().numpy()
                    for idx, env in enumerate(rollout_envs):
                        if done[idx]:
                            continue
                        ts = env.step(actions[idx])
                        env_steps += 1
                        next_obs = np.asarray(ts.observation["observations"], dtype=np.float32)
                        obs_batch[idx] = next_obs
                        returns[idx] += gammas[idx] * float(reward_fn(env.physics))
                        gammas[idx] *= discount
                        if ts.last():
                            done[idx] = True
                        else:
                            state_buffer.add(env.physics.get_state().copy(), next_obs.copy())

                if bootstrap and (~done).any():
                    active = np.flatnonzero(~done)
                    s_term = torch.as_tensor(obs_batch[active], dtype=torch.float32, device=device)
                    z_term = z_flat[active]
                    with torch.no_grad():
                        a_term = policy(agent, s_term, z_term)
                        v_term = fb_q_base(agent, s_term, a_term, z_term, w_r, head=fb_q_head).cpu().numpy()
                    returns[active] += gammas[active] * v_term

                ret_tensor = torch.as_tensor(returns.reshape(m, k), dtype=torch.float32, device=device)
                stats_step = lola_update_step(
                    latent_param,
                    log_std,
                    optimizer,
                    z_samples,
                    ret_tensor,
                    learn_std=learn_std,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    grad_clip=grad_clip,
                    normalize_advantages=normalize_adv,
                )
                with torch.no_grad():
                    new_mu = latent_param.compute_z()
                    cos = float(F.cosine_similarity(new_mu.unsqueeze(0), z_r.unsqueeze(0)).item())
                train_rows.append({
                    "task": task,
                    "step": grad_step,
                    "env_steps": env_steps,
                    "mean_return": stats_step["mean_return"],
                    "std_return": stats_step["std_return"],
                    "max_return": stats_step["max_return"],
                    "cos_to_zr": cos,
                    "loss": stats_step["loss"],
                    "critic_loss": float("nan"),
                    "sigma_mean": stats_step["sigma_mean"],
                })
                scheduler.trigger_due(env_steps)
        finally:
            d0_env.close()
            for env in rollout_envs:
                env.close()

        scheduler.finalize(env_steps)
        return train_rows


@torch.no_grad()
def _polyak_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    for p, tp in zip(online.parameters(), target.parameters()):
        tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


class RelaUpdater:
    """Residual Latent Adaptation (Sikchi et al., 2025, Algorithm 1, TD3-based)."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def run(
        self,
        *,
        task: str,
        domain: str,
        agent: FB,
        latent_param: LatentParam,
        z_r: torch.Tensor,
        w_r: torch.Tensor,
        reward_fn,
        dynamics,
        seed: int,
        eval_callback: Callable[[Sequence[int]], None],
        budget_cfg: Dict[str, Any],
        device: torch.device,
    ) -> List[Dict[str, Any]]:
        c = self.cfg
        max_env_steps = int(budget_cfg["max_env_steps"])
        eval_every_env_steps = int(budget_cfg.get("eval_every_env_steps") or 0)

        use_residual = bool(c.get("use_residual", True))
        hidden_dim = int(
            c["hidden_dim"]
            if c.get("hidden_dim") is not None
            else (c.get("residual_hidden_dim", 64) if use_residual else c.get("scratch_hidden_dim", 1024))
        )
        critic_layers = int(c.get("critic_layers", 2))
        fb_q_head = str(c.get("fb_q_head", "min"))
        discount = float(c.get("discount", 0.98))
        tau = float(c.get("tau", 0.005))
        utd = int(c.get("utd", 4))
        actor_update_freq = int(c.get("actor_update_freq", 4))
        batch_size = int(c.get("batch_size", 256))
        buffer_size = int(c.get("buffer_size", max(max_env_steps, 100000)))
        exploration_std = float(c.get("exploration_std", 0.1))
        target_policy_noise = float(c.get("target_policy_noise", 0.2))
        target_noise_clip = float(c.get("target_noise_clip", 0.3))
        warmup_steps = int(c.get("warmup_steps", batch_size))
        actor_warmup_steps = int(c.get("actor_warmup_steps", 0))
        critic_lr = float(c.get("critic_lr", 1e-4))
        actor_lr = float(c.get("actor_lr", 1e-4))
        weight_decay = float(c.get("weight_decay", 0.0))
        grad_clip = c.get("grad_clip", 1.0)
        log_every_env_steps = int(c.get("log_every_env_steps", 1000))
        train_offset = int(c.get("train_seed_offset", 10000))

        rng = np.random.default_rng(seed)
        gen = torch.Generator(device=device).manual_seed(seed)

        env = make_domain_env(domain, train_offset + seed, dynamics=dynamics)
        ts = env.reset()
        obs = np.asarray(ts.observation["observations"], dtype=np.float32)
        obs_dim = int(obs.shape[0])
        action_dim = int(env.action_spec().shape[0])

        critic = ResidualCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            layers=critic_layers,
            zero_init_last=use_residual,
        ).to(device)
        target_critic = copy.deepcopy(critic).to(device)
        target_critic.eval()

        critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        actor_opt = torch.optim.Adam(latent_param.parameters(), lr=actor_lr, weight_decay=weight_decay)

        buffer = TransitionBuffer(buffer_size, obs_dim, action_dim)
        scheduler = EvalScheduler.from_budget(
            max_steps=max_env_steps,
            eval_every=eval_every_env_steps,
            run_eval=eval_callback,
            start_step=0,
        )
        scheduler.trigger_due(0)

        # Freeze FB parameters during ReLA so gradients pass through actor & F into latent_param only.
        fb_grad_flags = [(p, p.requires_grad) for p in agent.parameters()]
        for p, _ in fb_grad_flags:
            p.requires_grad_(False)

        train_rows: List[Dict[str, Any]] = []
        ep_return = 0.0
        completed_returns: List[float] = []
        critic_losses: List[float] = []
        actor_losses: List[float] = []
        critic_updates = 0
        actor_updates = 0

        try:
            latent_param.eval()
            with torch.no_grad():
                z_curr = latent_param.compute_z().detach()

            for env_step in range(1, max_env_steps + 1):
                with torch.no_grad():
                    s_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    a_det = policy(agent, s_t, z_curr.unsqueeze(0))[0].cpu().numpy()
                if exploration_std > 0:
                    noise = rng.normal(0.0, exploration_std, size=action_dim).astype(np.float32)
                    action = np.clip(a_det + noise, -1.0, 1.0)
                else:
                    action = a_det

                ts = env.step(action)
                next_obs = np.asarray(ts.observation["observations"], dtype=np.float32)
                reward = float(reward_fn(env.physics))
                buffer.add(obs, action, reward, next_obs)
                ep_return += reward

                if ts.last():
                    completed_returns.append(ep_return)
                    ep_return = 0.0
                    ts = env.reset()
                    obs = np.asarray(ts.observation["observations"], dtype=np.float32)
                else:
                    obs = next_obs

                if env_step >= warmup_steps and len(buffer) >= batch_size:
                    for _ in range(utd):
                        s_b, a_b, r_b, s_next_b = buffer.sample(batch_size, rng, device)

                        # 1. TD3 critic update on Q_base(s, a) + Q_res(s, a)
                        with torch.no_grad():
                            z_exp = z_curr.unsqueeze(0).expand(batch_size, -1)
                            a_next = policy(agent, s_next_b, z_exp)
                            if target_policy_noise > 0:
                                eps_t = torch.randn(
                                    a_next.shape, generator=gen, device=device, dtype=a_next.dtype
                                ) * target_policy_noise
                                eps_t = eps_t.clamp(-target_noise_clip, target_noise_clip)
                                a_next = (a_next + eps_t).clamp(-1.0, 1.0)
                            tq1, tq2 = target_critic(s_next_b, a_next)
                            tq_res = torch.minimum(tq1, tq2)
                            if use_residual:
                                tq_base = fb_q_base(agent, s_next_b, a_next, z_r, w_r, head=fb_q_head)
                                target_q = r_b + discount * (tq_base + tq_res)
                                base_q = fb_q_base(agent, s_b, a_b, z_r, w_r, head=fb_q_head)
                            else:
                                target_q = r_b + discount * tq_res
                                base_q = None

                        q1_res, q2_res = critic(s_b, a_b)
                        if use_residual and base_q is not None:
                            q1_pred = base_q + q1_res
                            q2_pred = base_q + q2_res
                        else:
                            q1_pred = q1_res
                            q2_pred = q2_res

                        c_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)
                        critic_opt.zero_grad()
                        c_loss.backward()
                        if grad_clip is not None and grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
                        critic_opt.step()
                        _polyak_update(critic, target_critic, tau)
                        critic_losses.append(float(c_loss.item()))
                        critic_updates += 1

                        # 2. Latent actor update through frozen FB actor & Q_base + Q_res
                        if env_step >= actor_warmup_steps and critic_updates % actor_update_freq == 0:
                            for p in critic.parameters():
                                p.requires_grad_(False)
                            latent_param.train()
                            actor_opt.zero_grad()
                            z_phi = latent_param.compute_z()
                            a_phi = policy(agent, s_b, z_phi.unsqueeze(0).expand(batch_size, -1))
                            q1_a, _ = critic(s_b, a_phi)
                            if use_residual:
                                q_base_a = fb_q_base(agent, s_b, a_phi, z_r, w_r, head=fb_q_head)
                                q_actor = q_base_a + q1_a
                            else:
                                q_actor = q1_a
                            a_loss = -q_actor.mean()
                            a_loss.backward()
                            if grad_clip is not None and grad_clip > 0:
                                torch.nn.utils.clip_grad_norm_(latent_param.parameters(), grad_clip)
                            actor_opt.step()
                            latent_param.post_step()
                            latent_param.eval()
                            for p in critic.parameters():
                                p.requires_grad_(True)
                            with torch.no_grad():
                                z_curr = latent_param.compute_z().detach()
                            actor_losses.append(float(a_loss.item()))
                            actor_updates += 1

                if env_step % log_every_env_steps == 0 or env_step == max_env_steps:
                    with torch.no_grad():
                        cos = float(F.cosine_similarity(z_curr.unsqueeze(0), z_r.unsqueeze(0)).item())
                    if completed_returns:
                        ret_arr = np.asarray(completed_returns, dtype=np.float64)
                        mean_r = float(ret_arr.mean())
                        std_r = float(ret_arr.std())
                        max_r = float(ret_arr.max())
                    else:
                        mean_r = ep_return
                        std_r = 0.0
                        max_r = ep_return
                    train_rows.append({
                        "task": task,
                        "step": actor_updates,
                        "env_steps": env_step,
                        "mean_return": mean_r,
                        "std_return": std_r,
                        "max_return": max_r,
                        "cos_to_zr": cos,
                        "loss": float(np.mean(actor_losses)) if actor_losses else float("nan"),
                        "critic_loss": float(np.mean(critic_losses)) if critic_losses else float("nan"),
                        "sigma_mean": exploration_std,
                    })
                    completed_returns.clear()
                    critic_losses.clear()
                    actor_losses.clear()

                scheduler.trigger_due(env_step)
        finally:
            env.close()
            for p, flag in fb_grad_flags:
                p.requires_grad_(flag)

        scheduler.finalize(max_env_steps)
        return train_rows


def cem_sample_candidates(
    mean: torch.Tensor,
    std: torch.Tensor,
    pop_size: int,
    antithetic: bool,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample ``pop_size`` parameter vectors from ``N(mean, diag(std^2))``."""
    d = mean.shape[0]
    if antithetic:
        if pop_size % 2 != 0:
            raise ValueError(f"antithetic sampling requires even pop_size, got {pop_size}")
        half = torch.randn(pop_size // 2, d, generator=generator, device=mean.device, dtype=mean.dtype)
        eps = torch.cat([half, -half], dim=0)
    else:
        eps = torch.randn(pop_size, d, generator=generator, device=mean.device, dtype=mean.dtype)
    return mean.unsqueeze(0) + eps * std.unsqueeze(0)


def cem_update_step(
    mean: torch.Tensor,
    std: torch.Tensor,
    candidates: torch.Tensor,
    returns: torch.Tensor,
    *,
    num_elites: int,
    alpha: float = 0.7,
    min_sigma: float = 1e-2,
    max_sigma: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One diagonal Cross-Entropy Method update from candidate parameters and scalar returns."""
    pop_size = int(candidates.shape[0])
    k_elites = max(1, min(int(num_elites), pop_size))
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    returns_t = torch.as_tensor(returns, device=candidates.device, dtype=candidates.dtype).reshape(-1)
    top_idx = torch.topk(returns_t, k=k_elites, largest=True).indices
    elites = candidates[top_idx]
    elite_mean = elites.mean(dim=0)
    elite_std = elites.std(dim=0, unbiased=False) if k_elites > 1 else torch.zeros_like(std)
    new_mean = (1.0 - alpha) * mean + alpha * elite_mean
    new_std = ((1.0 - alpha) * std + alpha * elite_std).clamp(min=float(min_sigma), max=float(max_sigma))
    return new_mean, new_std


class CemUpdater:
    """Derivative-free Cross-Entropy Method updater operating in ``LatentParam`` parameter space."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def run(
        self,
        *,
        task: str,
        domain: str,
        agent: FB,
        latent_param: LatentParam,
        z_r: torch.Tensor,
        w_r: torch.Tensor,
        reward_fn,
        dynamics,
        seed: int,
        eval_callback: Callable[[Sequence[int]], None],
        budget_cfg: Dict[str, Any],
        device: torch.device,
    ) -> List[Dict[str, Any]]:
        c = self.cfg
        max_env_steps = int(budget_cfg["max_env_steps"])
        eval_every_env_steps = int(budget_cfg.get("eval_every_env_steps") or 0)

        pop_size = int(c.get("pop_size", 16))
        elite_frac = float(c.get("elite_frac", 0.25))
        raw_num_elites = c.get("num_elites")
        num_elites = int(raw_num_elites) if raw_num_elites is not None else max(2, int(round(pop_size * elite_frac)))
        init_sigma = float(c.get("init_sigma", 0.3))
        min_sigma = float(c.get("min_sigma", 1e-2))
        max_sigma = float(c.get("max_sigma", 2.0))
        alpha = float(c.get("alpha", 0.7))
        antithetic = bool(c.get("antithetic", True))
        param_scope = str(c.get("param_scope", "all"))

        gen = torch.Generator(device=device).manual_seed(seed)
        mean = latent_param.get_flat_params(param_scope).to(device=device)
        std = torch.full_like(mean, init_sigma)

        evaluator = CandidateEvaluator(
            agent=agent,
            domain=domain,
            reward_fn=reward_fn,
            w_r=w_r,
            dynamics=dynamics,
            episodes_per_z=int(c.get("train_episodes_per_z", 1)),
            episode_steps=c.get("episode_steps"),
            horizon=c.get("horizon"),
            num_start_states=int(c.get("num_start_states", 2)),
            init_state_prob=float(c.get("init_state_prob", 0.2)),
            bootstrap=bool(c.get("bootstrap", True)),
            fb_q_head=str(c.get("fb_q_head", "min")),
            discount=float(c.get("discount", 0.98)),
            buffer_size=int(c.get("buffer_size", 100000)),
            seed=seed,
            train_seed_offset=int(c.get("train_seed_offset", 10000)),
        )

        scheduler = EvalScheduler.from_budget(max_env_steps, eval_every_env_steps, eval_callback, start_step=0)
        scheduler.trigger_due(0)

        train_rows: List[Dict[str, Any]] = []
        env_steps = 0
        gen_idx = 0
        try:
            while env_steps < max_env_steps:
                gen_idx += 1
                candidates = cem_sample_candidates(mean, std, pop_size, antithetic, gen)
                zs = latent_param.eval_candidates_to_zs(candidates, scope=param_scope)
                fitness, batch_steps = evaluator.evaluate(zs)
                env_steps += batch_steps

                mean, std = cem_update_step(
                    mean,
                    std,
                    candidates,
                    torch.as_tensor(fitness, device=device, dtype=mean.dtype),
                    num_elites=num_elites,
                    alpha=alpha,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                )
                latent_param.set_flat_params(mean, scope=param_scope)
                mean = latent_param.get_flat_params(param_scope).to(device=device)

                with torch.no_grad():
                    mu_z = latent_param.compute_z()
                    cos = float(F.cosine_similarity(mu_z.unsqueeze(0), z_r.unsqueeze(0)).item())

                train_rows.append({
                    "task": task,
                    "step": gen_idx,
                    "env_steps": env_steps,
                    "mean_return": float(fitness.mean()),
                    "std_return": float(fitness.std()),
                    "max_return": float(fitness.max()),
                    "cos_to_zr": cos,
                    "loss": float("nan"),
                    "critic_loss": float("nan"),
                    "sigma_mean": float(std.mean().item()),
                })
                scheduler.trigger_due(env_steps)
        finally:
            evaluator.close()

        scheduler.finalize(env_steps)
        return train_rows


def build_updater(algo: str, cfg: Dict[str, Any]):
    """Instantiate the updater for ``algo`` ('reinforce', 'lola', or 'rela')."""
    key = normalize_algo_type(algo)
    if key == "reinforce":
        return ReinforceUpdater(cfg["reinforce"])
    if key == "lola":
        return LolaUpdater(cfg["lola"])
    if key == "rela":
        return RelaUpdater(cfg["rela"])
    raise ValueError(f"Unsupported algo {algo!r}")
