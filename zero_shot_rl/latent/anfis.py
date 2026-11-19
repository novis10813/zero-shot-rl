"""Adaptive Neuro-Fuzzy Inference System (ANFIS) in PyTorch.

Implements a five-layer Takagi-Sugeno ANFIS with R learnable multivariate rules rather than a
full M^D Cartesian grid, so it scales to continuous-control state dimensions without rule
explosion or floating-point underflow:

    Layer 1 (Membership)    log mu_{k,j}(x_j) for rule k in 1..R, dim j in 1..D
    Layer 2 (Firing)        log w_k(x) = t_norm_j(log mu_{k,j}(x_j)) / temperature
    Layer 3 (Normalisation) w_bar_k(x) = softmax_k(log w_k(x))
    Layer 4 (Consequent)    f_k(x) in R^{out_dim}  (zero_order, first_order, mlp, or custom)
    Layer 5 (Aggregation)   y(x) = output_activation(sum_k w_bar_k(x) f_k(x))

Membership and consequent layers are built via `MEMBERSHIP_REGISTRY` and `CONSEQUENT_REGISTRY`
(or passed as `nn.Module` instances) so YAML configs can freely compose architectures and future
KAN / hybrid blocks drop in with one registry entry.

`weighted_z(weights, backward_repr)` mirrors `agents.fb.agent.FB.infer_z` while preserving
gradients from the normalised latent z back into the weights produced by ANFIS.
"""

import math
from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F

MEMBERSHIPS = ("gaussian", "bell")
CONSEQUENTS = ("zero_order", "first_order", "mlp")
T_NORMS = ("product", "min")
OUTPUT_ACTIVATIONS = ("identity", "softplus", "sigmoid", "tanh")


def _inv_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _activation_fn(name: str):
    if name == "identity":
        return lambda x: x
    if name == "softplus":
        return F.softplus
    if name == "sigmoid":
        return torch.sigmoid
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return F.relu
    if name == "silu":
        return F.silu
    raise ValueError(f"unknown activation {name!r}")


class GaussianMembership(torch.nn.Module):
    """Rule-wise diagonal Gaussian membership: log mu_{k,j}(x_j) = -0.5 * ((x_j - c_{k,j}) / sigma_{k,j})^2."""

    def __init__(self, in_dim: int, num_rules: int, min_sigma: float = 1e-3, init_sigma: float = 1.0):
        super().__init__()
        if init_sigma <= min_sigma:
            raise ValueError(f"init_sigma ({init_sigma}) must exceed min_sigma ({min_sigma})")
        self.in_dim = in_dim
        self.num_rules = num_rules
        self.min_sigma = float(min_sigma)
        self.centers = torch.nn.Parameter(torch.randn(num_rules, in_dim) * 0.5)
        raw_init = _inv_softplus(init_sigma - self.min_sigma)
        self.raw_sigma = torch.nn.Parameter(torch.full((num_rules, in_dim), raw_init))

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma) + self.min_sigma

    def log_membership(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_dim] -> [..., num_rules, in_dim]."""
        diff = (x.unsqueeze(-2) - self.centers) / self.sigma
        return -0.5 * (diff ** 2)

    def membership(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_membership(x))

    @torch.no_grad()
    def set_params(self, centers: torch.Tensor, sigma: torch.Tensor) -> None:
        self.centers.copy_(centers)
        target = (sigma - self.min_sigma).clamp_min(1e-6)
        self.raw_sigma.copy_(torch.log(torch.expm1(target)))


class BellMembership(torch.nn.Module):
    """Rule-wise generalised bell membership: mu_{k,j}(x_j) = 1 / (1 + |(x_j - c_{k,j}) / a_{k,j}|^{2 b_{k,j}})."""

    def __init__(
        self,
        in_dim: int,
        num_rules: int,
        min_scale: float = 1e-3,
        init_scale: float = 1.0,
        init_slope: float = 1.0,
        eps: float = 1e-12,
    ):
        super().__init__()
        if init_scale <= min_scale:
            raise ValueError(f"init_scale ({init_scale}) must exceed min_scale ({min_scale})")
        if init_slope <= min_scale:
            raise ValueError(f"init_slope ({init_slope}) must exceed min_scale ({min_scale})")
        self.in_dim = in_dim
        self.num_rules = num_rules
        self.min_scale = float(min_scale)
        self.eps = float(eps)
        self.centers = torch.nn.Parameter(torch.randn(num_rules, in_dim) * 0.5)
        self.raw_scale = torch.nn.Parameter(
            torch.full((num_rules, in_dim), _inv_softplus(init_scale - self.min_scale))
        )
        self.raw_slope = torch.nn.Parameter(
            torch.full((num_rules, in_dim), _inv_softplus(init_slope - self.min_scale))
        )

    @property
    def scale(self) -> torch.Tensor:
        return F.softplus(self.raw_scale) + self.min_scale

    @property
    def slope(self) -> torch.Tensor:
        return F.softplus(self.raw_slope) + self.min_scale

    def log_membership(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_dim] -> [..., num_rules, in_dim]."""
        sq = ((x.unsqueeze(-2) - self.centers) / self.scale) ** 2 + self.eps
        return -torch.log1p(sq ** self.slope)

    def membership(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_membership(x))

    @torch.no_grad()
    def set_params(self, centers: torch.Tensor, scale: torch.Tensor) -> None:
        self.centers.copy_(centers)
        target = (scale - self.min_scale).clamp_min(1e-6)
        self.raw_scale.copy_(torch.log(torch.expm1(target)))


class ZeroOrderConsequent(torch.nn.Module):
    """Zero-order Takagi-Sugeno consequent: constant vector c_k in R^{out_dim} per rule."""

    def __init__(self, in_dim: int, out_dim: int, num_rules: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_rules = num_rules
        self.bias = torch.nn.Parameter(torch.empty(num_rules, out_dim).uniform_(-0.1, 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_dim] -> [..., num_rules, out_dim]."""
        return self.bias.expand(*x.shape[:-1], self.num_rules, self.out_dim)


class FirstOrderConsequent(torch.nn.Module):
    """First-order Takagi-Sugeno consequent: f_k(x) = W_k x + b_k per rule."""

    def __init__(self, in_dim: int, out_dim: int, num_rules: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_rules = num_rules
        bound = 1.0 / math.sqrt(in_dim)
        self.weight = torch.nn.Parameter(torch.empty(num_rules, out_dim, in_dim).uniform_(-bound, bound))
        self.bias = torch.nn.Parameter(torch.zeros(num_rules, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_dim] -> [..., num_rules, out_dim]."""
        return torch.einsum("...d,rod->...ro", x, self.weight) + self.bias


class MLPConsequent(torch.nn.Module):
    """Nonlinear MLP consequent predicting all R rule outputs [..., num_rules, out_dim]."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_rules: int,
        hidden: int = 32,
        layers: int = 1,
        activation: str = "silu",
    ):
        super().__init__()
        if hidden <= 0 or layers <= 0:
            raise ValueError("hidden and layers must be positive")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_rules = num_rules
        act_map = {"relu": torch.nn.ReLU, "silu": torch.nn.SiLU, "tanh": torch.nn.Tanh}
        if activation not in act_map:
            raise ValueError(f"activation must be one of {tuple(act_map)}, got {activation!r}")
        modules = []
        width = in_dim
        for _ in range(layers):
            modules += [torch.nn.Linear(width, hidden), act_map[activation]()]
            width = hidden
        modules.append(torch.nn.Linear(width, num_rules * out_dim))
        self.net = torch.nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_dim] -> [..., num_rules, out_dim]."""
        return self.net(x).view(*x.shape[:-1], self.num_rules, self.out_dim)


MEMBERSHIP_REGISTRY = {
    "gaussian": GaussianMembership,
    "bell": BellMembership,
}

CONSEQUENT_REGISTRY = {
    "zero_order": ZeroOrderConsequent,
    "first_order": FirstOrderConsequent,
    "mlp": MLPConsequent,
}


def _normalise_spec(spec: Union[str, Dict[str, Any], torch.nn.Module]):
    if isinstance(spec, torch.nn.Module):
        return spec, None
    if isinstance(spec, str):
        return spec, {"type": spec}
    if isinstance(spec, dict):
        if "type" not in spec:
            raise ValueError(f"component spec dict must contain 'type', got {spec!r}")
        return spec["type"], dict(spec)
    raise ValueError(f"expected str, dict or nn.Module, got {type(spec).__name__}")


def _build_membership(spec, in_dim: int, num_rules: int, min_sigma: float):
    kind, cfg = _normalise_spec(spec)
    if isinstance(kind, torch.nn.Module):
        return kind, None
    if kind not in MEMBERSHIP_REGISTRY:
        raise ValueError(f"membership must be one of {tuple(MEMBERSHIP_REGISTRY)}, got {kind!r}")
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    if kind == "gaussian":
        kwargs.setdefault("min_sigma", min_sigma)
    elif kind == "bell":
        if "min_sigma" in kwargs and "min_scale" not in kwargs:
            kwargs["min_scale"] = kwargs.pop("min_sigma")
        kwargs.setdefault("min_scale", min_sigma)
    return MEMBERSHIP_REGISTRY[kind](in_dim=in_dim, num_rules=num_rules, **kwargs), {"type": kind, **kwargs}


def _build_consequent(spec, in_dim: int, out_dim: int, num_rules: int):
    kind, cfg = _normalise_spec(spec)
    if isinstance(kind, torch.nn.Module):
        return kind, None
    if kind not in CONSEQUENT_REGISTRY:
        raise ValueError(f"consequent must be one of {tuple(CONSEQUENT_REGISTRY)}, got {kind!r}")
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    return (
        CONSEQUENT_REGISTRY[kind](in_dim=in_dim, out_dim=out_dim, num_rules=num_rules, **kwargs),
        {"type": kind, **kwargs},
    )


class ANFIS(torch.nn.Module):
    """Generic composable multi-input multi-output Takagi-Sugeno ANFIS.

    Args:
        in_dim: Input feature dimension D.
        out_dim: Output dimension O (e.g. 1 for scalar reward weights, d_z for direct latents).
        num_rules: Number of learnable fuzzy rules R.
        membership: "gaussian", "bell", a config dict `{"type": ..., ...}`, or an `nn.Module`
            exposing `log_membership(x)`.
        consequent: "first_order", "zero_order", "mlp", a config dict `{"type": ..., ...}`, or
            an `nn.Module` mapping `[..., D] -> [..., R, O]`.
        t_norm: "product" (sum of log-memberships) or "min" (minimum log-membership across dims).
        normalize_firing: If True (default), normalise rule strengths via softmax (standard ANFIS).
            If False, use unnormalised strengths psi_k(x) = exp(log w_k(x)) in (0, 1].
        output_activation: "identity", "softplus", "sigmoid", or "tanh" applied after aggregation.
        temperature: Divisor on aggregated log-memberships. Defaults to sqrt(in_dim) for "product"
            and 1.0 for "min".
        min_sigma: Default lower bound on membership scale parameters when not specified in `membership`.
        normalize_inputs: If True, standardise inputs with stored `x_mean` and `x_std` buffers.
        eps: Floor on `x_std` during input standardisation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_rules: int,
        membership: Union[str, Dict[str, Any], torch.nn.Module] = "gaussian",
        consequent: Union[str, Dict[str, Any], torch.nn.Module] = "first_order",
        t_norm: str = "product",
        normalize_firing: bool = True,
        output_activation: str = "identity",
        temperature: Optional[float] = None,
        min_sigma: float = 1e-3,
        normalize_inputs: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        if in_dim <= 0 or out_dim <= 0 or num_rules <= 0:
            raise ValueError("in_dim, out_dim and num_rules must be positive")
        if t_norm not in T_NORMS:
            raise ValueError(f"t_norm must be one of {T_NORMS}, got {t_norm!r}")
        if output_activation not in OUTPUT_ACTIVATIONS:
            raise ValueError(f"output_activation must be one of {OUTPUT_ACTIVATIONS}, got {output_activation!r}")
        default_temp = math.sqrt(in_dim) if t_norm == "product" else 1.0
        resolved_temp = default_temp if temperature is None else float(temperature)
        if resolved_temp <= 0:
            raise ValueError(f"temperature must be positive, got {resolved_temp}")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_rules = int(num_rules)
        self.t_norm = t_norm
        self.normalize_firing = bool(normalize_firing)
        self.output_activation = output_activation
        self._out_act = _activation_fn(output_activation)
        self.temperature = resolved_temp
        self.min_sigma = float(min_sigma)
        self.normalize_inputs = bool(normalize_inputs)
        self.eps = float(eps)

        self.membership, mem_cfg = _build_membership(membership, self.in_dim, self.num_rules, self.min_sigma)
        self.consequent, cons_cfg = _build_consequent(consequent, self.in_dim, self.out_dim, self.num_rules)

        self.register_buffer("x_mean", torch.zeros(self.in_dim))
        self.register_buffer("x_std", torch.ones(self.in_dim))

        self.config = (
            dict(
                in_dim=self.in_dim,
                out_dim=self.out_dim,
                num_rules=self.num_rules,
                membership=mem_cfg,
                consequent=cons_cfg,
                t_norm=self.t_norm,
                normalize_firing=self.normalize_firing,
                output_activation=self.output_activation,
                temperature=self.temperature,
                min_sigma=self.min_sigma,
                normalize_inputs=self.normalize_inputs,
                eps=self.eps,
            )
            if mem_cfg is not None and cons_cfg is not None
            else None
        )

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], in_dim: Optional[int] = None, out_dim: Optional[int] = None) -> "ANFIS":
        """Build an ANFIS from a YAML/dict configuration (e.g. the `anfis:` section)."""
        resolved_in = cfg.get("in_dim") if in_dim is None else in_dim
        resolved_out = cfg.get("out_dim", 1) if out_dim is None else out_dim
        if resolved_in is None:
            raise ValueError("in_dim must be provided in cfg or as argument")
        return cls(
            in_dim=int(resolved_in),
            out_dim=int(resolved_out),
            num_rules=int(cfg.get("num_rules", 8)),
            membership=cfg.get("membership", "gaussian"),
            consequent=cfg.get("consequent", "first_order"),
            t_norm=cfg.get("t_norm", "product"),
            normalize_firing=bool(cfg.get("normalize_firing", True)),
            output_activation=cfg.get("output_activation", "identity"),
            temperature=cfg.get("temperature", None),
            min_sigma=float(cfg.get("min_sigma", 1e-3)),
            normalize_inputs=bool(cfg.get("normalize_inputs", False)),
            eps=float(cfg.get("eps", 1e-6)),
        )

    @torch.no_grad()
    def set_input_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Store feature mean and std used when `normalize_inputs=True`."""
        self.x_mean.copy_(torch.as_tensor(mean, dtype=self.x_mean.dtype, device=self.x_mean.device))
        s = torch.as_tensor(std, dtype=self.x_std.dtype, device=self.x_std.device).clamp_min(self.eps)
        self.x_std.copy_(s)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_inputs:
            return x
        return (x - self.x_mean) / self.x_std

    @torch.no_grad()
    def init_from_data(self, x: torch.Tensor, seed: int = 0) -> None:
        """Initialise rule centers from random data points and scales from data std.

        If `normalize_inputs=True`, also sets `x_mean` and `x_std` from `x` first.
        """
        x = x.reshape(-1, self.in_dim).to(device=self.x_mean.device, dtype=self.x_mean.dtype)
        if len(x) == 0:
            raise ValueError("init_from_data requires at least one sample")
        if self.normalize_inputs:
            self.set_input_stats(x.mean(0), x.std(0, unbiased=False))
        xn = self.normalize(x)
        gen = torch.Generator(device=xn.device).manual_seed(seed)
        if len(xn) >= self.num_rules:
            idx = torch.randperm(len(xn), generator=gen, device=xn.device)[: self.num_rules]
        else:
            idx = torch.randint(len(xn), (self.num_rules,), generator=gen, device=xn.device)
        centers = xn[idx]
        std = xn.std(0, unbiased=False).clamp_min(self.min_sigma + 1e-2).expand(self.num_rules, -1)
        if hasattr(self.membership, "set_params"):
            self.membership.set_params(centers, std)

    def log_firing_strengths(self, x: torch.Tensor) -> torch.Tensor:
        """Unnormalised log firing strengths [..., num_rules] on already-normalised input `x`."""
        log_mu = self.membership.log_membership(x)
        if self.t_norm == "product":
            agg = log_mu.sum(dim=-1)
        else:
            agg = log_mu.min(dim=-1).values
        return agg / self.temperature

    def _weights_from_normalised(self, xn: torch.Tensor) -> torch.Tensor:
        log_w = self.log_firing_strengths(xn)
        if self.normalize_firing:
            return F.softmax(log_w, dim=-1)
        return torch.exp(log_w)

    def firing_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Rule firing weights [..., num_rules]: softmax-normalised if normalize_firing else exp(log_w) in (0, 1]."""
        return self._weights_from_normalised(self.normalize(x))

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        """Evaluate ANFIS on `x` [..., in_dim] -> [..., out_dim] (or (y, weights) if return_weights)."""
        xn = self.normalize(x)
        weights = self._weights_from_normalised(xn)
        rule_outputs = self.consequent(xn)
        out = self._out_act(torch.einsum("...r,...ro->...o", weights, rule_outputs))
        if return_weights:
            return out, weights
        return out

    def save(self, path, **meta) -> None:
        if self.config is None:
            raise ValueError("Cannot save ANFIS with custom membership/consequent modules via config dict")
        torch.save({"config": self.config, "state_dict": self.state_dict(), "meta": meta}, path)


def load_anfis(path, device: torch.device = torch.device("cpu")):
    """Load an ANFIS saved by `ANFIS.save`."""
    blob = torch.load(path, map_location=device, weights_only=False)
    model = ANFIS(**blob["config"]).to(device)
    model.load_state_dict(blob["state_dict"])
    return model.eval(), blob["meta"]


def weighted_z(weights: torch.Tensor, backward_repr: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable FB task-latent aggregation matching `agents.fb.agent.FB.infer_z`.

    Args:
        weights: State weights [N] or [N, 1] (e.g. `anfis(states)`).
        backward_repr: Frozen backward representations B(s) of shape [N, d_z].
        eps: Numerical floor in L2 normalisation.

    Returns:
        Normalised latent z of shape [d_z] with ||z||_2 = sqrt(d_z) (when numerator is nonzero).
    """
    w = weights.reshape(-1, 1)
    if w.shape[0] != backward_repr.shape[0]:
        raise ValueError(f"weights length {w.shape[0]} != backward_repr length {backward_repr.shape[0]}")
    z = (w.T @ backward_repr) / w.shape[0]
    return (math.sqrt(backward_repr.shape[1]) * F.normalize(z, dim=1, eps=eps)).squeeze(0)
