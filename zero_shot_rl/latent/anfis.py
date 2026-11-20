"""Classical Cartesian-grid Adaptive Neuro-Fuzzy Inference System (ANFIS) in PyTorch.

Implements a five-layer Takagi-Sugeno ANFIS where each input sensor `i` has its own
membership function family and `M_i` shared linguistic terms, and rules are formed by
the Cartesian product over all sensors (`R = prod_i M_i`):

    Layer 1 (Fuzzification) log mu_{i,m}(x_i) for sensor i in 1..D, term m in 1..M_i
    Layer 2 (Rule Firing)   log w_k(x) = t_norm_i(log mu_{i, m_i(k)}(x_i)) / temperature
    Layer 3 (Normalisation) w_bar_k(x) = softmax_k(log w_k(x))  (or exp(log w_k(x)))
    Layer 4 (Consequent)    f_k(x) in R^{out_dim}  (zero_order, first_order, mlp, or custom)
    Layer 5 (Aggregation)   y(x) = output_activation(sum_k w_bar_k(x) f_k(x))

Membership and consequent layers are built via `MEMBERSHIP_REGISTRY` and `CONSEQUENT_REGISTRY`
(or passed as `nn.Module` instances) so YAML configs can freely compose architectures.

`weighted_z(weights, backward_repr)` mirrors `agents.fb.agent.FB.infer_z` while preserving
gradients from the normalised latent z back into the weights produced by ANFIS.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

MEMBERSHIPS = ("gaussian", "bell", "triangular", "trapezoidal", "sigmoid", "psig")
CONSEQUENTS = ("zero_order", "first_order", "mlp")
T_NORMS = ("product", "min")
OUTPUT_ACTIVATIONS = ("identity", "softplus", "sigmoid", "tanh")


def _inv_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _default_centers(num_terms: int) -> torch.Tensor:
    if num_terms == 1:
        return torch.zeros(1)
    return torch.linspace(-1.5, 1.5, num_terms)


def _quantile_centers_and_spacing(x_1d: torch.Tensor, num_terms: int) -> Tuple[torch.Tensor, float]:
    """Compute evenly spaced quantile centers and representative spacing for a 1D sample tensor."""
    x_flat = x_1d.reshape(-1)
    std = float(x_flat.std(unbiased=False).item())
    if num_terms == 1:
        center = torch.quantile(x_flat, 0.5).unsqueeze(0)
        return center, max(std, 1.0)
    qs = torch.linspace(0.05, 0.95, num_terms, device=x_flat.device, dtype=x_flat.dtype)
    centers = torch.quantile(x_flat, qs)
    span = float((centers[-1] - centers[0]).item())
    if span < 1e-4:
        # Degenerate / near-constant sensor: spread synthetically around mean.
        mean = float(x_flat.mean().item())
        half = max(std, 0.5)
        centers = torch.linspace(mean - half, mean + half, num_terms, device=x_flat.device, dtype=x_flat.dtype)
        span = float((centers[-1] - centers[0]).item())
    spacing = span / max(num_terms - 1, 1)
    return centers, max(spacing, 1e-2)


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
    """1D Gaussian membership with `num_terms` terms: log mu_m(x) = -0.5 * ((x - c_m) / sigma_m)^2."""

    def __init__(self, num_terms: int, min_sigma: float = 1e-3, init_sigma: float = 1.0):
        super().__init__()
        if num_terms <= 0:
            raise ValueError(f"num_terms must be positive, got {num_terms}")
        if init_sigma <= min_sigma:
            raise ValueError(f"init_sigma ({init_sigma}) must exceed min_sigma ({min_sigma})")
        self.num_terms = int(num_terms)
        self.min_sigma = float(min_sigma)
        self.centers = torch.nn.Parameter(_default_centers(self.num_terms))
        raw_init = _inv_softplus(init_sigma - self.min_sigma)
        self.raw_sigma = torch.nn.Parameter(torch.full((self.num_terms,), raw_init))

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma) + self.min_sigma

    def log_membership(self, x: torch.Tensor) -> torch.Tensor:
        """x: [...] -> [..., num_terms]."""
        diff = (x.unsqueeze(-1) - self.centers) / self.sigma
        return -0.5 * (diff ** 2)

    def membership(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_membership(x))

    @torch.no_grad()
    def set_params(self, centers: torch.Tensor, sigma: torch.Tensor) -> None:
        self.centers.copy_(centers)
        target = (sigma - self.min_sigma).clamp_min(1e-6)
        self.raw_sigma.copy_(torch.log(torch.expm1(target)))

    @torch.no_grad()
    def init_from_1d(self, x_1d: torch.Tensor) -> None:
        centers, spacing = _quantile_centers_and_spacing(x_1d, self.num_terms)
        sigma = torch.full_like(centers, max(0.6 * spacing, self.min_sigma + 1e-2))
        self.set_params(centers, sigma)


class BellMembership(torch.nn.Module):
    """1D generalised bell membership: mu_m(x) = 1 / (1 + |(x - c_m) / a_m|^{2 b_m})."""

    def __init__(
        self,
        num_terms: int,
        min_scale: float = 1e-3,
        init_scale: float = 1.0,
        init_slope: float = 1.0,
        eps: float = 1e-12,
    ):
        super().__init__()
        if num_terms <= 0:
            raise ValueError(f"num_terms must be positive, got {num_terms}")
        if init_scale <= min_scale:
            raise ValueError(f"init_scale ({init_scale}) must exceed min_scale ({min_scale})")
        if init_slope <= min_scale:
            raise ValueError(f"init_slope ({init_slope}) must exceed min_scale ({min_scale})")
        self.num_terms = int(num_terms)
        self.min_scale = float(min_scale)
        self.eps = float(eps)
        self.centers = torch.nn.Parameter(_default_centers(self.num_terms))
        self.raw_scale = torch.nn.Parameter(
            torch.full((self.num_terms,), _inv_softplus(init_scale - self.min_scale))
        )
        self.raw_slope = torch.nn.Parameter(
            torch.full((self.num_terms,), _inv_softplus(init_slope - self.min_scale))
        )

    @property
    def scale(self) -> torch.Tensor:
        return F.softplus(self.raw_scale) + self.min_scale

    @property
    def slope(self) -> torch.Tensor:
        return F.softplus(self.raw_slope) + self.min_scale

    def log_membership(self, x: torch.Tensor) -> torch.Tensor:
        """x: [...] -> [..., num_terms]."""
        sq = ((x.unsqueeze(-1) - self.centers) / self.scale) ** 2 + self.eps
        return -torch.log1p(sq ** self.slope)

    def membership(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_membership(x))

    @torch.no_grad()
    def set_params(self, centers: torch.Tensor, scale: torch.Tensor) -> None:
        self.centers.copy_(centers)
        target = (scale - self.min_scale).clamp_min(1e-6)
        self.raw_scale.copy_(torch.log(torch.expm1(target)))

    @torch.no_grad()
    def init_from_1d(self, x_1d: torch.Tensor) -> None:
        centers, spacing = _quantile_centers_and_spacing(x_1d, self.num_terms)
        scale = torch.full_like(centers, max(0.5 * spacing, self.min_scale + 1e-2))
        self.set_params(centers, scale)


class TriangularMembership(torch.nn.Module):
    """1D triangular membership parameterized by peak `b` and positive left/right half-widths.

    Guaranteed ordered support `a = b - w_left < b < c = b + w_right`. Output is floored at `eps`
    before taking log to keep log-memberships finite.
    """

    def __init__(
        self,
        num_terms: int,
        min_width: float = 1e-3,
        init_width: float = 1.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        if num_terms <= 0:
            raise ValueError(f"num_terms must be positive, got {num_terms}")
        if init_width <= min_width:
            raise ValueError(f"init_width ({init_width}) must exceed min_width ({min_width})")
        self.num_terms = int(num_terms)
        self.min_width = float(min_width)
        self.eps = float(eps)
        self.centers = torch.nn.Parameter(_default_centers(self.num_terms))
        raw_w = _inv_softplus(init_width - self.min_width)
        self.raw_left = torch.nn.Parameter(torch.full((self.num_terms,), raw_w))
        self.raw_right = torch.nn.Parameter(torch.full((self.num_terms,), raw_w))

    @property
    def left_width(self) -> torch.Tensor:
        return F.softplus(self.raw_left) + self.min_width

    @property
    def right_width(self) -> torch.Tensor:
        return F.softplus(self.raw_right) + self.min_width

    def membership(self, x: torch.Tensor) -> torch.Tensor:
        """x: [...] -> [..., num_terms] in [eps, 1]."""
        xu = x.unsqueeze(-1)
        lw = self.left_width
        rw = self.right_width
        left_slope = (xu - (self.centers - lw)) / lw
        right_slope = ((self.centers + rw) - xu) / rw
        return torch.minimum(left_slope, right_slope).clamp(min=self.eps, max=1.0)

    def log_membership(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(self.membership(x))

    @torch.no_grad()
    def init_from_1d(self, x_1d: torch.Tensor) -> None:
        centers, spacing = _quantile_centers_and_spacing(x_1d, self.num_terms)
        self.centers.copy_(centers)
        w = max(1.2 * spacing, self.min_width + 1e-2)
        raw_w = _inv_softplus(w - self.min_width)
        self.raw_left.fill_(raw_w)
        self.raw_right.fill_(raw_w)


class TrapezoidalMembership(torch.nn.Module):
    """1D trapezoidal membership parameterized by center, positive half-top, and left/right slopes.

    Guaranteed ordered vertices `a < b < c < d` with `b = center - half_top`, `c = center + half_top`,
    `a = b - left_width`, `d = c + right_width`.
    """

    def __init__(
        self,
        num_terms: int,
        min_width: float = 1e-3,
        init_slope_width: float = 1.0,
        init_top_width: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        if num_terms <= 0:
            raise ValueError(f"num_terms must be positive, got {num_terms}")
        if init_slope_width <= min_width:
            raise ValueError(f"init_slope_width ({init_slope_width}) must exceed min_width ({min_width})")
        if init_top_width <= min_width:
            raise ValueError(f"init_top_width ({init_top_width}) must exceed min_width ({min_width})")
        self.num_terms = int(num_terms)
        self.min_width = float(min_width)
        self.eps = float(eps)
        self.centers = torch.nn.Parameter(_default_centers(self.num_terms))
        raw_top = _inv_softplus(0.5 * init_top_width - self.min_width) if 0.5 * init_top_width > min_width else _inv_softplus(min_width)
        raw_slope = _inv_softplus(init_slope_width - self.min_width)
        self.raw_half_top = torch.nn.Parameter(torch.full((self.num_terms,), raw_top))
        self.raw_left = torch.nn.Parameter(torch.full((self.num_terms,), raw_slope))
        self.raw_right = torch.nn.Parameter(torch.full((self.num_terms,), raw_slope))

    @property
    def half_top(self) -> torch.Tensor:
        return F.softplus(self.raw_half_top) + self.min_width

    @property
    def left_width(self) -> torch.Tensor:
        return F.softplus(self.raw_left) + self.min_width

    @property
    def right_width(self) -> torch.Tensor:
        return F.softplus(self.raw_right) + self.min_width

    def membership(self, x: torch.Tensor) -> torch.Tensor:
        """x: [...] -> [..., num_terms] in [eps, 1]."""
        xu = x.unsqueeze(-1)
        b = self.centers - self.half_top
        c = self.centers + self.half_top
        lw = self.left_width
        rw = self.right_width
        left_slope = (xu - (b - lw)) / lw
        right_slope = ((c + rw) - xu) / rw
        return torch.minimum(left_slope, right_slope).clamp(min=self.eps, max=1.0)

    def log_membership(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(self.membership(x))

    @torch.no_grad()
    def init_from_1d(self, x_1d: torch.Tensor) -> None:
        centers, spacing = _quantile_centers_and_spacing(x_1d, self.num_terms)
        self.centers.copy_(centers)
        ht = max(0.25 * spacing, self.min_width + 1e-2)
        sw = max(0.9 * spacing, self.min_width + 1e-2)
        self.raw_half_top.fill_(_inv_softplus(ht - self.min_width))
        self.raw_left.fill_(_inv_softplus(sw - self.min_width))
        self.raw_right.fill_(_inv_softplus(sw - self.min_width))


class SigmoidMembership(torch.nn.Module):
    """1D sigmoidal membership: mu_m(x) = sigmoid(a_m * (x - c_m)) with learnable signed slope `a_m`."""

    def __init__(self, num_terms: int, init_slope: float = 2.0):
        super().__init__()
        if num_terms <= 0:
            raise ValueError(f"num_terms must be positive, got {num_terms}")
        self.num_terms = int(num_terms)
        self.init_slope = float(init_slope)
        self.centers = torch.nn.Parameter(_default_centers(self.num_terms))
        slopes = torch.full((self.num_terms,), abs(self.init_slope))
        if self.num_terms >= 2:
            # First term acts as left shoulder ("low"), remaining as right shoulders ("high").
            slopes[0] = -abs(self.init_slope)
        self.slopes = torch.nn.Parameter(slopes)

    def log_membership(self, x: torch.Tensor) -> torch.Tensor:
        """x: [...] -> [..., num_terms], exact log sigmoid via -softplus."""
        return -F.softplus(-self.slopes * (x.unsqueeze(-1) - self.centers))

    def membership(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_membership(x))

    @torch.no_grad()
    def init_from_1d(self, x_1d: torch.Tensor) -> None:
        centers, spacing = _quantile_centers_and_spacing(x_1d, self.num_terms)
        self.centers.copy_(centers)
        mag = max(abs(self.init_slope) / max(spacing, 0.25), 0.5)
        slopes = torch.full_like(centers, mag)
        if self.num_terms >= 2:
            slopes[0] = -mag
        self.slopes.copy_(slopes)


class PSigMembership(torch.nn.Module):
    """1D product-of-two-sigmoids membership (smooth differentiable trapezoid).

    mu_m(x) = sigmoid(a1_m * (x - (c_m - h_m))) * sigmoid(-a2_m * (x - (c_m + h_m))),
    with strictly positive half-width `h_m` and positive slopes `a1_m, a2_m`.
    """

    def __init__(
        self,
        num_terms: int,
        min_width: float = 1e-3,
        init_width: float = 1.0,
        init_slope: float = 2.0,
    ):
        super().__init__()
        if num_terms <= 0:
            raise ValueError(f"num_terms must be positive, got {num_terms}")
        if init_width <= min_width:
            raise ValueError(f"init_width ({init_width}) must exceed min_width ({min_width})")
        if init_slope <= min_width:
            raise ValueError(f"init_slope ({init_slope}) must exceed min_width ({min_width})")
        self.num_terms = int(num_terms)
        self.min_width = float(min_width)
        self.init_slope = float(init_slope)
        self.centers = torch.nn.Parameter(_default_centers(self.num_terms))
        raw_hw = _inv_softplus(max(0.5 * init_width - self.min_width, self.min_width))
        raw_s = _inv_softplus(init_slope - self.min_width)
        self.raw_half_width = torch.nn.Parameter(torch.full((self.num_terms,), raw_hw))
        self.raw_slope_left = torch.nn.Parameter(torch.full((self.num_terms,), raw_s))
        self.raw_slope_right = torch.nn.Parameter(torch.full((self.num_terms,), raw_s))

    @property
    def half_width(self) -> torch.Tensor:
        return F.softplus(self.raw_half_width) + self.min_width

    @property
    def slope_left(self) -> torch.Tensor:
        return F.softplus(self.raw_slope_left) + self.min_width

    @property
    def slope_right(self) -> torch.Tensor:
        return F.softplus(self.raw_slope_right) + self.min_width

    def log_membership(self, x: torch.Tensor) -> torch.Tensor:
        """x: [...] -> [..., num_terms], exact log product of two sigmoids."""
        xu = x.unsqueeze(-1)
        hw = self.half_width
        left = -F.softplus(-self.slope_left * (xu - (self.centers - hw)))
        right = -F.softplus(self.slope_right * (xu - (self.centers + hw)))
        return left + right

    def membership(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_membership(x))

    @torch.no_grad()
    def init_from_1d(self, x_1d: torch.Tensor) -> None:
        centers, spacing = _quantile_centers_and_spacing(x_1d, self.num_terms)
        self.centers.copy_(centers)
        hw = max(0.5 * spacing, self.min_width + 1e-2)
        s = max(self.init_slope / max(0.5 * spacing, 0.25), self.min_width + 0.5)
        self.raw_half_width.fill_(_inv_softplus(hw - self.min_width))
        self.raw_slope_left.fill_(_inv_softplus(s - self.min_width))
        self.raw_slope_right.fill_(_inv_softplus(s - self.min_width))


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
    "triangular": TriangularMembership,
    "trapezoidal": TrapezoidalMembership,
    "sigmoid": SigmoidMembership,
    "psig": PSigMembership,
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


def _merge_membership_spec(
    spec: Optional[Union[str, Dict[str, Any], torch.nn.Module]],
    default_membership: Union[str, Dict[str, Any]],
):
    """Resolve a sensor's membership spec against `default_membership`.

    Extra parameters from `default_membership` are only inherited when the resolved membership
    `type` matches `default_membership['type']`, preventing cross-type kwargs leakage (e.g.
    passing `init_sigma` to `bell`).
    """
    _, default_cfg = _normalise_spec(default_membership)
    if spec is None:
        return dict(default_cfg)
    if isinstance(spec, torch.nn.Module):
        return spec
    kind, cfg = _normalise_spec(spec)
    if default_cfg is not None and kind == default_cfg["type"]:
        merged = dict(default_cfg)
        merged.update(cfg)
        return merged
    return cfg


def parse_sensor_specs(anfis_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract ordered enabled sensor specs from an `anfis:` config dictionary.

    Supports `anfis_cfg["sensors"]` as either:
      - a mapping `{name: {enabled: true|false, num_terms: M, membership: ...}}`, or
      - a list of `{name, num_terms, membership, [enabled]}` dicts.
    """
    raw_sensors = anfis_cfg.get("sensors")
    if not raw_sensors:
        raise ValueError("anfis.sensors must be a non-empty mapping or list of sensor specifications")
    default_mem = anfis_cfg.get("default_membership", {"type": "gaussian"})

    specs: List[Dict[str, Any]] = []
    if isinstance(raw_sensors, dict):
        items = [(name, entry) for name, entry in raw_sensors.items()]
    elif isinstance(raw_sensors, list):
        items = [(entry.get("name", f"x{i}"), entry) for i, entry in enumerate(raw_sensors)]
    else:
        raise ValueError(f"anfis.sensors must be a dict or list, got {type(raw_sensors).__name__}")

    for name, entry in items:
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"sensor {name!r} spec must be a mapping, got {type(entry).__name__}")
        if not entry.get("enabled", True):
            continue
        if "num_terms" not in entry:
            raise ValueError(f"enabled sensor {name!r} must specify 'num_terms'")
        num_terms = int(entry["num_terms"])
        if num_terms <= 0:
            raise ValueError(f"sensor {name!r} num_terms must be positive, got {num_terms}")
        mem_spec = _merge_membership_spec(entry.get("membership"), default_mem)
        specs.append({
            "name": str(name),
            "num_terms": num_terms,
            "membership": mem_spec,
        })

    if not specs:
        raise ValueError("at least one sensor in anfis.sensors must have enabled: true")
    return specs


def _build_sensor_membership(spec, num_terms: int):
    kind, cfg = _normalise_spec(spec)
    if isinstance(kind, torch.nn.Module):
        return kind, None
    if kind not in MEMBERSHIP_REGISTRY:
        raise ValueError(f"membership must be one of {tuple(MEMBERSHIP_REGISTRY)}, got {kind!r}")
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    return MEMBERSHIP_REGISTRY[kind](num_terms=num_terms, **kwargs), {"type": kind, **kwargs}


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
    """Classical Cartesian-grid multi-input multi-output Takagi-Sugeno ANFIS.

    Each sensor `i` has its own 1D membership function module with `M_i` shared linguistic terms.
    Rules are the full Cartesian product over all sensors (`num_rules = prod_i M_i`).

    Args:
        sensors: Non-empty sequence of per-sensor spec dicts:
            `{"name": str, "num_terms": int, "membership": str | dict | nn.Module}`.
        out_dim: Output dimension O (e.g. 1 for scalar reward weights, d_z for direct latents).
        consequent: "first_order", "zero_order", "mlp", a config dict `{"type": ..., ...}`, or
            an `nn.Module` mapping `[..., D] -> [..., R, O]`.
        t_norm: "product" (sum of log-memberships) or "min" (minimum log-membership across sensors).
        normalize_firing: If True (default), normalise rule strengths via softmax (standard ANFIS).
            If False, use unnormalised strengths psi_k(x) = exp(log w_k(x)) in (0, 1].
        output_activation: "identity", "softplus", "sigmoid", or "tanh" applied after aggregation.
        temperature: Divisor on aggregated log-memberships. Defaults to sqrt(in_dim) for "product"
            and 1.0 for "min".
        normalize_inputs: If True, standardise inputs with stored `x_mean` and `x_std` buffers.
        max_rules: Optional upper bound on `prod(num_terms)` to prevent accidental rule explosion.
        eps: Floor on `x_std` during input standardisation.
    """

    def __init__(
        self,
        sensors: Sequence[Dict[str, Any]],
        out_dim: int = 1,
        consequent: Union[str, Dict[str, Any], torch.nn.Module] = "first_order",
        t_norm: str = "product",
        normalize_firing: bool = True,
        output_activation: str = "identity",
        temperature: Optional[float] = None,
        normalize_inputs: bool = False,
        max_rules: Optional[int] = 1024,
        eps: float = 1e-6,
    ):
        super().__init__()
        if not sensors:
            raise ValueError("sensors must be a non-empty sequence")
        if out_dim <= 0:
            raise ValueError("out_dim must be positive")
        if t_norm not in T_NORMS:
            raise ValueError(f"t_norm must be one of {T_NORMS}, got {t_norm!r}")
        if output_activation not in OUTPUT_ACTIVATIONS:
            raise ValueError(f"output_activation must be one of {OUTPUT_ACTIVATIONS}, got {output_activation!r}")

        self.in_dim = len(sensors)
        self.out_dim = int(out_dim)
        self.t_norm = t_norm
        self.normalize_firing = bool(normalize_firing)
        self.output_activation = output_activation
        self._out_act = _activation_fn(output_activation)
        default_temp = math.sqrt(self.in_dim) if t_norm == "product" else 1.0
        resolved_temp = default_temp if temperature is None else float(temperature)
        if resolved_temp <= 0:
            raise ValueError(f"temperature must be positive, got {resolved_temp}")
        self.temperature = resolved_temp
        self.normalize_inputs = bool(normalize_inputs)
        self.max_rules = None if max_rules is None else int(max_rules)
        self.eps = float(eps)

        memberships = []
        sensor_names: List[str] = []
        term_counts: List[int] = []
        serialised_sensors: List[Dict[str, Any]] = []
        all_serialisable = True

        for i, s in enumerate(sensors):
            if not isinstance(s, dict):
                raise ValueError(f"each sensor spec must be a dict, got {type(s).__name__}")
            name = str(s.get("name", f"x{i}"))
            if "num_terms" not in s:
                raise ValueError(f"sensor {name!r} spec missing 'num_terms'")
            m_count = int(s["num_terms"])
            if m_count <= 0:
                raise ValueError(f"sensor {name!r} num_terms must be positive, got {m_count}")
            mem_mod, mem_cfg = _build_sensor_membership(s.get("membership", "gaussian"), num_terms=m_count)
            memberships.append(mem_mod)
            sensor_names.append(name)
            term_counts.append(m_count)
            if mem_cfg is None:
                all_serialisable = False
            else:
                serialised_sensors.append({"name": name, "num_terms": m_count, "membership": mem_cfg})

        num_rules = math.prod(term_counts)
        if self.max_rules is not None and num_rules > self.max_rules:
            raise ValueError(
                f"Cartesian product of num_terms {term_counts} produces {num_rules} rules, "
                f"exceeding max_rules={self.max_rules}"
            )

        self.sensor_names: Tuple[str, ...] = tuple(sensor_names)
        self.term_counts: Tuple[int, ...] = tuple(term_counts)
        self.num_rules = int(num_rules)
        self.memberships = torch.nn.ModuleList(memberships)

        grid = torch.cartesian_prod(*[torch.arange(m, dtype=torch.long) for m in term_counts])
        if self.in_dim == 1:
            grid = grid.unsqueeze(-1)
        self.register_buffer("rule_indices", grid)

        self.consequent, cons_cfg = _build_consequent(consequent, self.in_dim, self.out_dim, self.num_rules)

        self.register_buffer("x_mean", torch.zeros(self.in_dim))
        self.register_buffer("x_std", torch.ones(self.in_dim))

        self.config = (
            dict(
                sensors=serialised_sensors,
                out_dim=self.out_dim,
                consequent=cons_cfg,
                t_norm=self.t_norm,
                normalize_firing=self.normalize_firing,
                output_activation=self.output_activation,
                temperature=self.temperature,
                normalize_inputs=self.normalize_inputs,
                max_rules=self.max_rules,
                eps=self.eps,
            )
            if all_serialisable and cons_cfg is not None
            else None
        )

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], out_dim: Optional[int] = None) -> "ANFIS":
        """Build a Cartesian-grid ANFIS from a YAML/dict configuration (the `anfis:` section)."""
        sensor_specs = parse_sensor_specs(cfg)
        resolved_out = cfg.get("out_dim", 1) if out_dim is None else out_dim
        return cls(
            sensors=sensor_specs,
            out_dim=int(resolved_out),
            consequent=cfg.get("consequent", "first_order"),
            t_norm=cfg.get("t_norm", "product"),
            normalize_firing=bool(cfg.get("normalize_firing", True)),
            output_activation=cfg.get("output_activation", "identity"),
            temperature=cfg.get("temperature", None),
            normalize_inputs=bool(cfg.get("normalize_inputs", False)),
            max_rules=cfg.get("max_rules", 1024),
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
        """Initialise each sensor's linguistic terms from empirical quantiles of `x`.

        If `normalize_inputs=True`, also sets `x_mean` and `x_std` from `x` first.
        The `seed` argument is kept for call-site symmetry (quantile init is deterministic).
        """
        _ = seed
        x = x.reshape(-1, self.in_dim).to(device=self.x_mean.device, dtype=self.x_mean.dtype)
        if len(x) == 0:
            raise ValueError("init_from_data requires at least one sample")
        if self.normalize_inputs:
            self.set_input_stats(x.mean(0), x.std(0, unbiased=False))
        xn = self.normalize(x)
        for i, mem in enumerate(self.memberships):
            if hasattr(mem, "init_from_1d"):
                mem.init_from_1d(xn[:, i])

    def sensor_memberships(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Per-sensor membership values `[..., M_i]` after optional input normalisation."""
        xn = self.normalize(x)
        return [mem.membership(xn[..., i]) for i, mem in enumerate(self.memberships)]

    def log_firing_strengths(self, x: torch.Tensor) -> torch.Tensor:
        """Unnormalised log firing strengths `[..., num_rules]` on already-normalised input `x`."""
        agg: Optional[torch.Tensor] = None
        for i, mem in enumerate(self.memberships):
            log_mu_i = mem.log_membership(x[..., i])  # [..., M_i]
            term_log = log_mu_i[..., self.rule_indices[:, i]]  # [..., R]
            if agg is None:
                agg = term_log
            elif self.t_norm == "product":
                agg = agg + term_log
            else:
                agg = torch.minimum(agg, term_log)
        assert agg is not None
        return agg / self.temperature

    def _weights_from_normalised(self, xn: torch.Tensor) -> torch.Tensor:
        log_w = self.log_firing_strengths(xn)
        if self.normalize_firing:
            return F.softmax(log_w, dim=-1)
        return torch.exp(log_w)

    def firing_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Rule firing weights `[..., num_rules]`: softmax-normalised if normalize_firing else exp(log_w) in (0, 1]."""
        return self._weights_from_normalised(self.normalize(x))

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        """Evaluate ANFIS on `x` `[..., in_dim]` -> `[..., out_dim]` (or `(y, weights)` if return_weights)."""
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
        weights: State weights `[N]` or `[N, 1]` (e.g. `anfis(states)`).
        backward_repr: Frozen backward representations B(s) of shape `[N, d_z]`.
        eps: Numerical floor in L2 normalisation.

    Returns:
        Normalised latent z of shape `[d_z]` with ||z||_2 = sqrt(d_z) (when numerator is nonzero).
    """
    w = weights.reshape(-1, 1)
    if w.shape[0] != backward_repr.shape[0]:
        raise ValueError(f"weights length {w.shape[0]} != backward_repr length {backward_repr.shape[0]}")
    z = (w.T @ backward_repr) / w.shape[0]
    return (math.sqrt(backward_repr.shape[1]) * F.normalize(z, dim=1, eps=eps)).squeeze(0)
