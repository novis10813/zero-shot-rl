"""Walker environment with simulator parameters set from a config.

A config has a `base` mapping, loaded at construction and on every reset, and an `events`
list. Each event has a `step` and a `set` mapping, and `set` is applied right before the
`step`-th `env.step` call of the episode (control steps counted from 0 after reset). An event
overwrites only the fields it lists, so later events stack on earlier ones. A reset restores
every field to `base`, and fields that `base` does not list to the values of the XML model.

    base:
      geom_friction: {all: 0.7}
      actuator_gear: {right_hip: 100}
    events:
      - step: 400
        set:
          geom_friction: {all: 0.2}

Fields and the values they accept (a scalar, a full list, or a mapping from element name):

    geom_friction    sliding friction: scalar or {geom name | all: scalar}; a list of 3 or
                     {geom name | all: list of 3} sets sliding, torsional and rolling friction
    actuator_gear    scalar, list of 6, or {actuator name: scalar}; the first gear column
    dof_damping      scalar, list of 9, or {joint name: scalar}; a scalar covers the root too
    opt.gravity      list of 3
    body_mass        scalar, list, or {body name: scalar}; the inertia is scaled with the mass

Unknown fields, names and shapes raise ValueError. `load_dynamics(path)` reads a yaml file
whose top level is such a config. The task reward is computed outside the environment from
`env.physics`, so these parameters cannot change it.
"""

import mujoco
import numpy as np
import yaml
from dm_control.rl import control

import custom_dmc_tasks.walker as cw

FIELDS = ("geom_friction", "actuator_gear", "dof_damping", "opt.gravity", "body_mass")
_NAME_TYPES = {
    "geom_friction": mujoco.mjtObj.mjOBJ_GEOM,
    "actuator_gear": mujoco.mjtObj.mjOBJ_ACTUATOR,
    "dof_damping": mujoco.mjtObj.mjOBJ_JOINT,
    "body_mass": mujoco.mjtObj.mjOBJ_BODY,
}


def load_dynamics(path):
    with open(path) as f:
        return parse_dynamics(yaml.safe_load(f))


def parse_dynamics(config):
    """Check the shape of a config and return {"base": dict, "events": list sorted by step}."""
    config = config or {}
    extra = set(config) - {"base", "events"}
    if extra:
        raise ValueError(f"Unknown dynamics keys {sorted(extra)}")
    base = config.get("base") or {}
    _check_fields(base, "base")
    events = []
    for i, event in enumerate(config.get("events") or []):
        if set(event) != {"step", "set"}:
            raise ValueError(f"events[{i}] needs exactly `step` and `set`, got {sorted(event)}")
        if not isinstance(event["step"], int) or isinstance(event["step"], bool) or event["step"] < 0:
            raise ValueError(f"events[{i}].step must be a non-negative integer")
        _check_fields(event["set"], f"events[{i}].set")
        events.append({"step": event["step"], "set": event["set"]})
    return {"base": base, "events": sorted(events, key=lambda e: e["step"])}


def _check_fields(fields, where):
    unknown = set(fields) - set(FIELDS)
    if unknown:
        raise ValueError(f"Unknown fields {sorted(unknown)} in {where}, expected {FIELDS}")


def _element_ids(model, field, mapping):
    """Resolve mapping keys to element ids of the field's object type."""
    ids = {}
    for name in mapping:
        if field == "geom_friction" and name == "all":
            ids[name] = slice(None)
            continue
        i = mujoco.mj_name2id(model.ptr, _NAME_TYPES[field], name)
        if i < 0:
            raise ValueError(f"{field}: no element named {name!r}")
        ids[name] = i
    return ids


def _values(value, size, field):
    value = np.asarray(value, dtype=float)
    if value.ndim == 0:
        return np.full(size, float(value))
    if value.shape != (size,):
        raise ValueError(f"{field}: expected {size} values, got shape {value.shape}")
    return value


def _apply_field(physics, field, value, nominal):
    model = physics.model
    if field == "opt.gravity":
        gravity = np.asarray(value, dtype=float)
        if gravity.shape != (3,):
            raise ValueError("opt.gravity: expected a list of 3 values")
        model.opt.gravity[:] = gravity
        return

    if field == "geom_friction":
        target = model.geom_friction
    elif field == "actuator_gear":
        target = model.actuator_gear
    elif field == "dof_damping":
        target = model.dof_damping
    else:
        target = model.body_mass

    if isinstance(value, dict):
        for name, i in _element_ids(model, field, value).items():
            _set_element(model, field, target, i, value[name], nominal)
        return

    if field == "geom_friction":
        row = np.asarray(value, dtype=float)
        if row.ndim == 0:
            target[:, 0] = row
        elif row.shape == (3,):
            target[:] = row
        else:
            raise ValueError("geom_friction: expected a scalar, 3 values or a mapping")
    elif field == "actuator_gear":
        target[:, 0] = _values(value, model.nu, field)
    elif field == "dof_damping":
        target[:] = _values(value, model.nv, field)
    else:
        _set_masses(model, np.arange(model.nbody), _values(value, model.nbody, field), nominal)


def _set_element(model, field, target, i, value, nominal):
    if field == "geom_friction":
        row = np.asarray(value, dtype=float)
        if row.ndim == 0:
            target[i, 0] = row
        elif row.shape == (3,):
            target[i] = row
        else:
            raise ValueError("geom_friction: expected a scalar or 3 values per element")
    elif field == "actuator_gear":
        target[i, 0] = float(value)
    elif field == "dof_damping":
        target[model.jnt_dofadr[i]] = float(value)
    else:
        _set_masses(model, np.array([i]), np.array([float(value)]), nominal)


def _set_masses(model, bodies, masses, nominal):
    """Scale each body's inertia by new mass over XML mass, and refresh derived constants."""
    for b, m in zip(bodies, masses):
        if m <= 0:
            raise ValueError("body_mass: masses must be positive")
        model.body_mass[b] = m
        model.body_inertia[b] = nominal["body_inertia"][b] * m / nominal["body_mass"][b]


class ShiftedWalkerEnv(control.Environment):
    """dm_control environment whose simulator parameters follow a dynamics config."""

    def __init__(self, physics, task, dynamics=None, **kwargs):
        super().__init__(physics, task, **kwargs)
        self._dynamics = parse_dynamics(dynamics)
        model = self.physics.model
        self._nominal = {
            "geom_friction": model.geom_friction.copy(),
            "actuator_gear": model.actuator_gear.copy(),
            "dof_damping": model.dof_damping.copy(),
            "opt.gravity": model.opt.gravity.copy(),
            "body_mass": model.body_mass.copy(),
            "body_inertia": model.body_inertia.copy(),
        }
        self._n_steps = 0
        # Validate names and shapes now instead of in the middle of an episode.
        self._load_base()
        for event in self._dynamics["events"]:
            self._apply(event["set"])
        self._load_base()

    def _apply(self, fields):
        for field, value in fields.items():
            _apply_field(self.physics, field, value, self._nominal)
        if "body_mass" in fields:
            self._refresh_constants()

    def _load_base(self):
        model = self.physics.model
        model.geom_friction[:] = self._nominal["geom_friction"]
        model.actuator_gear[:] = self._nominal["actuator_gear"]
        model.dof_damping[:] = self._nominal["dof_damping"]
        model.opt.gravity[:] = self._nominal["opt.gravity"]
        changed_mass = not np.array_equal(model.body_mass, self._nominal["body_mass"])
        model.body_mass[:] = self._nominal["body_mass"]
        model.body_inertia[:] = self._nominal["body_inertia"]
        self._apply(self._dynamics["base"])
        if changed_mass and "body_mass" not in self._dynamics["base"]:
            self._refresh_constants()

    def _refresh_constants(self):
        """Recompute mass-derived model constants on a scratch MjData, keeping the episode state."""
        model = self.physics.model.ptr
        mujoco.mj_setConst(model, mujoco.MjData(model))

    def reset(self):
        self._load_base()
        self._n_steps = 0
        return super().reset()

    def step(self, action):
        if not self._reset_next_step:
            for event in self._dynamics["events"]:
                if event["step"] == self._n_steps:
                    self._apply(event["set"])
            self._n_steps += 1
        return super().step(action)


def make_shifted_env(seed, dynamics):
    """The flip walker of cdmc.make("walker", "flip") with flat observations and `dynamics`."""
    physics = cw.Physics.from_xml_string(*cw.get_model_and_assets())
    task = cw.PlanarWalker(move_speed=cw._RUN_SPEED, flip=True, random=seed)  # pylint: disable=protected-access
    return ShiftedWalkerEnv(
        physics,
        task,
        dynamics=dynamics,
        time_limit=cw._DEFAULT_TIME_LIMIT,  # pylint: disable=protected-access
        control_timestep=cw._CONTROL_TIMESTEP,  # pylint: disable=protected-access
        flat_observation=True,
    )
