"""Per-domain scalar sensor registry and environment factory for DMC domains.

Maps each supported ExORL/DMC domain (`walker`, `quadruped`, `point_mass_maze`, `jaco`) to its
scalar physics/sensor extractors so YAML configs (e.g. `configs/anfis/<domain>.yaml`) can select
any subset of sensors without mixing sensor namespaces across domains.
"""

from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

import custom_dmc_tasks as cdmc
from zero_shot_rl.env import dmc
from zero_shot_rl.env.walker import make_env as make_walker_env

Extractor = Callable[[Any], float]

SUPPORTED_DOMAINS: Tuple[str, ...] = ("walker", "quadruped", "point_mass_maze", "jaco")

DOMAIN_TASKS: Dict[str, Tuple[str, ...]] = {
    "walker": ("stand", "walk", "run", "flip"),
    "quadruped": ("stand", "jump", "roll", "roll_fast"),
    "point_mass_maze": ("reach_top_left", "reach_top_right", "reach_bottom_left", "reach_bottom_right"),
    "jaco": ("reach_top_left", "reach_top_right", "reach_bottom_left", "reach_bottom_right"),
}

_DEFAULT_DOMAIN_TASK: Dict[str, str] = {
    "walker": "flip",
    "quadruped": "jump",
    "point_mass_maze": "reach_bottom_left",
    "jaco": "reach_bottom_left",
}


def _qpos(joint: str) -> Extractor:
    return lambda p, j=joint: float(np.asarray(p.named.data.qpos[j]).item())


def _qvel(joint: str) -> Extractor:
    return lambda p, j=joint: float(np.asarray(p.named.data.qvel[j]).item())


def _xmat(body: str, component: str) -> Extractor:
    return lambda p, b=body, c=component: float(p.named.data.xmat[b, c])


def _sensor_idx(name: str, idx: int, arcsinh: bool = False) -> Extractor:
    if arcsinh:
        return lambda p, n=name, i=idx: float(np.arcsinh(p.named.data.sensordata[n][i]))
    return lambda p, n=name, i=idx: float(p.named.data.sensordata[n][i])


def _site_xpos_idx(site: str, idx: int) -> Extractor:
    return lambda p, s=site, i=idx: float(p.named.data.site_xpos[s][i])


def _build_walker_sensors() -> Dict[str, Extractor]:
    sensors: Dict[str, Extractor] = {
        # CoM / Reward physics
        "torso_height": lambda p: float(p.torso_height()),
        "torso_upright": lambda p: float(p.torso_upright()),
        "horizontal_velocity": lambda p: float(p.horizontal_velocity()),
        "angmomentum": lambda p: float(p.angmomentum()),
        "com_vz": _sensor_idx("torso_subtreelinvel", 2),
    }
    joints = (
        "rootz",
        "rootx",
        "rooty",
        "right_hip",
        "right_knee",
        "right_ankle",
        "left_hip",
        "left_knee",
        "left_ankle",
    )
    for j in joints:
        sensors[f"qvel_{j}"] = _qvel(j)
    for j in joints:
        sensors[f"qpos_{j}"] = _qpos(j)

    bodies = (
        "torso",
        "right_thigh",
        "right_leg",
        "right_foot",
        "left_thigh",
        "left_leg",
        "left_foot",
    )
    for b in bodies:
        sensors[f"orient_{b}_xx"] = _xmat(b, "xx")
        sensors[f"orient_{b}_xz"] = _xmat(b, "xz")
    return sensors


def _build_quadruped_sensors() -> Dict[str, Extractor]:
    sensors: Dict[str, Extractor] = {
        # CoM / Torso physics
        "torso_upright": lambda p: float(p.torso_upright()),
        "com_height": lambda p: float(p.com_height()),
        "com_x": _sensor_idx("center_of_mass", 0),
        "com_y": _sensor_idx("center_of_mass", 1),
        "torso_vel_x": _sensor_idx("velocimeter", 0),
        "torso_vel_y": _sensor_idx("velocimeter", 1),
        "torso_vel_z": _sensor_idx("velocimeter", 2),
        "torso_speed": lambda p: float(np.linalg.norm(p.torso_velocity())),
        # IMU sensors
        "imu_accel_x": _sensor_idx("imu_accel", 0),
        "imu_accel_y": _sensor_idx("imu_accel", 1),
        "imu_accel_z": _sensor_idx("imu_accel", 2),
        "imu_gyro_x": _sensor_idx("imu_gyro", 0),
        "imu_gyro_y": _sensor_idx("imu_gyro", 1),
        "imu_gyro_z": _sensor_idx("imu_gyro", 2),
    }

    legs = ("front_left", "front_right", "back_right", "back_left")
    axes = ("x", "y", "z")
    # Toe force/torque sensors (arcsinh-scaled to match DMC physics.force_torque())
    for leg in legs:
        for idx, axis in enumerate(axes):
            sensors[f"force_toe_{leg}_{axis}"] = _sensor_idx(f"force_toe_{leg}", idx, arcsinh=True)
    for leg in legs:
        for idx, axis in enumerate(axes):
            sensors[f"torque_toe_{leg}_{axis}"] = _sensor_idx(f"torque_toe_{leg}", idx, arcsinh=True)

    # Hinge joint positions and velocities (4 legs x 4 hinge joints)
    hinges = ("yaw", "pitch", "knee", "ankle")
    for leg in legs:
        for hinge in hinges:
            jname = f"{hinge}_{leg}"
            sensors[f"qpos_{jname}"] = _qpos(jname)
    for leg in legs:
        for hinge in hinges:
            jname = f"{hinge}_{leg}"
            sensors[f"qvel_{jname}"] = _qvel(jname)
    return sensors


def _build_point_mass_maze_sensors() -> Dict[str, Extractor]:
    return {
        "pos_x": _qpos("root_x"),
        "pos_y": _qpos("root_y"),
        "vel_x": _qvel("root_x"),
        "vel_y": _qvel("root_y"),
        "speed": lambda p: float(np.hypot(p.named.data.qvel["root_x"], p.named.data.qvel["root_y"])),
        "origin_dist": lambda p: float(np.hypot(p.named.data.qpos["root_x"], p.named.data.qpos["root_y"])),
    }


def _build_jaco_sensors() -> Dict[str, Extractor]:
    sensors: Dict[str, Extractor] = {
        # End-effector (pinch site) 3D position used by reach rewards
        "pinch_pos_x": _site_xpos_idx("jaco_arm/jaco_hand/pinchsite", 0),
        "pinch_pos_y": _site_xpos_idx("jaco_arm/jaco_hand/pinchsite", 1),
        "pinch_pos_z": _site_xpos_idx("jaco_arm/jaco_hand/pinchsite", 2),
    }
    for i in range(1, 7):
        sensors[f"qpos_joint_{i}"] = _qpos(f"jaco_arm/joint_{i}")
    for i in range(1, 7):
        sensors[f"qvel_joint_{i}"] = _qvel(f"jaco_arm/joint_{i}")
    for i in range(1, 4):
        sensors[f"qpos_finger_{i}"] = _qpos(f"jaco_arm/jaco_hand/finger_{i}")
    for i in range(1, 4):
        sensors[f"qvel_finger_{i}"] = _qvel(f"jaco_arm/jaco_hand/finger_{i}")
    axes = ("x", "y", "z")
    for i in range(1, 7):
        for idx, axis in enumerate(axes):
            sensors[f"torque_joint_{i}_{axis}"] = _sensor_idx(f"jaco_arm/joint_{i}_torque", idx)
    return sensors


SENSOR_REGISTRY: Dict[str, Dict[str, Extractor]] = {
    "walker": _build_walker_sensors(),
    "quadruped": _build_quadruped_sensors(),
    "point_mass_maze": _build_point_mass_maze_sensors(),
    "jaco": _build_jaco_sensors(),
}


def list_sensors(domain: str) -> List[str]:
    """Return ordered list of all scalar sensor names registered for `domain`."""
    if domain not in SENSOR_REGISTRY:
        raise ValueError(f"unknown domain {domain!r}; choose from {SUPPORTED_DOMAINS}")
    return list(SENSOR_REGISTRY[domain].keys())


def validate_sensors(domain: str, sensor_names: Sequence[str]) -> List[str]:
    """Validate that `domain` is supported and every sensor name exists for that domain."""
    if domain not in SENSOR_REGISTRY:
        raise ValueError(f"unknown domain {domain!r}; choose from {SUPPORTED_DOMAINS}")
    registry = SENSOR_REGISTRY[domain]
    names = list(sensor_names)
    if not names:
        raise ValueError(f"at least one sensor must be enabled for domain {domain!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate sensor names in {names!r}")
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise ValueError(
            f"unknown sensor(s) {unknown!r} for domain {domain!r}; "
            f"available: {list(registry.keys())}"
        )
    return names


def extract_sensors(physics: Any, domain: str, sensor_names: Sequence[str]) -> np.ndarray:
    """Extract a 1D float32 vector of the selected scalar sensors from a MuJoCo `physics` object."""
    if domain not in SENSOR_REGISTRY:
        raise ValueError(f"unknown domain {domain!r}; choose from {SUPPORTED_DOMAINS}")
    registry = SENSOR_REGISTRY[domain]
    return np.array([registry[name](physics) for name in sensor_names], dtype=np.float32)


def make_domain_env(domain: str, seed: int, dynamics=None):
    """Instantiate a flat-observation DMC environment for `domain`."""
    if domain not in SENSOR_REGISTRY:
        raise ValueError(f"unknown domain {domain!r}; choose from {SUPPORTED_DOMAINS}")
    if domain == "walker":
        return make_walker_env(seed, dynamics=dynamics)
    if dynamics is not None:
        raise ValueError(f"shifted dynamics YAML is only supported for domain 'walker', got {domain!r}")
    if domain == "jaco":
        return dmc.make(name=f"jaco_{_DEFAULT_DOMAIN_TASK['jaco']}", seed=seed)
    return cdmc.make(
        domain=domain,
        task=_DEFAULT_DOMAIN_TASK[domain],
        environment_kwargs={"flat_observation": True},
        task_kwargs={"random": seed},
    )
