"""Walker environment and deterministic standard FB actor shared by the rollout entries."""

import subprocess

import numpy as np
import torch

import custom_dmc_tasks as cdmc
from zero_shot_rl.env.dynamics import make_shifted_env

STEPS = 1000  # episode length


def make_env(seed, dynamics=None):
    """`dynamics` is a config of zero_shot_rl.env.dynamics; None gives the stock environment."""
    if dynamics is not None:
        return make_shifted_env(seed, dynamics)
    return cdmc.make(
        domain="walker",
        task="flip",
        environment_kwargs={"flat_observation": True},
        task_kwargs={"random": seed},
    )


def first_state(seed):
    """Observation after the reset of make_env(seed), and the actuator names."""
    env = make_env(seed)
    obs = env.reset().observation["observations"]
    joints = list(env.physics.named.data.ctrl.axes.row.names)
    return obs, joints


def set_physics(env, state):
    """reset_context runs mj_forward, so position-derived quantities match the state."""
    env.reset()
    with env.physics.reset_context():
        env.physics.set_state(state)


def policy(agent, obs, z):
    """Deterministic action: clipped actor mean."""
    action, _ = agent.actor(obs, z, 0.2, sample=False)
    return action.clamp(-1.0, 1.0)


def act(agent, obs, z):
    """`policy` for one observation: numpy [d_s] and tensor z [d_z] -> numpy [d_a]."""
    with torch.no_grad():
        o = torch.as_tensor(obs, dtype=torch.float32, device=z.device).unsqueeze(0)
        return policy(agent, o, z.unsqueeze(0))[0].cpu().numpy()


def run_episodes(agent, z, envs, reward_fn=None, stride=None, physics=False, frame_stride=None):
    """Deterministic STEPS-step episodes of `envs` from their next reset, with one batched
    `policy` call per step. Returns a dict whose arrays have the env on the first axis:

        actions       [n, STEPS, d_a]
        rewards       [n, STEPS] task reward after each step, return [n]  (with reward_fn)
        observations  [n, STEPS // stride, d_s] observation before the action of every
                      stride-th step, physics [n, STEPS // stride, d_p]  (with physics=True)
        frames        n lists of frames after the action of every frame_stride-th step
    """
    n = len(envs)
    obs = np.stack([env.reset().observation["observations"] for env in envs])
    z = z.expand(n, -1)
    actions, observations, states = [], [], []
    rewards, frames = [[] for _ in envs], [[] for _ in envs]
    returns = np.zeros(n)
    for t in range(STEPS):
        if stride and t % stride == 0:
            observations.append(obs.copy())
            if physics:
                states.append(np.stack([env.physics.get_state().copy() for env in envs]))
        with torch.no_grad():
            a = policy(agent, torch.as_tensor(obs, dtype=torch.float32, device=z.device), z)
        a = a.cpu().numpy()
        actions.append(a)
        for i, env in enumerate(envs):
            obs[i] = env.step(a[i]).observation["observations"]
            if reward_fn is not None:
                r = reward_fn(env.physics)
                rewards[i].append(r)
                returns[i] += r
            if frame_stride and t % frame_stride == 0:
                frames[i].append(env.physics.render(height=240, width=320, camera_id=0))
    out = {"actions": np.stack(actions, 1)}
    if reward_fn is not None:
        out["rewards"] = np.stack([np.array(r) for r in rewards])
        out["return"] = returns
    if stride:
        out["observations"] = np.stack(observations, 1)
        if physics:
            out["physics"] = np.stack(states, 1)
    if frame_stride:
        out["frames"] = frames
    return out


def forward(agent, obs, actions, z):
    """F(s, a, z) averaged over the two heads."""
    F1, F2 = agent.FB.forward_representation(observation=obs, action=actions, z=z)
    return (F1 + F2) / 2


def render_episode(agent, z_np):
    """Frames of one 1000-step episode from environment seed 0, every second step.

    Rendering does not work in forked workers. Batch-size and numerical differences
    can change trajectories, so this need not reproduce evaluated episode 0 exactly.
    """
    env = make_env(0)
    try:
        z = torch.as_tensor(z_np, dtype=torch.float32)
        return run_episodes(agent, z, [env], frame_stride=2)["frames"][0]
    finally:
        env.close()


def write_video(frames, path):
    h, w, _ = frames[0].shape
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", "30", "-i", "-", "-pix_fmt", "yuv420p", str(path)],
        input=np.stack(frames).tobytes(), check=True,
    )
