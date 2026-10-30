"""Walker environment and deterministic standard FB actor shared by the rollout entries."""

import subprocess

import numpy as np
import torch

import custom_dmc_tasks as cdmc


def make_env(seed):
    return cdmc.make(
        domain="walker",
        task="flip",
        environment_kwargs={"flat_observation": True},
        task_kwargs={"random": seed},
    )


def policy(agent, obs, z):
    """Deterministic action: clipped actor mean."""
    action, _ = agent.actor(obs, z, 0.2, sample=False)
    return action.clamp(-1.0, 1.0)


def forward(agent, obs, actions, z):
    """F(s, a, z) averaged over the two heads."""
    F1, F2 = agent.FB.forward_representation(observation=obs, action=actions, z=z)
    return (F1 + F2) / 2


def render_episode(agent, z_np):
    """Frames of one 1000-step episode from environment seed 0, every second step.

    Rendering does not work in forked workers. Batch-size and numerical differences
    can change trajectories, so this need not reproduce evaluated episode 0 exactly.
    """
    z = torch.as_tensor(z_np, dtype=torch.float32).unsqueeze(0)
    env = make_env(0)
    frames = []
    try:
        obs = env.reset().observation["observations"]
        for step in range(1000):
            with torch.no_grad():
                action = policy(agent, torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0), z)
            obs = env.step(action.numpy()[0]).observation["observations"]
            if step % 2 == 0:
                frames.append(env.physics.render(height=240, width=320, camera_id=0))
    finally:
        env.close()
    return frames


def write_video(frames, path):
    h, w, _ = frames[0].shape
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", "30", "-i", "-", "-pix_fmt", "yuv420p", str(path)],
        input=np.stack(frames).tobytes(), check=True,
    )
