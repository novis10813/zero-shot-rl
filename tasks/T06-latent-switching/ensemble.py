"""Ensemble of forward networks for a fixed standard FB checkpoint.

B and the actor of the checkpoint stay fixed. K forward networks F_k(s, a, z), each with
its own preprocessors, one head, optimiser and Polyak target, are trained from random
initialisations on independent minibatches with the measure loss of agents/fb/agent.py,
without the minimum over heads:

    M_k(s, a, s'') = F_k(s, a, z)^T B(s'')
    loss_k = 1/2 E_{s'' != s'}[(M_k(s, a, s'') - discount Fbar_k(s', a', z)^T B(s''))^2]
             - E[M_k(s, a, s')],     a' = pi(s', z) with the training noise

with Fbar_k the target of member k and z drawn as in FB training. The spread of
F_k(s, a, z_r)^T z_hat_r over k is a trust signal for the value of the checkpoint.
Every --save_every steps it writes <out stem>_<step>.pt, and <out> at the end; load
them with load_ensemble.

Usage (repo root):
    PYTHONPATH=.:tasks/T06-latent-switching uv run python -m ensemble \
        --checkpoint <fb.pt> --dataset_path <dataset.npz> --out <ensemble.pt>
"""

import copy
import math
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from agents.base import load_agent
from agents.fb.base import AbstractPreprocessor, ForwardModel
from agents.fb.replay_buffer import FBReplayBuffer
from rewards import RewardFunctionConstructor
from utils import set_seed_everywhere
from zero_shot_rl.provenance import sha256

TRAIN_STD = 0.2  # agents/fb/config.yaml std_dev_schedule
ARCHITECTURE = (  # constructor arguments of FB that define one forward network
    "observation_length", "action_length", "z_dimension", "preprocessor_hidden_dimension",
    "preprocessor_output_dimension", "preprocessor_hidden_layers", "preprocessor_activation",
    "forward_hidden_dimension", "forward_hidden_layers", "forward_activation",
)


class ForwardNet(torch.nn.Module):
    """One forward network: the obs-action and obs-z preprocessors and one head of
    agents.fb.models.ForwardRepresentation."""

    def __init__(self, observation_length, action_length, z_dimension,
                 preprocessor_hidden_dimension, preprocessor_output_dimension,
                 preprocessor_hidden_layers, preprocessor_activation, forward_hidden_dimension,
                 forward_hidden_layers, forward_activation, device):
        super().__init__()
        preprocessor = {
            "observation_length": observation_length,
            "hidden_dimension": preprocessor_hidden_dimension,
            "feature_space_dimension": preprocessor_output_dimension,
            "hidden_layers": preprocessor_hidden_layers,
            "activation": preprocessor_activation,
            "device": device,
        }
        self.obs_action_preprocessor = AbstractPreprocessor(
            concatenated_variable_length=action_length, **preprocessor
        )
        self.obs_z_preprocessor = AbstractPreprocessor(
            concatenated_variable_length=z_dimension, **preprocessor
        )
        self.head = ForwardModel(
            preprocessor_feature_space_dimension=preprocessor_output_dimension,
            number_of_preprocessed_features=2,
            z_dimension=z_dimension,
            hidden_dimension=forward_hidden_dimension,
            hidden_layers=forward_hidden_layers,
            device=device,
            activation=forward_activation,
        )

    def forward(self, observation, action, z):
        h = torch.cat([
            self.obs_action_preprocessor(torch.cat([observation, action], -1)),
            self.obs_z_preprocessor(torch.cat([observation, z], -1)),
        ], -1)
        return self.head(h)


class FEnsemble(torch.nn.Module):
    """K independent forward networks; forward returns F of shape [K, N, z_dimension]."""

    def __init__(self, members, device, **architecture):
        super().__init__()
        self.init_kwargs = {"members": members, **architecture}
        self.members = torch.nn.ModuleList(
            ForwardNet(device=device, **architecture) for _ in range(members)
        )

    def forward(self, observation, action, z):
        return torch.stack([member(observation, action, z) for member in self.members])


def save_ensemble(ensemble, path, **meta):
    torch.save({"init_kwargs": ensemble.init_kwargs, "state_dict": ensemble.state_dict(),
                **meta}, path)


def load_ensemble(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    ensemble = FEnsemble(device=device, **checkpoint["init_kwargs"])
    ensemble.load_state_dict(checkpoint["state_dict"])
    return ensemble


class EnsembleTrainer:
    """Trains an FEnsemble against the fixed B and actor of an FB agent."""

    def __init__(self, agent, ensemble, learning_rate=1e-4, tau=0.01, train_std=TRAIN_STD):
        self.agent, self.ensemble, self.tau, self.train_std = agent, ensemble, tau, train_std
        self.target = copy.deepcopy(ensemble).requires_grad_(False)
        self.optimisers = [torch.optim.Adam(m.parameters(), lr=learning_rate)
                           for m in ensemble.members]

    def sample_z(self, observations):
        """z as in FB.update: uniform on the sphere of radius sqrt(d), a z_mix_ratio share
        replaced by the normalised B of other states in the batch."""
        agent, n = self.agent, len(observations)
        zs = agent.sample_z(size=n)
        backward_input = observations[torch.randperm(n)]
        mix = np.where(np.random.rand(n) < agent._z_mix_ratio)[0]
        with torch.no_grad():
            mix_zs = agent.FB.backward_representation(backward_input[mix])
        zs[mix] = math.sqrt(agent._z_dimension) * torch.nn.functional.normalize(mix_zs, dim=1)
        return zs

    def member_loss(self, k, batch):
        agent = self.agent
        zs = self.sample_z(batch.observations)
        with torch.no_grad():
            next_actions, _ = agent.actor(batch.next_observations, zs, self.train_std,
                                          sample=True)
            B_next = agent.FB.backward_representation(batch.next_observations)
            target_M = self.target.members[k](batch.next_observations, next_actions, zs) @ B_next.T
        M = self.ensemble.members[k](batch.observations, batch.actions, zs) @ B_next.T
        off_diagonal = ~torch.eye(len(M), dtype=torch.bool, device=M.device)
        return (0.5 * (M - batch.discounts * target_M)[off_diagonal].pow(2).mean()
                - M.diag().mean())

    def update(self, batches):
        """One step of every member, member k on batches[k]; returns the losses."""
        losses = []
        for k, (batch, optimiser) in enumerate(zip(batches, self.optimisers)):
            loss = self.member_loss(k, batch)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            for param in self.ensemble.members[k].parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            optimiser.step()
            losses.append(loss.item())
        for param, target in zip(self.ensemble.parameters(), self.target.parameters()):
            target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)
        return losses


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--transitions", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=5000)
    parser.add_argument("--save_every", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed_everywhere(args.seed)
    agent = load_agent(Path(args.checkpoint), device).requires_grad_(False)
    agent.eval()
    buffer = FBReplayBuffer(
        reward_constructor=RewardFunctionConstructor(
            domain_name="walker", task_names=["stand"], seed=args.seed, device=device
        ),
        dataset_path=Path(args.dataset_path),
        discount=agent.FB._discount,
        transitions=args.transitions,
        device=device,
    )
    architecture = {key: agent._init_kwargs[key] for key in ARCHITECTURE}
    ensemble = FEnsemble(members=args.members, device=device, **architecture)
    trainer = EnsembleTrainer(agent, ensemble, args.learning_rate, args.tau)
    meta = {"checkpoint": args.checkpoint, "checkpoint_sha256": sha256(Path(args.checkpoint)),
            "seed": args.seed}

    # fixed batch for the log: value of the constant reward, exact 1/(1-discount)
    with torch.no_grad():
        u_1 = agent.FB.backward_representation(buffer.storage["next_observations"]).mean(0)
    probe = buffer.sample(2048)
    probe_z = trainer.sample_z(probe.observations)
    with torch.no_grad():
        F1, F2 = agent.FB.forward_representation(probe.observations, probe.actions, probe_z)
    print(f"FB checkpoint: F^T u_1 {torch.min(F1 @ u_1, F2 @ u_1).mean():.2f}, "
          f"1/(1-discount) {1 / (1 - agent.FB._discount):.1f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    losses, start = [], time.time()
    for step in range(1, args.steps + 1):
        losses.append(trainer.update([buffer.sample(args.batch_size)
                                      for _ in range(args.members)]))
        if step % args.log_every == 0:
            with torch.no_grad():
                F = ensemble(probe.observations, probe.actions, probe_z)
                Q = torch.einsum("knd, nd -> kn", F, probe_z)
            c = (F @ u_1).mean(1)
            print(f"step {step} loss {np.mean(losses):.3f} F^T u_1 per member "
                  f"{' '.join(f'{v:.2f}' for v in c.tolist())} std_k Q {Q.std(0).mean():.3f} "
                  f"{(time.time() - start) / step * 1000:.1f} ms/step", flush=True)
            losses = []
        if step % args.save_every == 0 and step < args.steps:
            save_ensemble(ensemble, out.with_name(f"{out.stem}_{step}{out.suffix}"),
                          step=step, **meta)
    save_ensemble(ensemble, out, step=args.steps, **meta)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
