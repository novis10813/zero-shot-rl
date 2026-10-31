"""Ensemble of forward networks for a fixed standard FB checkpoint.

B and the actor of the checkpoint stay fixed. K forward networks F_k(s, a, z), each with
its own preprocessors, one head and Polyak target, are trained from random
initialisations on independent minibatches, as one batched network, with the measure loss of agents/fb/agent.py,
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
from types import SimpleNamespace

import torch

from agents.base import load_agent
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


class EnsembleLinear(torch.nn.Module):
    """K independent linear layers applied as one batched matmul: [K, N, in] -> [K, N, out].
    Each member is initialised like torch.nn.Linear."""

    def __init__(self, members, in_features, out_features, device):
        super().__init__()
        bound = 1 / math.sqrt(in_features)
        self.weight = torch.nn.Parameter(
            torch.empty(members, in_features, out_features, device=device).uniform_(-bound, bound)
        )
        self.bias = torch.nn.Parameter(
            torch.empty(members, 1, out_features, device=device).uniform_(-bound, bound)
        )

    def forward(self, x):
        return torch.baddbmm(self.bias, x, self.weight)


def ensemble_mlp(members, input_dimension, output_dimension, hidden_dimension, hidden_layers,
                 device, preprocessor=False):
    """K copies of agents.base.AbstractMLP with ReLU activations."""
    sizes = [input_dimension] + [hidden_dimension] * hidden_layers + [output_dimension]
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(EnsembleLinear(members, a, b, device))
        if i < len(sizes) - 2 or preprocessor:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


class FEnsemble(torch.nn.Module):
    """K independent forward networks, each with the obs-action and obs-z preprocessors and
    one head of agents.fb.models.ForwardRepresentation. Inputs are [N, .], shared by all
    members, or [K, N, .]; F has shape [K, N, z_dimension]."""

    def __init__(self, members, device, observation_length, action_length, z_dimension,
                 preprocessor_hidden_dimension, preprocessor_output_dimension,
                 preprocessor_hidden_layers, preprocessor_activation, forward_hidden_dimension,
                 forward_hidden_layers, forward_activation):
        super().__init__()
        if preprocessor_activation != "relu" or forward_activation != "relu":
            raise NotImplementedError("only relu activations")
        self.init_kwargs = {key: value for key, value in locals().items()
                            if key not in ("self", "device", "__class__")}
        self.members = members
        preprocessor = {"output_dimension": preprocessor_output_dimension,
                        "hidden_dimension": preprocessor_hidden_dimension,
                        "hidden_layers": preprocessor_hidden_layers, "device": device,
                        "preprocessor": True}
        self.obs_action_preprocessor = ensemble_mlp(
            members, observation_length + action_length, **preprocessor
        )
        self.obs_z_preprocessor = ensemble_mlp(
            members, observation_length + z_dimension, **preprocessor
        )
        self.head = ensemble_mlp(members, 2 * preprocessor_output_dimension, z_dimension,
                                 forward_hidden_dimension, forward_hidden_layers, device)

    def forward(self, observation, action, z):
        if observation.dim() == 2:
            observation, action, z = (x.expand(self.members, -1, -1)
                                      for x in (observation, action, z))
        h = torch.cat([
            self.obs_action_preprocessor(torch.cat([observation, action], -1)),
            self.obs_z_preprocessor(torch.cat([observation, z], -1)),
        ], -1)
        return self.head(h)


def save_ensemble(ensemble, path, **meta):
    torch.save({"init_kwargs": ensemble.init_kwargs, "state_dict": ensemble.state_dict(),
                **meta}, path)


def load_ensemble(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    ensemble = FEnsemble(device=device, **checkpoint["init_kwargs"])
    ensemble.load_state_dict(checkpoint["state_dict"])
    return ensemble


def split(batch, members):
    """A batch of members * N transitions as members independent batches of N."""
    return SimpleNamespace(**{
        key: getattr(batch, key).view(members, -1, *getattr(batch, key).shape[1:])
        for key in ("observations", "next_observations", "actions", "discounts")
    })


def flat(fn, *xs):
    """fn of [M, .] tensors applied to [..., .] tensors."""
    shape = xs[0].shape[:-1]
    return fn(*(x.reshape(-1, x.shape[-1]) for x in xs)).view(*shape, -1)


class EnsembleTrainer:
    """Trains an FEnsemble against the fixed B and actor of an FB agent. The loss is the
    sum of the members' losses, so with Adam and element-wise gradient clipping every
    member trains as if alone."""

    def __init__(self, agent, ensemble, learning_rate=1e-4, tau=0.01, train_std=TRAIN_STD):
        self.agent, self.ensemble, self.tau, self.train_std = agent, ensemble, tau, train_std
        self.target = copy.deepcopy(ensemble).requires_grad_(False)
        self.optimiser = torch.optim.Adam(ensemble.parameters(), lr=learning_rate)

    def sample_z(self, observations):
        """z as in FB.update, for every batch of observations [..., N, obs]: uniform on the
        sphere of radius sqrt(d), a z_mix_ratio share replaced by the normalised B of other
        states in the same batch."""
        agent, shape = self.agent, observations.shape[:-1]
        zs = agent.sample_z(size=math.prod(shape)).view(*shape, -1)
        perm = torch.rand(shape, device=observations.device).argsort(-1)
        other = torch.gather(observations, -2, perm.unsqueeze(-1).expand_as(observations))
        with torch.no_grad():
            mix_zs = flat(agent.FB.backward_representation, other)
        mix_zs = math.sqrt(agent._z_dimension) * torch.nn.functional.normalize(mix_zs, dim=-1)
        mix = torch.rand(shape, device=observations.device) < agent._z_mix_ratio
        return torch.where(mix.unsqueeze(-1), mix_zs, zs)

    def losses(self, batch):
        """Loss of every member k on its batch, fields of shape [K, N, .]."""
        agent = self.agent
        zs = self.sample_z(batch.observations)
        with torch.no_grad():
            next_actions = flat(lambda o, z: agent.actor(o, z, self.train_std, sample=True)[0],
                                batch.next_observations, zs)
            B_next = flat(agent.FB.backward_representation, batch.next_observations)
            target_M = self.target(batch.next_observations, next_actions, zs) @ B_next.mT
        M = self.ensemble(batch.observations, batch.actions, zs) @ B_next.mT
        off_diagonal = ~torch.eye(M.shape[-1], dtype=torch.bool, device=M.device)
        td = (M - batch.discounts * target_M).pow(2)
        return (0.5 * td[:, off_diagonal].mean(-1)
                - torch.diagonal(M, dim1=-2, dim2=-1).mean(-1))

    def update(self, batch):
        """One step of every member; returns the losses, a [K] tensor."""
        losses = self.losses(batch)
        self.optimiser.zero_grad(set_to_none=True)
        losses.sum().backward()
        for param in self.ensemble.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimiser.step()
        with torch.no_grad():
            for param, target in zip(self.ensemble.parameters(), self.target.parameters()):
                target.lerp_(param, self.tau)
        return losses.detach()


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
        losses.append(trainer.update(split(buffer.sample(args.members * args.batch_size),
                                           args.members)))
        if step % args.log_every == 0:
            with torch.no_grad():
                F = ensemble(probe.observations, probe.actions, probe_z)
                Q = torch.einsum("knd, nd -> kn", F, probe_z)
            c = (F @ u_1).mean(1)
            print(f"step {step} loss {torch.stack(losses).mean():.3f} F^T u_1 per member "
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
