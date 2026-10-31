"""The forward-network ensemble against the fixed T05 seed 0 FB checkpoint."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from agents.base import load_agent
from agents.fb.base import AbstractPreprocessor
from ensemble import (ARCHITECTURE, EnsembleTrainer, FEnsemble, ensemble_mlp, load_ensemble,
                      save_ensemble)

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "artifacts/T04-mcfb-pessimism/zsrl-32/800000.pt"
CPU = torch.device("cpu")


def batch(agent, members, n=64):
    obs_len, act_len = agent._init_kwargs["observation_length"], agent._init_kwargs["action_length"]
    return SimpleNamespace(observations=torch.randn(members, n, obs_len),
                           next_observations=torch.randn(members, n, obs_len),
                           actions=torch.rand(members, n, act_len) * 2 - 1,
                           discounts=torch.full((members, n, 1), 0.98))


class EnsembleMLPTest(unittest.TestCase):
    def test_member_matches_abstract_mlp(self):
        torch.manual_seed(0)
        net = ensemble_mlp(3, 7, 5, 16, 2, CPU, preprocessor=True)
        reference = AbstractPreprocessor(observation_length=4, concatenated_variable_length=3,
                                         hidden_dimension=16, feature_space_dimension=5,
                                         hidden_layers=2, activation="relu", device=CPU)
        linears = [m for m in reference.trunk if isinstance(m, torch.nn.Linear)]
        self.assertEqual([isinstance(m, torch.nn.ReLU) for m in net],
                         [isinstance(m, torch.nn.ReLU) for m in reference.trunk])
        for layer, linear in zip([m for m in net if not isinstance(m, torch.nn.ReLU)], linears):
            linear.weight.data.copy_(layer.weight[1].T)
            linear.bias.data.copy_(layer.bias[1, 0])
        x = torch.randn(3, 10, 7)
        torch.testing.assert_close(net(x)[1], reference(x[1]))


@unittest.skipUnless(CHECKPOINT.exists(), f"needs {CHECKPOINT}")
class EnsembleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.agent = load_agent(CHECKPOINT, CPU).requires_grad_(False)
        cls.architecture = {key: cls.agent._init_kwargs[key] for key in ARCHITECTURE}

    def ensemble(self, members):
        return FEnsemble(members=members, device=CPU, **self.architecture)

    def test_only_members_change(self):
        ensemble = self.ensemble(3)
        trainer = EnsembleTrainer(self.agent, ensemble)
        agent_before = {k: v.clone() for k, v in self.agent.state_dict().items()}
        online_before = {k: v.clone() for k, v in ensemble.state_dict().items()}
        target_before = {k: v.clone() for k, v in trainer.target.state_dict().items()}
        losses = trainer.update(batch(self.agent, 3))
        self.assertEqual(losses.shape, (3,))
        for key, value in self.agent.state_dict().items():
            torch.testing.assert_close(value, agent_before[key], rtol=0, atol=0)
        online = ensemble.state_dict()
        for key, value in online.items():
            for k in range(3):  # every member moves
                self.assertFalse(torch.equal(value[k], online_before[key][k]))
        # the target moves tau of the way to the updated members
        for key, value in trainer.target.state_dict().items():
            torch.testing.assert_close(value, 0.01 * online[key] + 0.99 * target_before[key])

    def test_member_loss_touches_only_its_member(self):
        ensemble = self.ensemble(3)
        trainer = EnsembleTrainer(self.agent, ensemble)
        trainer.losses(batch(self.agent, 3))[1].backward()
        for param in ensemble.parameters():
            self.assertTrue(param.grad[1].abs().sum() > 0)
            self.assertEqual(param.grad[0].abs().sum(), 0)
            self.assertEqual(param.grad[2].abs().sum(), 0)

    def test_members_independent(self):
        ensemble = self.ensemble(2)
        b = batch(self.agent, 1)
        obs, actions = b.observations[0], b.actions[0]
        z = self.agent.sample_z(len(obs))
        with torch.no_grad():
            F = ensemble(obs, actions, z)
        self.assertEqual(F.shape, (2, len(obs), self.architecture["z_dimension"]))
        self.assertFalse(torch.allclose(F[0], F[1]))

    def test_sample_z_on_sphere(self):
        trainer = EnsembleTrainer(self.agent, self.ensemble(1))
        z = trainer.sample_z(batch(self.agent, 2, 256).observations)
        d = self.architecture["z_dimension"]
        self.assertEqual(z.shape, (2, 256, d))
        torch.testing.assert_close(z.norm(dim=-1), torch.full((2, 256), d**0.5))

    def test_save_load(self):
        ensemble = self.ensemble(2)
        b = batch(self.agent, 2)
        z = self.agent.sample_z(2 * 64).view(2, 64, -1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ensemble.pt"
            save_ensemble(ensemble, path, step=1)
            loaded = load_ensemble(path, CPU)
        with torch.no_grad():
            torch.testing.assert_close(loaded(b.observations, b.actions, z),
                                       ensemble(b.observations, b.actions, z))


if __name__ == "__main__":
    unittest.main()
