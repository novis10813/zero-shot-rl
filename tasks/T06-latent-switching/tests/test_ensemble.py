"""The forward-network ensemble against the fixed T05 seed 0 FB checkpoint."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from agents.base import load_agent
from ensemble import ARCHITECTURE, EnsembleTrainer, FEnsemble, load_ensemble, save_ensemble

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "artifacts/T04-mcfb-pessimism/zsrl-32/800000.pt"
CPU = torch.device("cpu")


def batch(agent, n=64):
    obs_len, act_len = agent._init_kwargs["observation_length"], agent._init_kwargs["action_length"]
    return SimpleNamespace(observations=torch.randn(n, obs_len),
                           next_observations=torch.randn(n, obs_len),
                           actions=torch.rand(n, act_len) * 2 - 1,
                           discounts=torch.full((n, 1), 0.98))


@unittest.skipUnless(CHECKPOINT.exists(), f"needs {CHECKPOINT}")
class EnsembleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.agent = load_agent(CHECKPOINT, CPU).requires_grad_(False)
        cls.architecture = {key: cls.agent._init_kwargs[key] for key in ARCHITECTURE}

    def test_only_members_change(self):
        ensemble = FEnsemble(members=3, device=CPU, **self.architecture)
        trainer = EnsembleTrainer(self.agent, ensemble)
        agent_before = {k: v.clone() for k, v in self.agent.state_dict().items()}
        member_before = [{k: v.clone() for k, v in m.state_dict().items()}
                         for m in ensemble.members]
        target_before = {k: v.clone() for k, v in trainer.target.state_dict().items()}
        losses = trainer.update([batch(self.agent) for _ in range(3)])
        self.assertEqual(len(losses), 3)
        for key, value in self.agent.state_dict().items():
            torch.testing.assert_close(value, agent_before[key], rtol=0, atol=0)
        for member, before in zip(ensemble.members, member_before):
            self.assertTrue(any(not torch.equal(v, before[k])
                                for k, v in member.state_dict().items()))
        # the target moves tau of the way to the updated members
        online = ensemble.state_dict()
        for key, value in trainer.target.state_dict().items():
            torch.testing.assert_close(value, 0.01 * online[key] + 0.99 * target_before[key])

    def test_members_independent(self):
        ensemble = FEnsemble(members=2, device=CPU, **self.architecture)
        b = batch(self.agent)
        z = self.agent.sample_z(len(b.observations))
        with torch.no_grad():
            F = ensemble(b.observations, b.actions, z)
        self.assertEqual(F.shape, (2, len(b.observations), self.architecture["z_dimension"]))
        self.assertFalse(torch.allclose(F[0], F[1]))

    def test_sample_z_on_sphere(self):
        trainer = EnsembleTrainer(self.agent, FEnsemble(members=1, device=CPU,
                                                        **self.architecture))
        z = trainer.sample_z(batch(self.agent, 256).observations)
        d = self.architecture["z_dimension"]
        torch.testing.assert_close(z.norm(dim=1), torch.full((256,), d**0.5))

    def test_save_load(self):
        ensemble = FEnsemble(members=2, device=CPU, **self.architecture)
        b = batch(self.agent)
        z = self.agent.sample_z(len(b.observations))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ensemble.pt"
            save_ensemble(ensemble, path, step=1)
            loaded = load_ensemble(path, CPU)
        with torch.no_grad():
            torch.testing.assert_close(loaded(b.observations, b.actions, z),
                                       ensemble(b.observations, b.actions, z))


if __name__ == "__main__":
    unittest.main()
