"""Component tests for zero_shot_rl.online.value_correction and its entry module."""

import math
import tempfile
import unittest
from pathlib import Path
from argparse import Namespace
from unittest import mock

import torch

from zero_shot_rl.env.walker import policy
from zero_shot_rl.eval import value_correction as entry
from zero_shot_rl.eval import value_correction_checkpoints as checkpoints
from zero_shot_rl.online.value_correction import Buffer, QAdapt, reward_scale

D_S, D_A, D_Z = 24, 6, 5


class FakeFB(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = torch.nn.ModuleList(torch.nn.Linear(D_S + D_A + D_Z, D_Z) for _ in range(2))
        self.backward_net = torch.nn.Linear(D_S, D_Z)

    def forward_representation(self, observation, action, z):
        x = torch.cat([observation, action, z], dim=-1)
        return tuple(head(x) for head in self.heads)

    def backward_representation(self, observation):
        return self.backward_net(observation)


class FakeAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.FB = FakeFB()
        self.actor_net = torch.nn.Linear(D_S + D_Z, D_A)

    def actor(self, obs, z, std, sample=False):
        return 2 * torch.tanh(self.actor_net(torch.cat([obs, z], dim=-1))), None


def make(variant, **kwargs):
    torch.manual_seed(0)
    agent = FakeAgent()
    z = math.sqrt(D_Z) * torch.nn.functional.normalize(torch.randn(D_Z), dim=0)
    options = {"candidates": 8, "hidden": 32, **kwargs}
    return agent, z, QAdapt(agent, z, 0.3, variant, D_S, D_A, **options)


def batch(n, reward=1.0, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, D_S, generator=g), torch.rand(n, D_A, generator=g) * 2 - 1,
            torch.full((n,), reward), torch.randn(n, D_S, generator=g))


class TestQAdapt(unittest.TestCase):
    def test_candidates(self):
        agent, z, model = make("direct")
        obs = torch.randn(4, D_S)
        candidates = model.propose(obs, torch.Generator().manual_seed(0))
        self.assertEqual(candidates.shape, (4, 8, D_A))
        with torch.no_grad():
            torch.testing.assert_close(candidates[:, 0], policy(agent, obs, z.expand(4, -1)))
        self.assertLessEqual(candidates.abs().max().item(), 1.0)
        self.assertGreater((candidates[:, 1:] - candidates[:, :1]).abs().max().item(), 0.0)

    def test_radius_bounds_the_offsets(self):
        obs = torch.randn(16, D_S)
        _, _, model = make("direct", candidates=64, radius=0.1)
        near = model.propose(obs, torch.Generator().manual_seed(0))
        offsets = (near - near[:, :1]).norm(dim=-1)
        self.assertLessEqual(offsets.max().item(), 0.1 + 1e-6)
        self.assertGreater(offsets[:, 1:].min().item(), 0.0)
        _, _, off = make("direct", candidates=64)
        unbounded = off.propose(obs, torch.Generator().manual_seed(0))
        self.assertGreater((unbounded - unbounded[:, :1]).norm(dim=-1).max().item(), 0.1)

    def test_zero_radius_is_the_fb_action(self):
        _, _, model = make("direct", radius=0.0)
        obs = torch.randn(8, D_S)
        candidates = model.propose(obs, torch.Generator().manual_seed(0))
        torch.testing.assert_close(candidates, candidates[:, :1].expand_as(candidates))
        action, info = model.act(obs, torch.Generator().manual_seed(0))
        torch.testing.assert_close(action, candidates[:, 0])
        torch.testing.assert_close(info["shift"], torch.zeros(8))

    def test_residual_starts_as_fb_value(self):
        agent, z, model = make("residual")
        obs, actions, _, _ = batch(16)
        with torch.no_grad():
            F1, F2 = agent.FB.forward_representation(obs, actions, z.expand(16, -1))
            expected = 0.3 * torch.minimum(F1 @ z, F2 @ z)
            q = model.q(obs, actions)
        torch.testing.assert_close(q[0], expected)
        torch.testing.assert_close(q[1], expected)

    def test_act_is_greedy_and_explores(self):
        _, _, model = make("residual")
        obs = torch.randn(64, D_S)
        action, info = model.act(obs, torch.Generator().manual_seed(3))
        index, value = info["index"], info["value"]
        candidates = model.propose(obs, torch.Generator().manual_seed(3))
        values = model.values(obs, candidates)
        torch.testing.assert_close(index, values.argmax(1))
        torch.testing.assert_close(value, values.max(1).values)
        torch.testing.assert_close(action, candidates[torch.arange(64), index])
        torch.testing.assert_close(info["shift"], (action - candidates[:, 0]).norm(dim=1))
        torch.testing.assert_close(info["gain"], value - values[:, 0])
        torch.testing.assert_close(info["at_fb"], (index == 0).float())
        self.assertGreaterEqual(info["gain"].min().item(), 0.0)
        _, info = model.act(obs, torch.Generator().manual_seed(3), explore=1.0)
        self.assertGreater((info["index"] != index).float().mean().item(), 0.5)

    def test_direct_converges_to_constant_reward_value(self):
        # s' = s and a = pi(s) with one candidate, so the target is evaluated on the batch
        agent, z, model = make("direct", candidates=1, discount=0.5, tau=0.1, learning_rate=1e-3)
        g = torch.Generator().manual_seed(0)
        obs, _, rewards, _ = batch(256)
        with torch.no_grad():
            data = (obs, policy(agent, obs, z.expand(256, -1)), rewards, obs)
        for _ in range(1500):
            model.update(data, g)
        with torch.no_grad():
            q = model.q(data[0], data[1])
        torch.testing.assert_close(q, torch.full_like(q, 2.0), atol=0.05, rtol=0)

    def test_update_leaves_fb_unchanged_and_is_deterministic(self):
        losses = []
        for _ in range(2):
            agent, _, model = make("residual")
            before = [p.clone() for p in agent.parameters()]
            g = torch.Generator().manual_seed(0)
            losses.append([model.update(batch(32), g) for _ in range(5)])
            for p, p_before in zip(agent.parameters(), before):
                self.assertTrue(torch.equal(p, p_before))
                self.assertIsNone(p.grad)
        self.assertEqual(losses[0], losses[1])
        self.assertNotEqual(losses[0][0], losses[0][-1])

    def test_save_and_load_round_trip(self):
        agent, _, model = make("residual", radius=0.3, layers=3)
        g = torch.Generator().manual_seed(0)
        for _ in range(3):
            model.update(batch(32), g)
        with tempfile.TemporaryDirectory() as d:
            model.save(Path(d) / "m.pt")
            loaded = QAdapt.load(Path(d) / "m.pt", agent, D_S, D_A)
        obs = torch.randn(8, D_S)
        a, _ = model.act(obs, torch.Generator().manual_seed(5))
        b, _ = loaded.act(obs, torch.Generator().manual_seed(5))
        torch.testing.assert_close(a, b)
        self.assertEqual((loaded.radius, loaded.layers), (0.3, 3))

    def test_reward_scale(self):
        agent = FakeAgent()
        obs, rewards = torch.randn(50, D_S), torch.rand(50, 1)
        with torch.no_grad():
            z = (rewards * agent.FB.backward_representation(obs)).mean(0)
        self.assertAlmostEqual(reward_scale(agent, obs, rewards),
                               z.norm().item() / math.sqrt(D_Z), places=6)

    def test_buffer_samples_stored_transitions(self):
        buffer = Buffer(10, D_S, D_A, torch.device("cpu"))
        for i in range(3):
            buffer.add(torch.full((D_S,), float(i)), torch.zeros(D_A), float(i),
                       torch.full((D_S,), i + 1.0))
        obs, _, rewards, next_obs = buffer.sample(100, torch.Generator().manual_seed(0))
        torch.testing.assert_close(obs[:, 0], rewards)
        torch.testing.assert_close(next_obs[:, 0], rewards + 1)
        self.assertEqual(set(rewards.tolist()), {0.0, 1.0, 2.0})


class FakeEnv:
    """The parts of the Walker environment the entry module uses, without the simulator:
    the observation is noise from the seed and `physics` is the number of steps taken."""

    def __init__(self, seed, dynamics=None):
        self.generator = torch.Generator().manual_seed(seed)
        self.physics = 0

    def observation_spec(self):
        return {"observations": Namespace(shape=(D_S,))}

    def action_spec(self):
        return Namespace(shape=(D_A,))

    def _time_step(self):
        return Namespace(observation={"observations": torch.randn(D_S, generator=self.generator).numpy()})

    def reset(self):
        self.physics = 0
        return self._time_step()

    def step(self, action):
        assert action.shape == (D_A,)
        self.physics += 1
        return self._time_step()

    def close(self):
        pass


@mock.patch.multiple(entry, make_env=FakeEnv, STEPS=20, VALUE_STEPS=10)
class TestEntry(unittest.TestCase):
    def run_once(self):
        torch.manual_seed(0)
        agent = FakeAgent()
        z = math.sqrt(D_Z) * torch.nn.functional.normalize(torch.randn(D_Z), dim=0)
        args = Namespace(seed=0, train_seed=1000, dynamics=None, candidates=4, sigma=0.2, radius=0.3,
                         hidden=16, learning_rate=1e-3, tau=0.01, discount=0.98, episodes=2,
                         eval_at=[0, 2], eval_episodes=3, explore=0.3, warmup=10,
                         batch_size=8, save_every=0, out_dir=".")
        return entry.run(args, agent, z, 0.3, "residual", "walk", lambda physics: physics / 20)

    def test_eval_points_are_a_fixed_interval(self):
        self.assertEqual(entry.eval_points(100, 25), [0, 25, 50, 75, 100])
        self.assertEqual(entry.eval_points(100, 30), [0, 30, 60, 90, 100])

    def test_frozen_fb_rows(self):
        agent = FakeAgent()
        args = Namespace(seed=0, dynamics=None, eval_episodes=3)
        z = torch.ones(D_Z)
        rows = entry.frozen_rows(args, agent, z, "walk", lambda physics: physics / 20)
        self.assertEqual([r["seed"] for r in rows], [0, 1, 2])
        for row in rows:
            self.assertEqual(list(row), entry.EVAL_FIELDS)
            self.assertEqual((row["variant"], row["shift"], row["gain"], row["a0_frac"]),
                             ("fb", 0.0, 0.0, 1.0))
            self.assertAlmostEqual(row["return"], sum(range(1, 21)) / 20)

    def test_saved_critics_are_tested_like_in_training(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "q").mkdir()
            torch.manual_seed(0)
            agent = FakeAgent()
            z = math.sqrt(D_Z) * torch.nn.functional.normalize(torch.randn(D_Z), dim=0)
            args = Namespace(seed=0, train_seed=1000, dynamics=None, candidates=4, sigma=0.2,
                             radius=0.3, hidden=16, learning_rate=1e-3, tau=0.01, discount=0.98,
                             episodes=2, eval_at=[2], eval_episodes=3, explore=0.3, warmup=10,
                             batch_size=8, save_every=1, out_dir=d)
            eval_rows, _, _ = entry.run(args, agent, z, 0.3, "residual", "walk",
                                        lambda physics: physics / 20)
            found = checkpoints.find(Path(d) / "q")
            self.assertEqual([(v, t, n) for v, t, n, _ in found],
                             [("residual", "walk", n) for n in (0, 1, 2)])
            self.assertEqual([n for *_, n, _ in checkpoints.find(Path(d) / "q", [1])], [1])
            model = QAdapt.load(found[-1][3], agent, D_S, D_A)
            returns, shift, gain, a0 = entry.evaluate(args, agent, z, model,
                                                      lambda physics: physics / 20)
        last = [r for r in eval_rows if r["train_episodes"] == 2]
        self.assertEqual([r["return"] for r in last], list(returns))
        self.assertEqual([r["shift"] for r in last], list(shift))
        self.assertEqual([r["a0_frac"] for r in last], list(a0))
        self.assertTrue(all(0.0 <= x <= 1.0 for x in a0))

    def test_run_rows_and_determinism(self):
        eval_rows, train_rows, _ = self.run_once()
        self.assertEqual([(r["train_episodes"], r["seed"]) for r in eval_rows],
                         [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)])
        for row in eval_rows:
            self.assertEqual(list(row), entry.EVAL_FIELDS)
            self.assertAlmostEqual(row["return"], sum(range(1, 21)) / 20)
        self.assertEqual([r["episode"] for r in train_rows], [0, 1])
        for row in train_rows:
            self.assertEqual(list(row), entry.TRAIN_FIELDS)
            self.assertAlmostEqual(row["return"], sum(range(1, 21)) / 20)
        discounted = [sum(0.98 ** i * (t + 1 + i) / 20 for i in range(20 - t)) for t in range(10)]
        self.assertAlmostEqual(train_rows[0]["q_actual"], sum(discounted) / 10)
        self.assertTrue(math.isfinite(train_rows[0]["loss"]))
        again, train_again, _ = self.run_once()
        self.assertEqual(eval_rows, again)
        self.assertEqual(train_rows, train_again)


if __name__ == "__main__":
    unittest.main()
