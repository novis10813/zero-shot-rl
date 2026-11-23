"""Component and integration tests for zero_shot_rl.finetune.anfis_updaters and anfis_z algos/params."""

import copy
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agents.fb.agent import FB
from zero_shot_rl.env.dynamics import parse_dynamics
from zero_shot_rl.env.sensors import make_domain_env
from zero_shot_rl.finetune.anfis_updaters import (
    ANFISLatent,
    DirectZLatent,
    EvalScheduler,
    PhysicsStateBuffer,
    fb_q_base,
    load_latent_checkpoint,
    lola_rloo_advantages,
    lola_update_step,
    normalize_algo_type,
    normalize_param_type,
    set_env_physics,
    unnormalized_w_r,
)
from zero_shot_rl.finetune.anfis_z import DEFAULT_CONFIG, train_task, write_table6_csv
from zero_shot_rl.latent.anfis import ANFIS, weighted_z
from zero_shot_rl.online.value_correction import reward_scale


def _make_tiny_fb(obs_dim: int = 24, act_dim: int = 6, z_dim: int = 12) -> FB:
    torch.manual_seed(0)
    agent = FB(
        observation_length=obs_dim,
        action_length=act_dim,
        preprocessor_hidden_dimension=16,
        preprocessor_output_dimension=8,
        preprocessor_hidden_layers=1,
        preprocessor_activation="relu",
        z_dimension=z_dim,
        forward_hidden_dimension=16,
        forward_hidden_layers=1,
        forward_number_of_features=2,
        backward_hidden_dimension=16,
        backward_hidden_layers=1,
        actor_hidden_dimension=16,
        actor_hidden_layers=1,
        forward_activation="relu",
        backward_activation="relu",
        actor_activation="relu",
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-4,
        learning_rate_coefficient=1.0,
        orthonormalisation_coefficient=1.0,
        discount=0.98,
        batch_size=16,
        z_mix_ratio=0.5,
        gaussian_actor=False,
        std_dev_clip=0.3,
        std_dev_schedule="0.2",
        tau=0.01,
        device=torch.device("cpu"),
        name="fb",
    )
    agent.eval()
    return agent


def _tiny_anfis_cfg() -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["anfis"]["sensors"] = {
        "torso_height": {"enabled": True, "num_terms": 2},
        "torso_upright": {"enabled": True, "num_terms": 2},
    }
    cfg["warmstart"]["steps"] = 10
    cfg["eval"]["eval_episodes"] = 1
    cfg["eval"]["eval_seed_start"] = 0
    return cfg


class ANFISUpdatersUnitTest(unittest.TestCase):
    def test_normalize_names(self):
        self.assertEqual(normalize_param_type("anfis"), "anfis")
        self.assertEqual(normalize_param_type("z"), "z")
        self.assertEqual(normalize_param_type("direct_z"), "z")
        self.assertEqual(normalize_algo_type("reinforce"), "reinforce")
        self.assertEqual(normalize_algo_type("lola"), "lola")
        self.assertEqual(normalize_algo_type("rela"), "rela")
        with self.assertRaises(ValueError):
            normalize_param_type("unknown")
        with self.assertRaises(ValueError):
            normalize_algo_type("unknown")

    def test_direct_z_and_anfis_checkpoint_roundtrip(self):
        torch.manual_seed(0)
        z_dim = 12
        z_r = math.sqrt(z_dim) * F.normalize(torch.randn(z_dim), dim=0)

        dz = DirectZLatent(z_dim=z_dim, init_z=z_r)
        torch.testing.assert_close(dz.compute_z(), z_r, atol=1e-5, rtol=0)
        dz_rand = DirectZLatent(z_dim=z_dim, init_z=None, seed=7)
        torch.testing.assert_close(dz_rand.compute_z().norm(), torch.tensor(math.sqrt(z_dim)), atol=1e-5, rtol=0)
        self.assertLess(abs(float(F.cosine_similarity(dz_rand.compute_z().unsqueeze(0), z_r.unsqueeze(0)).item())), 0.99)

        with tempfile.TemporaryDirectory() as tmp:
            p_z = Path(tmp) / "walk_final.pt"
            dz.save_checkpoint(p_z, z=dz.compute_z().detach(), z_r=z_r, algo="lola")
            loaded_mod, meta = load_latent_checkpoint(p_z)
            self.assertIsInstance(loaded_mod, DirectZLatent)
            torch.testing.assert_close(loaded_mod.compute_z(), z_r, atol=1e-5, rtol=0)
            torch.testing.assert_close(meta["z"], z_r, atol=1e-5, rtol=0)

            feats = torch.randn(32, 2)
            B = torch.randn(32, z_dim)
            anfis_model = ANFIS.from_config(_tiny_anfis_cfg()["anfis"], out_dim=1)
            anfis_param = ANFISLatent(anfis_model, feats, B)
            p_a = Path(tmp) / "stand_final.pt"
            z_a = anfis_param.compute_z().detach()
            anfis_param.save_checkpoint(p_a, z=z_a, z_r=z_r, algo="rela")
            loaded_a, meta_a = load_latent_checkpoint(p_a)
            self.assertIsInstance(loaded_a, ANFIS)
            torch.testing.assert_close(meta_a["z"], z_a, atol=1e-5, rtol=0)

    def test_unnormalized_w_r_matches_reward_scale_and_fb_q_base_grad(self):
        agent = _make_tiny_fb(obs_dim=24, act_dim=6, z_dim=12)
        obs = torch.randn(64, 24)
        rewards = torch.rand(64, 1)
        with torch.no_grad():
            B = agent.FB.backward_representation(obs)
        z_r = weighted_z(rewards, B)
        w_r = unnormalized_w_r(rewards, B)
        scale = reward_scale(agent, obs, rewards)
        torch.testing.assert_close(w_r, scale * z_r, atol=1e-5, rtol=1e-4)

        for p in agent.parameters():
            p.requires_grad_(False)
        actions = torch.zeros(8, 6, requires_grad=True)
        q_min = fb_q_base(agent, obs[:8], actions, z_r, w_r, head="min")
        q_mean = fb_q_base(agent, obs[:8], actions, z_r, w_r, head="mean")
        self.assertEqual(q_min.shape, (8,))
        self.assertTrue((q_min <= q_mean + 1e-6).all())
        q_min.mean().backward()
        self.assertIsNotNone(actions.grad)
        self.assertGreater(float(actions.grad.norm().item()), 0.0)

    def test_eval_scheduler_handles_multi_milestone_jumps(self):
        recorded = []
        sched = EvalScheduler.from_budget(max_steps=5000, eval_every=2000, run_eval=lambda ms: recorded.append(list(ms)))
        sched.trigger_due(0)
        sched.trigger_due(4500)
        sched.finalize(4500)
        self.assertEqual(recorded, [[0], [2000, 4000], [5000]])
        self.assertEqual(sched.evaluated, [0, 2000, 4000, 5000])

    def test_lola_rloo_advantages_exact_formula(self):
        rets = torch.tensor([[1.0, 2.0, 5.0], [10.0, 10.0, 10.0]])
        adv = lola_rloo_advantages(rets, normalize=False)
        # For [1, 2, 5]: LOO means are [3.5, 3.0, 1.5] -> adv = [-2.5, -1.0, +3.5]
        expected = torch.tensor([[-2.5, -1.0, 3.5], [0.0, 0.0, 0.0]])
        torch.testing.assert_close(adv, expected, atol=1e-6, rtol=0)

    def test_lola_update_improves_both_anfis_and_direct_z_and_updates_sigma(self):
        for param_kind in ("anfis", "z"):
            torch.manual_seed(3)
            z_dim = 16
            v_star = F.normalize(torch.randn(z_dim), dim=0)
            if param_kind == "anfis":
                feats = torch.randn(64, 2)
                B = torch.randn(64, z_dim)
                model = ANFIS.from_config(_tiny_anfis_cfg()["anfis"], out_dim=1)
                model.init_from_data(feats, seed=99)
                latent_param = ANFISLatent(model, feats, B)
            else:
                latent_param = DirectZLatent(z_dim=z_dim, init_z=None, seed=99)

            log_std = torch.nn.Parameter(torch.full((z_dim,), math.log(0.4)))
            opt = torch.optim.Adam([
                {"params": list(latent_param.parameters()), "lr": 5e-2},
                {"params": [log_std], "lr": 1e-2},
            ])
            gen = torch.Generator().manual_seed(3)

            with torch.no_grad():
                score_before = float((latent_param.compute_z() @ v_star).item())
                log_std_before = log_std.detach().clone()

            for _ in range(30):
                with torch.no_grad():
                    mu = latent_param.compute_z()
                    sigma = torch.exp(log_std)
                    eps = torch.randn(2, 5, z_dim, generator=gen)
                    z_samples = mu.view(1, 1, -1) + sigma.view(1, 1, -1) * eps
                    z_sphere = math.sqrt(z_dim) * F.normalize(z_samples, dim=-1, eps=1e-8)
                    rets = z_sphere @ v_star
                lola_update_step(latent_param, log_std, opt, z_samples, rets, learn_std=True, grad_clip=1.0)

            with torch.no_grad():
                score_after = float((latent_param.compute_z() @ v_star).item())
            self.assertGreater(score_after, score_before + 0.5, f"failed for {param_kind}")
            self.assertGreater(float((log_std.detach() - log_std_before).abs().max().item()), 1e-4)

    def test_set_env_physics_preserves_shifted_dynamics(self):
        dyn = parse_dynamics({"base": {"geom_friction": {"all": 0.15}}})
        env = make_domain_env("walker", seed=0, dynamics=dyn)
        try:
            env.reset()
            for _ in range(4):
                ts = env.step(np.zeros(6, dtype=np.float32))
            buf = PhysicsStateBuffer(capacity=16)
            buf.add(env.physics.get_state().copy(), ts.observation["observations"].copy())
            saved_phys, _ = buf.sample(np.random.default_rng(0))

            set_env_physics(env, saved_phys)
            np.testing.assert_allclose(env.physics.get_state(), saved_phys, atol=1e-7)
            self.assertAlmostEqual(float(env.physics.model.geom_friction[0, 0]), 0.15, places=6)
        finally:
            env.close()


class ANFISUpdatersIntegrationTest(unittest.TestCase):
    """Fast end-to-end integration tests of train_task across all 3 algos and both parameterizations."""

    def setUp(self):
        self.agent = _make_tiny_fb(obs_dim=24, act_dim=6, z_dim=12)
        torch.manual_seed(0)
        self.obs = torch.randn(32, 24)
        self.feats = torch.randn(32, 2)
        self.rewards = torch.rand(32, 1)
        with torch.no_grad():
            self.B = self.agent.FB.backward_representation(self.obs).detach()
        self.reward_fn = lambda physics: float(physics.torso_upright())

    def test_reinforce_lola_rela_with_anfis_and_direct_z(self):
        cases = [
            ("reinforce", "anfis", True),
            ("reinforce", "z", True),
            ("lola", "anfis", True),
            ("lola", "z", False),  # no-bootstrap ablation
            ("rela", "anfis", True),
            ("rela", "z", False),  # scratch critic ablation
        ]
        fb_snap = {k: v.detach().clone() for k, v in self.agent.state_dict().items()}

        for algo, param, flag in cases:
            with self.subTest(algo=algo, param=param, flag=flag), tempfile.TemporaryDirectory() as tmp:
                cfg = _tiny_anfis_cfg()
                cfg["algo"] = algo
                cfg["param"] = param
                cfg["budget"]["max_env_steps"] = 60
                cfg["budget"]["eval_every_env_steps"] = 30
                if algo == "reinforce":
                    cfg["reinforce"]["pop_size"] = 2
                    cfg["reinforce"]["episode_steps"] = 15  # 30 env steps / iter
                    cfg["reinforce"]["train_episodes_per_z"] = 1
                elif algo == "lola":
                    cfg["lola"]["horizon"] = 5
                    cfg["lola"]["num_start_states"] = 2
                    cfg["lola"]["num_latents"] = 3  # 30 env steps / grad step
                    cfg["lola"]["bootstrap"] = flag
                    cfg["lola"]["init_state_prob"] = 0.5
                elif algo == "rela":
                    cfg["rela"]["use_residual"] = flag
                    cfg["rela"]["residual_hidden_dim"] = 16
                    cfg["rela"]["scratch_hidden_dim"] = 16
                    cfg["rela"]["batch_size"] = 8
                    cfg["rela"]["warmup_steps"] = 10
                    cfg["rela"]["utd"] = 2
                    cfg["rela"]["actor_update_freq"] = 2
                    cfg["rela"]["log_every_env_steps"] = 30

                out_dir = Path(tmp)
                res = train_task(
                    task="stand",
                    task_index=0,
                    domain="walker",
                    agent=self.agent,
                    features=self.feats if param == "anfis" else None,
                    backward_repr=self.B,
                    target_rewards=self.rewards,
                    reward_fn=self.reward_fn,
                    cfg=cfg,
                    dynamics=None,
                    out_dir=out_dir,
                    device=torch.device("cpu"),
                )

                # Milestones 0, 30, 60 must all be evaluated and writable to table6.csv.
                self.assertEqual(sorted(res["iqm_by_step"].keys()), [0, 30, 60])
                self.assertGreater(len(res["train_rows"]), 0)
                self.assertEqual(res["train_rows"][-1]["env_steps"], 60)

                t6_path = out_dir / "table6.csv"
                write_table6_csv(t6_path, ["stand"], {"stand": res["iqm_by_step"]})
                self.assertTrue(t6_path.exists())

                # Saved checkpoints load cleanly via load_latent_checkpoint.
                for tag in ("final", "best"):
                    ckpt = out_dir / "models" / f"stand_{tag}.pt"
                    self.assertTrue(ckpt.exists())
                    loaded, meta = load_latent_checkpoint(ckpt)
                    self.assertIsNotNone(loaded)
                    self.assertEqual(meta["algo"], algo)
                    self.assertEqual(meta["z"].shape, (12,))
                    torch.testing.assert_close(
                        meta["z"].norm(), torch.tensor(math.sqrt(12)), atol=1e-4, rtol=0
                    )

                # Frozen FB checkpoint weights must never be mutated by any updater.
                for k, v in self.agent.state_dict().items():
                    torch.testing.assert_close(v, fb_snap[k])

    def test_reinforce_legacy_iter_mode_backward_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_anfis_cfg()
            cfg["algo"] = "reinforce"
            cfg["param"] = "anfis"
            cfg["budget"]["max_env_steps"] = None
            cfg["reinforce"]["iters"] = 2
            cfg["reinforce"]["pop_size"] = 2
            cfg["reinforce"]["episode_steps"] = 10
            cfg["eval"]["eval_every"] = 1
            res = train_task(
                task="stand",
                task_index=0,
                domain="walker",
                agent=self.agent,
                features=self.feats,
                backward_repr=self.B,
                target_rewards=self.rewards,
                reward_fn=self.reward_fn,
                cfg=cfg,
                dynamics=None,
                out_dir=Path(tmp),
                device=torch.device("cpu"),
            )
            self.assertEqual(sorted(res["iqm_by_step"].keys()), [0, 1, 2])
            self.assertEqual([r["step"] for r in res["train_rows"]], [1, 2])
            self.assertEqual([r["env_steps"] for r in res["train_rows"]], [20, 40])


if __name__ == "__main__":
    unittest.main()
