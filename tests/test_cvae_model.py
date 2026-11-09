"""CVAE ensemble, ICP threshold and scoring of zero_shot_rl.latent.cvae_model and cvae_score."""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from zero_shot_rl.latent.cvae_model import CVAEEnsemble, fit, icp_threshold, is_ood, load_cvae_ensemble
from zero_shot_rl.latent.cvae_score import score_file
from zero_shot_rl.latent.dynamics_model import flatten, split_episodes

A = np.array([[0.1, 0.0, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.3]])
B = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])


def episodes(n, gain=1.0, seed=0):
    """Linear dynamics s' = s + A s + gain B a + noise, [n, 50, dim] arrays."""
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(n, 50, 3))
    a = rng.uniform(-1, 1, size=(n, 50, 2))
    s2 = s + s @ A.T + gain * a @ B.T + 0.01 * rng.normal(size=s.shape)
    return {"s": s, "a": a, "s2": s2, "episode": np.arange(n)}


def args(**kw):
    base = dict(lr=3e-3, weight_decay=0.0, epochs=60, patience=10, batch_size=128, log_every=1000)
    return Namespace(**{**base, **kw})


def trained(target="next_state", members=2):
    torch.manual_seed(0)
    files = [episodes(20)]
    masks = split_episodes([20], 4, seed=0)
    train, val = flatten(files, masks, False), flatten(files, masks, True)
    model = CVAEEnsemble(3, 2, members=members, latent=2, hidden=64, layers=2, target=target)
    model, best = fit(model, train, val, args(), torch.device("cpu"))
    return model, val, best


class CVAETest(unittest.TestCase):
    def test_shifted_dynamics_have_larger_error(self):
        model, val, best = trained()
        self.assertEqual(len(best), 2)
        test, shifted = episodes(5, seed=7), episodes(5, gain=2.0, seed=7)
        a = torch.as_tensor(test["a"], dtype=torch.float32).reshape(-1, 2)
        s, s2, s2_shift = (x.reshape(-1, 3) for x in (test["s"], test["s2"], shifted["s2"]))
        r0 = (model.error(s, a, s2) ** 2).mean(-1)
        r1 = (model.error(s, a, s2_shift) ** 2).mean(-1)
        self.assertEqual(r0.shape, (2, 250))
        self.assertGreater(r1.mean().item(), 10 * r0.mean().item())

    def test_members_differ(self):
        model, val, _ = trained()
        a = torch.as_tensor(val["a"], dtype=torch.float32)
        e = model.error(val["s"], a, val["s2"])
        self.assertGreater((e[0] - e[1]).abs().max().item(), 0)

    def test_delta_target_trains(self):
        model, val, _ = trained("delta", members=1)
        a = torch.as_tensor(val["a"], dtype=torch.float32)
        self.assertLess(model.error(val["s"], a, val["s2"]).norm(dim=-1).median().item(), 0.5)

    def test_fit_info_records_epoch_and_kl_per_dim(self):
        model, _, _ = trained(members=2)
        self.assertEqual(len(model.fit_info), 2)
        self.assertEqual(model.fit_info[0]["val_kl_per_dim"].shape, (2,))
        self.assertGreaterEqual(model.fit_info[0]["best_epoch"], 1)

    def test_error_is_reconstruction_residual(self):
        model, val, _ = trained("next_state", members=1)
        a = torch.as_tensor(val["a"], dtype=torch.float32)
        s32 = torch.as_tensor(val["s"], dtype=torch.float32)
        f = model.displacement(val["s"], a, val["s2"])
        np.testing.assert_allclose(model.error(val["s"], a, val["s2"]).numpy(),
                                   (val["s2"] - val["s"]) - f.numpy(), atol=1e-4)
        self.assertEqual(f.shape, (1, len(s32), 3))

    def test_fixed_statistics_are_used(self):
        torch.manual_seed(0)
        files = [episodes(10)]
        masks = split_episodes([10], 2, seed=0)
        train, val = flatten(files, masks, False), flatten(files, masks, True)
        model = CVAEEnsemble(3, 2, members=1, latent=2, hidden=8, layers=1)
        stats = ((np.arange(5.0), np.arange(1.0, 6.0)), (np.arange(3.0), np.arange(1.0, 4.0)))
        fit(model, train, val, args(epochs=1), torch.device("cpu"), stats)
        for buffer, expected in ((model.x_mean, stats[0][0]), (model.x_std, stats[0][1]),
                                 (model.y_mean, stats[1][0]), (model.y_std, stats[1][1])):
            np.testing.assert_allclose(buffer.numpy(), expected)

    def test_icp_threshold_and_rule(self):
        calibration = np.arange(1.0, 101.0).reshape(2, 50)
        self.assertEqual(icp_threshold(calibration, 0.95), 95.0)
        self.assertEqual(icp_threshold(calibration, 0.0), 1.0)
        self.assertEqual(icp_threshold(calibration, 1.0), 100.0)
        errors = np.array([[1.0, 96.0, 99.0], [50.0, 97.0, 10.0]])   # [N=2, 3 transitions]
        np.testing.assert_array_equal(is_ood(errors, 95.0), [False, True, False])

    def test_score_file_shapes_and_baseline(self):
        model, _, _ = trained()
        test = episodes(3, seed=9)
        paired = {"observations": test["s"], "actions": test["a"], "next_rollout": test["s2"],
                  "next_shift": episodes(3, gain=2.0, seed=9)["s2"], "episode": np.arange(3)}
        out = score_file(model, paired)
        self.assertEqual(out["member_rollout"].shape, (3, 50, 2))
        self.assertEqual(out["mean_shift"].shape, (3, 50))
        np.testing.assert_allclose(out["min_rollout"], out["member_rollout"].min(-1))
        np.testing.assert_allclose(out["mean_rollout"], out["member_rollout"].mean(-1))
        self.assertEqual(out["mse_shift"].shape, (3,))
        self.assertEqual(float(out["gap_rollout"].max()), 0.0)
        self.assertGreater(np.median(out["mean_shift"]), np.median(out["mean_rollout"]))

    def test_unknown_target_is_rejected(self):
        with self.assertRaises(ValueError):
            CVAEEnsemble(3, 2, target="state")

    def test_save_load_roundtrip(self):
        torch.manual_seed(0)
        model = CVAEEnsemble(3, 2, members=2, latent=2, hidden=8, layers=1).eval()
        s, a = torch.randn(4, 3), torch.randn(4, 2)
        s2 = s + 0.1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.pt"
            model.save(path, note="x")
            loaded, meta = load_cvae_ensemble(path)
        self.assertEqual(meta["note"], "x")
        torch.testing.assert_close(loaded.error(s, a, s2), model.error(s, a, s2))


if __name__ == "__main__":
    unittest.main()
