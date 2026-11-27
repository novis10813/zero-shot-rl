import json
import subprocess
import unittest

import numpy as np
import torch

from analyze import interactive_explorer as ie
from zero_shot_rl.latent.anfis import ANFIS


class InteractiveExplorerTest(unittest.TestCase):
    def _make_model(self, sensors):
        torch.manual_seed(7)
        model = ANFIS(
            sensors=sensors,
            out_dim=1,
            consequent="first_order",
            t_norm="product",
            normalize_firing=True,
            output_activation="identity",
            normalize_inputs=True,
        )
        rng = torch.Generator().manual_seed(11)
        x_init = torch.randn(500, len(sensors), generator=rng) * 2.0 + 0.5
        model.init_from_data(x_init)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn(p.shape, generator=rng) * 0.2)
        return model.eval()

    def test_js_anfis_forward_matches_pytorch(self):
        sensors_4 = [
            {"name": "torso_height", "num_terms": 3, "membership": "gaussian"},
            {"name": "torso_upright", "num_terms": 2, "membership": "gaussian"},
            {"name": "horizontal_velocity", "num_terms": 3, "membership": "gaussian"},
            {"name": "angmomentum", "num_terms": 2, "membership": "gaussian"},
        ]
        sensors_8 = [
            {"name": f"s{i}", "num_terms": 2, "membership": "gaussian"} for i in range(8)
        ]
        for sensors in (sensors_4, sensors_8):
            model = self._make_model(sensors)
            spec = ie.export_condition_spec(model)
            weights = ie.export_model_weights(model)
            rng = np.random.default_rng(123)
            X = rng.normal(0.3, 1.8, size=(32, len(sensors))).astype(np.float32)
            with torch.no_grad():
                expected = model(torch.from_numpy(X)).squeeze(-1).numpy()

            script = (
                ie.JS_CORE_MATH
                + "\nconst input = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n"
                + "const decoded = decodeSeedWeights(input.weights, input.spec);\n"
                + "const out = input.X.map(row => evalAnfisSingle(row, decoded, input.spec));\n"
                + "const base = input.X[0];\n"
                + "const xs = [-1.5, 0.0, 0.7, 2.1];\n"
                + "const ys = [-0.8, 0.3, 1.4];\n"
                + "const s1d = Array.from(evalAnfis1DSweep(base, 1, xs, decoded, input.spec));\n"
                + "const s1dRef = xs.map(v => { const r = base.slice(); r[1] = v; return evalAnfisSingle(r, decoded, input.spec); });\n"
                + "const g2d = Array.from(evalAnfis2DGrid(base, 0, 2, xs, ys, decoded, input.spec));\n"
                + "const g2dRef = [];\n"
                + "for (let j = 0; j < ys.length; j++) for (let i = 0; i < xs.length; i++) {\n"
                + "  const r = base.slice(); r[0] = xs[i]; r[2] = ys[j]; g2dRef.push(evalAnfisSingle(r, decoded, input.spec));\n"
                + "}\n"
                + "process.stdout.write(JSON.stringify({out, s1d, s1dRef, g2d, g2dRef}));\n"
            )
            proc = subprocess.run(
                ["node", "-e", script],
                input=json.dumps({"spec": spec, "weights": weights, "X": X.tolist()}),
                text=True,
                capture_output=True,
                check=True,
            )
            res = json.loads(proc.stdout)
            got = np.asarray(res["out"], dtype=np.float32)
            max_err = float(np.max(np.abs(got - expected)))
            self.assertLess(max_err, 1e-5, f"max_err={max_err} for D={len(sensors)}")
            self.assertLess(float(np.max(np.abs(np.asarray(res["s1d"]) - np.asarray(res["s1dRef"])))), 1e-12)
            self.assertLess(float(np.max(np.abs(np.asarray(res["g2d"]) - np.asarray(res["g2dRef"])))), 1e-12)

    def test_js_true_reward_matches_dmc(self):
        from dm_control.utils import rewards as dm_rewards

        def true_reward_ref(task, x):
            h, up, vx, ang = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
            standing = dm_rewards.tolerance(h, bounds=(1.2, float("inf")), margin=0.6)
            upright = (1.0 + up) / 2.0
            sr = (3.0 * standing + upright) / 4.0
            if task == "stand":
                return sr
            if task == "walk":
                mv = dm_rewards.tolerance(
                    vx, bounds=(1.0, float("inf")), margin=0.5, value_at_margin=0.5, sigmoid="linear"
                )
            elif task == "run":
                mv = dm_rewards.tolerance(
                    vx, bounds=(8.0, float("inf")), margin=4.0, value_at_margin=0.5, sigmoid="linear"
                )
            elif task == "flip":
                mv = dm_rewards.tolerance(
                    ang, bounds=(5.0, float("inf")), margin=5.0, value_at_margin=0.0, sigmoid="linear"
                )
            else:
                raise ValueError(task)
            return sr * (5.0 * mv + 1.0) / 6.0

        rng = np.random.default_rng(42)
        X = np.stack(
            [
                rng.uniform(0.05, 1.6, size=40),
                rng.uniform(-1.0, 1.0, size=40),
                rng.uniform(-3.0, 10.0, size=40),
                rng.uniform(-12.0, 18.0, size=40),
            ],
            axis=1,
        )
        script = (
            ie.JS_CORE_MATH
            + "\nconst input = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n"
            + "const res = {};\n"
            + "for (const t of input.tasks) {\n"
            + "  res[t] = input.X.map(r => trueReward(t, r[0], r[1], r[2], r[3]));\n"
            + "}\n"
            + "process.stdout.write(JSON.stringify(res));\n"
        )
        tasks = ["stand", "walk", "run", "flip"]
        proc = subprocess.run(
            ["node", "-e", script],
            input=json.dumps({"tasks": tasks, "X": X.tolist()}),
            text=True,
            capture_output=True,
            check=True,
        )
        got_all = json.loads(proc.stdout)
        for t in tasks:
            exp = true_reward_ref(t, X)
            got = np.asarray(got_all[t], dtype=np.float64)
            self.assertLess(float(np.max(np.abs(got - exp))), 1e-6, f"mismatch on {t}")

    def test_build_html_contains_core_components(self):
        payload = {"conditions": {}, "ref_samples_b64": "", "ref_sensor_names": [], "ref_n": 0}
        html = ie.build_html(payload)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("evalAnfisSingle", html)
        self.assertIn("trueReward", html)
        self.assertIn("__ANFIS_DATA__", html)


if __name__ == "__main__":
    unittest.main()
