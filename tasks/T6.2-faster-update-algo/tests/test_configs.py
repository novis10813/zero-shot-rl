"""Validate T6.2-faster-update-algo paper-native LoLA/ReLA configs and per-batch sweep files."""

import unittest
from pathlib import Path

import yaml

from zero_shot_rl.finetune.anfis_z import load_config

REPO = Path(__file__).resolve().parents[3]
TASK_DIR = REPO / "tasks/T6.2-faster-update-algo"


class T62ConfigsTest(unittest.TestCase):
    def test_lola_and_rela_paper_configs(self):
        for fname, expected_algo, expected_param in (
            ("lola_z.yaml", "lola", "z"),
            ("rela_z.yaml", "rela", "z"),
            ("lola_anfis.yaml", "lola", "anfis"),
            ("cem_z.yaml", "cem", "z"),
            ("cem_anfis.yaml", "cem", "anfis"),
            ("cmaes_z.yaml", "cmaes", "z"),
            ("cmaes_anfis.yaml", "cmaes", "anfis"),
        ):
            cfg = load_config(str(TASK_DIR / "configs" / fname))
            self.assertEqual(cfg["algo"], expected_algo)
            self.assertEqual(cfg["param"], expected_param)
            self.assertTrue(cfg["zero_shot_init"])
            self.assertEqual(cfg["budget"]["max_env_steps"], 800000)
            self.assertEqual(cfg["budget"]["eval_every_env_steps"], 40000)
            self.assertEqual(cfg["eval"]["eval_episodes"], 20)
            self.assertEqual(cfg["eval"]["eval_seed_start"], 5)
            self.assertEqual(cfg["eval"]["tasks"], ["stand", "walk", "run", "flip"])

    def test_ood_sweeps_are_bounded_to_10_jobs_each(self):
        sweeps = sorted((TASK_DIR / "configs").glob("sweep_*.yaml"))
        self.assertEqual(len(sweeps), 8)
        for sw_path in sweeps:
            with open(sw_path) as f:
                data = yaml.safe_load(f)
            self.assertEqual(data.get("on", data.get(True)), "idlab")
            self.assertEqual(data["vram"], "24G")
            self.assertEqual(len(data["grid"]["ft_seed"]), 10)
            self.assertEqual(set(data["grid"].keys()), {"ft_seed"})
            self.assertIn("OMP_NUM_THREADS=1", data["command"])
            self.assertIn("MKL_NUM_THREADS=1", data["command"])


if __name__ == "__main__":
    unittest.main()
