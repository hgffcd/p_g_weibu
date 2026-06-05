import unittest

import numpy as np

from stage2.curriculum import target_speed_for_progress
from stage2.envs import Stage2EncirclementEnv, Stage2VectorizedEncirclementEnv
from utils.config import load_config


class Stage2EnvTest(unittest.TestCase):
    def test_fixed_stage2_reset_matches_shapes(self):
        cfg = load_config("config_stage2_moving.yaml")
        env = Stage2EncirclementEnv(cfg)
        obs, state = env.reset(seed=0)
        self.assertEqual(obs.shape, (cfg["environment"]["num_pursuers"], env.obs_dim))
        self.assertEqual(state["pursuer_pos"].shape, (cfg["environment"]["num_pursuers"], 2))
        self.assertFalse(env.last_reset_info["randomized"])

    def test_randomized_reset_changes_target(self):
        cfg = load_config("config_stage2_randomized.yaml")
        env = Stage2EncirclementEnv(cfg)
        env.reset(seed=1)
        target_a = env.target_pos.copy()
        env.reset(seed=2)
        target_b = env.target_pos.copy()
        self.assertTrue(env.last_reset_info["randomized"])
        self.assertGreater(np.linalg.norm(target_a - target_b), 1e-6)

    def test_vectorized_stage2_shapes(self):
        cfg = load_config("config_stage2_randomized.yaml")
        vec = Stage2VectorizedEncirclementEnv(cfg, num_envs=3)
        obs, states = vec.reset(seed=3)
        self.assertEqual(obs.shape[0], 3)
        self.assertEqual(obs.shape[1], cfg["environment"]["num_pursuers"])
        self.assertEqual(states.shape[0], 3)

    def test_target_speed_curriculum_increases(self):
        cfg = load_config("config_stage2_moving.yaml")
        start = target_speed_for_progress(1, 100, cfg)
        middle = target_speed_for_progress(50, 100, cfg)
        end = target_speed_for_progress(100, 100, cfg)
        self.assertLessEqual(start, middle)
        self.assertLessEqual(middle, end)
        self.assertAlmostEqual(end, cfg["environment"]["target_max_velocity"])


if __name__ == "__main__":
    unittest.main()
