import unittest

import numpy as np

from envs import EncirclementEnv
from utils.config import load_config


class TestEncirclementEnv(unittest.TestCase):
    def setUp(self):
        self.config = load_config("config.yaml")
        self.env = EncirclementEnv(self.config)

    def test_reset_shapes(self):
        obs, state = self.env.reset(seed=1)
        self.assertEqual(obs.shape, (self.env.num_pursuers, self.env.obs_dim))
        self.assertEqual(state["pursuer_pos"].shape, (self.env.num_pursuers, 2))
        self.assertTrue(np.allclose(state["pursuer_vel"], 0.0))

    def test_step_shapes(self):
        self.env.reset(seed=1)
        actions = np.zeros((self.env.num_pursuers, 2), dtype=float)
        obs, rewards, done, info = self.env.step(actions)
        self.assertEqual(obs.shape, (self.env.num_pursuers, self.env.obs_dim))
        self.assertEqual(rewards.shape, (self.env.num_pursuers,))
        self.assertIsInstance(done, bool)
        self.assertIn("success", info)


if __name__ == "__main__":
    unittest.main()
