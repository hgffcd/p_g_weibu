import math
import unittest

import numpy as np

from envs import EncirclementEnv
from utils.config import load_config


class TestReward(unittest.TestCase):
    def setUp(self):
        self.config = load_config("config.yaml")
        self.env = EncirclementEnv(self.config)
        self.env.reset(seed=1)

    def test_global_reward_when_all_finished(self):
        n = self.env.num_pursuers
        angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        self.env.target_pos = np.array([0.0, 0.0], dtype=float)
        self.env.pursuer_pos = np.stack([
            self.env.capture_distance * np.cos(angles),
            self.env.capture_distance * np.sin(angles),
        ], axis=1)
        self.env.obstacles = []
        finished = self.env.check_finished_each_agent()
        rewards = self.env.compute_rewards(collision=False, finished=finished)
        self.assertTrue(np.all(finished))
        self.assertTrue(np.allclose(rewards, self.env.global_reward))

    def test_collision_penalty_lowers_step_reward(self):
        self.env.pursuer_pos[0] = self.env.pursuer_pos[1]
        reward_collision = self.env.compute_rewards(collision=True, finished=np.zeros(self.env.num_pursuers, dtype=bool))
        reward_clear = self.env.compute_rewards(collision=False, finished=np.zeros(self.env.num_pursuers, dtype=bool))
        self.assertLess(float(np.mean(reward_collision)), float(np.mean(reward_clear)))


if __name__ == "__main__":
    unittest.main()
