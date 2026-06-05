import unittest

import numpy as np

from agents.apf_guide import APFGuidePolicy
from algorithms.regulators import guide_steps
from envs import EncirclementEnv
from utils.config import load_config


class TestEnvironmentPyPriority(unittest.TestCase):
    def test_environment_py_guide_action_shape_and_limit(self):
        config = load_config("config.yaml")
        config["guide_policy"]["mode"] = "environment_py"
        env = EncirclementEnv(config)
        env.reset(seed=1)
        guide = APFGuidePolicy(config)
        actions = guide.act(env)
        self.assertEqual(actions.shape, (env.num_pursuers, 2))
        self.assertTrue(np.all(np.abs(actions) <= config["environment"]["max_acceleration"] + 1e-6))
        _, _, _, info = env.step(actions)
        self.assertIn("collision_type", info)

    def test_environment_py_jsrl_regulator(self):
        config = load_config("config.yaml")
        config["regulators"]["jsrl_mode"] = "environment_py"
        config["regulators"]["cg"] = 1.0
        config["regulators"]["rho0"] = 0.8
        steps = guide_steps(current_episode=50, total_episodes=100, max_steps=200, config=config)
        self.assertEqual(steps, 80)

    def test_environment_py_formation_correction_is_bounded(self):
        config = load_config("config.yaml")
        config["guide_policy"]["mode"] = "environment_py"
        config["guide_policy"]["formation_correction"] = True
        config["guide_policy"]["formation_gain"] = 2.5
        config["guide_policy"]["formation_damping"] = 1.2
        config["guide_policy"]["formation_weight"] = 1.0
        config["guide_policy"]["formation_start_step"] = 0
        config["guide_policy"]["formation_ramp_steps"] = 1
        config["guide_policy"]["formation_activate_distance_error"] = 2.0
        config["guide_policy"]["formation_obstacle_gate_margin"] = 1.0
        config["guide_policy"]["formation_candidate_count"] = 5
        env = EncirclementEnv(config)
        env.reset(seed=1)
        guide = APFGuidePolicy(config)
        actions = guide.act(env)
        cached_actions = guide.act(env)
        self.assertEqual(actions.shape, (env.num_pursuers, 2))
        self.assertTrue(np.all(np.isfinite(actions)))
        self.assertTrue(np.all(np.abs(actions) <= config["environment"]["max_acceleration"] + 1e-6))
        self.assertTrue(np.allclose(actions, cached_actions))


if __name__ == "__main__":
    unittest.main()
