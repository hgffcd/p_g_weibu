import unittest

from agents import PlanarEncirclementController
from envs import EncirclementEnv
from utils.config import load_config


class TestPlanarController(unittest.TestCase):
    def test_controller_action_shape(self):
        config = load_config("config.yaml")
        env = EncirclementEnv(config)
        env.reset(seed=1)
        controller = PlanarEncirclementController(config)
        actions = controller.act(env)
        self.assertEqual(actions.shape, (env.num_pursuers, 2))


if __name__ == "__main__":
    unittest.main()
