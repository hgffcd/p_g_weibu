import unittest

from agents import LinearGaussianActor, LinearValueCritic
from envs import EncirclementEnv
from utils.config import load_config


class TestModelForward(unittest.TestCase):
    def test_actor_and_critic_forward(self):
        config = load_config("config.yaml")
        env = EncirclementEnv(config)
        obs, _ = env.reset(seed=1)
        actor = LinearGaussianActor(env.obs_dim, 2, config["environment"]["max_acceleration"], seed=3)
        actions = actor.act(obs)
        self.assertEqual(actions.shape, (env.num_pursuers, 2))

        flat_state = env.flat_state()
        critic = LinearValueCritic(flat_state.shape[0], seed=4)
        value = critic.value(flat_state)
        self.assertIsInstance(value, float)


if __name__ == "__main__":
    unittest.main()
