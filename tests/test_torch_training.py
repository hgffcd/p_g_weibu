import unittest

import torch

from agents.torch_networks import RecurrentActor, RecurrentCritic
from algorithms.mra_rlec import TorchMRARLECTrainer
from envs import EncirclementEnv
from utils.config import load_config


class TestTorchTraining(unittest.TestCase):
    def test_recurrent_actor_critic_forward(self):
        config = load_config("config.yaml")
        env = EncirclementEnv(config)
        obs, _ = env.reset(seed=1)
        flat_state = env.flat_state()
        actor = RecurrentActor(obs.shape[-1], 2, config["environment"]["max_acceleration"])
        critic = RecurrentCritic(flat_state.shape[0])

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        action, log_prob, hidden = actor.act(obs_tensor)
        self.assertEqual(action.shape, (env.num_pursuers, 2))
        self.assertEqual(log_prob.shape, (env.num_pursuers,))
        self.assertEqual(hidden.shape[1], env.num_pursuers)

        value, critic_hidden = critic(torch.as_tensor(flat_state[None, :], dtype=torch.float32))
        self.assertEqual(value.shape, (1,))
        self.assertEqual(critic_hidden.shape[1], 1)

    def test_torch_training_smoke(self):
        config = load_config("config.yaml")
        config["training"]["episodes"] = 1
        config["training"]["ppo_epochs"] = 1
        config["training"]["save_checkpoint"] = False
        config["environment"]["max_steps"] = 4
        config["training"]["device"] = "cpu"
        trainer = TorchMRARLECTrainer(config)
        history = trainer.train()
        self.assertEqual(len(history), 1)
        self.assertIn("actor_loss", history[0])
        self.assertIn("critic_loss", history[0])
        self.assertIn("bc_loss", history[0])


if __name__ == "__main__":
    unittest.main()
