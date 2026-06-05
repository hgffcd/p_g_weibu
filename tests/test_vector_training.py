import unittest

from algorithms.vector_mappo import VectorMAPPOTrainer
from envs.vector_env import VectorizedEncirclementEnv
from utils.config import load_config


class TestVectorTraining(unittest.TestCase):
    def test_vector_env_shapes(self):
        config = load_config("config.yaml")
        envs = VectorizedEncirclementEnv(config, num_envs=2)
        obs, states = envs.reset(seed=1)
        self.assertEqual(obs.shape[0], 2)
        self.assertEqual(obs.shape[1], config["environment"]["num_pursuers"])
        self.assertEqual(states.shape[0], 2)

    def test_vector_trainer_smoke(self):
        config = load_config("config_vector_server.yaml")
        config["training"]["device"] = "cpu"
        config["training"]["save_checkpoint"] = False
        config["training"]["log_dir"] = "logs_vector_test"
        config["training"]["policy_mode"] = "residual"
        config["training"]["residual_scale"] = 0.25
        config["vector_training"]["num_envs"] = 2
        config["vector_training"]["rollout_length"] = 4
        config["vector_training"]["updates"] = 1
        config["vector_training"]["curriculum_progress"] = "updates"
        config["vector_training"]["eval_interval"] = 0
        trainer = VectorMAPPOTrainer(config)
        history = trainer.train()
        self.assertEqual(len(history), 1)
        self.assertIn("success_rate", history[0])
        self.assertEqual(history[0]["schedule_step"], 1)
        self.assertLessEqual(history[0]["guide_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
