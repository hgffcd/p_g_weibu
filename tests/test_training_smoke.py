import unittest

from algorithms import MRARLECTrainer
from utils.config import load_config


class TestTrainingSmoke(unittest.TestCase):
    def test_short_training_loop(self):
        config = load_config("config.yaml")
        config["training"]["episodes"] = 2
        config["environment"]["max_steps"] = 5
        trainer = MRARLECTrainer(config)
        history = trainer.train()
        self.assertEqual(len(history), 2)
        self.assertIn("reward", history[0])
        self.assertLessEqual(history[0]["steps"], 5)


if __name__ == "__main__":
    unittest.main()
