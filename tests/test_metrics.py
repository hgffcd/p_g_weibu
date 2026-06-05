import unittest

from envs import EncirclementEnv
from utils.config import load_config
from utils.metrics import aggregate_episodes, communication_meta_messages, step_metric, summarize_episode


class TestMetrics(unittest.TestCase):
    def test_step_and_summary_metrics(self):
        config = load_config("config.yaml")
        env = EncirclementEnv(config)
        env.reset(seed=1)
        _, reward, _, info = env.step([[0.0, 0.0]] * env.num_pursuers)
        metric = step_metric(env, reward)
        episode = summarize_episode([metric], info, 1, env.dt)
        summary = aggregate_episodes([episode])
        self.assertIn("average_step_reward", summary)
        self.assertIn("success_rate", summary)

    def test_communication_metrics(self):
        metrics = communication_meta_messages(5)
        self.assertLess(metrics["bidirectional_meta_messages"], metrics["fully_connected_meta_messages"])


if __name__ == "__main__":
    unittest.main()
