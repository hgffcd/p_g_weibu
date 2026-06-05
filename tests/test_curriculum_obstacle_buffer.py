import unittest

from algorithms.regulators import obstacle_buffer, obstacle_radius
from utils.config import load_config


class TestCurriculumObstacleBuffer(unittest.TestCase):
    def test_buffer_is_zero_before_obstacle_curriculum_starts(self):
        config = load_config("config_vector_server.yaml")
        radius = obstacle_radius(1, 2000, 0.18, config)
        buffer = obstacle_buffer(1, 2000, 0.18, 0.05, config)
        self.assertEqual(radius, 0.0)
        self.assertEqual(buffer, 0.0)

    def test_buffer_reaches_original_value_after_curriculum(self):
        config = load_config("config_vector_server.yaml")
        radius = obstacle_radius(2000, 2000, 0.18, config)
        buffer = obstacle_buffer(2000, 2000, 0.18, 0.05, config)
        self.assertAlmostEqual(radius, 0.18)
        self.assertAlmostEqual(buffer, 0.05)


if __name__ == "__main__":
    unittest.main()
