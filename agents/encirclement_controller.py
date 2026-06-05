"""Deterministic planar encirclement controller.

This is not claimed as the paper's learned policy. It is a compact planar-circle
baseline used to validate the encirclement environment and Section V metrics
without ROS2/Gazebo vehicle simulation.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from envs.encirclement_env import EncirclementEnv
from utils.geometry import clip_by_component, norm, safe_unit


class PlanarEncirclementController:
    """PD controller toward uniformly spaced points on the capture circle."""

    def __init__(self, config: Dict[str, Any]):
        env_cfg = config["environment"]
        guide_cfg = config["guide_policy"]
        self.capture_distance = float(env_cfg["capture_distance"])
        self.max_acceleration = float(env_cfg["max_acceleration"])
        self.kp = float(config.get("controller", {}).get("kp", 1.8))
        self.kd = float(config.get("controller", {}).get("kd", 1.2))
        self.obstacle_gain = float(config.get("controller", {}).get("obstacle_gain", guide_cfg.get("ko", 4.0)))
        self.obstacle_threshold = float(config.get("controller", {}).get("obstacle_threshold", 0.75))
        self.pursuer_gain = float(config.get("controller", {}).get("pursuer_gain", 2.5))
        self.pursuer_threshold = float(config.get("controller", {}).get("pursuer_threshold", 0.65))

    def act(self, env: EncirclementEnv) -> np.ndarray:
        desired = self._desired_points(env)
        actions = []
        for i in range(env.num_pursuers):
            acc = self.kp * (desired[i] - env.pursuer_pos[i]) - self.kd * env.pursuer_vel[i]
            acc += self._obstacle_repulsion(env, i)
            acc += self._pursuer_repulsion(env, i)
            actions.append(acc)
        return clip_by_component(np.vstack(actions), self.max_acceleration)

    def _desired_points(self, env: EncirclementEnv) -> np.ndarray:
        rel = env.pursuer_pos - env.target_pos[None, :]
        current_angles = np.arctan2(rel[:, 1], rel[:, 0])
        order = np.argsort(current_angles)
        base_angle = -math.pi / 2.0

        desired = np.zeros_like(env.pursuer_pos)
        for rank, pursuer_idx in enumerate(order):
            angle = base_angle + 2.0 * math.pi * rank / env.num_pursuers
            desired[pursuer_idx] = env.target_pos + self.capture_distance * np.array([math.cos(angle), math.sin(angle)])
        return desired

    def _obstacle_repulsion(self, env: EncirclementEnv, i: int) -> np.ndarray:
        repulsion = np.zeros(2, dtype=float)
        for obstacle in env.obstacles:
            diff = env.pursuer_pos[i] - obstacle.pos
            clearance = max(norm(diff) - obstacle.radius - env.pursuer_radius, 1e-8)
            if clearance < self.obstacle_threshold:
                repulsion += self.obstacle_gain * (self.obstacle_threshold - clearance) / clearance * safe_unit(diff)
        return repulsion

    def _pursuer_repulsion(self, env: EncirclementEnv, i: int) -> np.ndarray:
        repulsion = np.zeros(2, dtype=float)
        for j in range(env.num_pursuers):
            if i == j:
                continue
            diff = env.pursuer_pos[i] - env.pursuer_pos[j]
            clearance = max(norm(diff) - 2.0 * env.pursuer_radius, 1e-8)
            if clearance < self.pursuer_threshold:
                repulsion += self.pursuer_gain * (self.pursuer_threshold - clearance) / clearance * safe_unit(diff)
        return repulsion
