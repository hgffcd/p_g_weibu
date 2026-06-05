"""Minimal EMOCA environment based on the paper's Section II and IV-C."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from utils.geometry import (
    bearing_angle,
    clip_by_component,
    flatten_state,
    included_angle,
    neighbor_indices,
    norm,
    safe_unit,
)


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float
    buffer: float

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


class EncirclementEnv:
    """2-D encirclement environment.

    Dynamics follow Section II-A, Eq. (1). Rewards follow GERD in Section IV-C.1,
    Eqs. (15)-(18). Observations follow Section IV-C.2, Eq. (19).
    """

    obs_dim = 24

    def __init__(self, config: Dict[str, Any]):
        env_cfg = config["environment"]
        reward_cfg = config["reward"]
        self.config = config
        self.num_pursuers = int(env_cfg["num_pursuers"])
        self.max_steps = int(env_cfg["max_steps"])
        self.dt = float(env_cfg["dt"])
        self.capture_distance = float(env_cfg["capture_distance"])
        self.delta_alpha = float(env_cfg["delta_alpha"])
        self.delta_distance = float(env_cfg["delta_distance"])
        self.pursuer_radius = float(env_cfg["pursuer_radius"])
        self.target_radius = float(env_cfg["target_radius"])
        self.safety_buffer = float(env_cfg["safety_buffer"])
        self.safety_buffer_scale = float(env_cfg["safety_buffer_scale"])
        self.max_velocity = float(env_cfg["max_velocity"])
        self.target_max_velocity = float(env_cfg["target_max_velocity"])
        self.max_acceleration = float(env_cfg["max_acceleration"])
        self.target_policy = str(env_cfg.get("target_policy", "static"))
        self.initial_spacing = float(env_cfg.get("initial_spacing", 0.6))
        self.initial_y = float(env_cfg.get("initial_y", 0.0))
        self.target_initial_position = np.array(env_cfg["target_initial_position"], dtype=float)
        self.base_obstacles = [Obstacle(*map(float, item)) for item in env_cfg.get("obstacles", [])]

        self.k1 = float(reward_cfg["k1"])
        self.k2 = float(reward_cfg["k2"])
        self.w1 = float(reward_cfg["w1"])
        self.w2 = float(reward_cfg["w2"])
        self.w3 = float(reward_cfg["w3"])
        self.collision_penalty = float(reward_cfg["collision_penalty"])
        self.local_reward = float(reward_cfg["local_reward"])
        self.global_reward = float(reward_cfg["global_reward"])
        self.angle_shaping_weight = float(reward_cfg.get("angle_shaping_weight", 0.0))
        self.distance_shaping_weight = float(reward_cfg.get("distance_shaping_weight", 0.0))
        self.timeout_penalty = float(reward_cfg.get("timeout_penalty", 0.0))

        self.pursuer_pos = np.zeros((self.num_pursuers, 2), dtype=float)
        self.pursuer_vel = np.zeros((self.num_pursuers, 2), dtype=float)
        self.target_pos = np.zeros(2, dtype=float)
        self.target_vel = np.zeros(2, dtype=float)
        self.obstacles = list(self.base_obstacles)
        self.finished = np.zeros(self.num_pursuers, dtype=bool)
        self.step_count = 0

    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset using the initialization described in Section II-A.

        Exact coordinates are an engineering assumption documented in assumptions.md.
        """
        if seed is not None:
            np.random.default_rng(seed)
        offsets = (np.arange(self.num_pursuers) - (self.num_pursuers - 1) / 2.0) * self.initial_spacing
        self.pursuer_pos = np.stack([offsets, np.full_like(offsets, self.initial_y)], axis=1).astype(float)
        self.pursuer_vel = np.zeros((self.num_pursuers, 2), dtype=float)
        self.target_pos = self.target_initial_position.astype(float).copy()
        self.target_vel = np.zeros(2, dtype=float)
        self.obstacles = [Obstacle(o.x, o.y, o.radius, o.buffer) for o in self.base_obstacles]
        self.finished = np.zeros(self.num_pursuers, dtype=bool)
        self.step_count = 0
        return self.compute_observations(), self.get_state()

    def get_state(self) -> Dict[str, Any]:
        return {
            "pursuer_pos": self.pursuer_pos.copy(),
            "pursuer_vel": self.pursuer_vel.copy(),
            "target_pos": self.target_pos.copy(),
            "target_vel": self.target_vel.copy(),
            "obstacles": np.array([[o.x, o.y, o.radius, o.buffer] for o in self.obstacles], dtype=float),
            "finished": self.finished.copy(),
            "step_count": self.step_count,
        }

    def flat_state(self) -> np.ndarray:
        state = self.get_state()
        return flatten_state([
            state["pursuer_pos"],
            state["pursuer_vel"],
            state["target_pos"],
            state["target_vel"],
            state["obstacles"],
            state["finished"].astype(float),
            np.array([state["step_count"]], dtype=float),
        ])

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        """Apply Eq. (1) with component-wise velocity and acceleration limits."""
        actions = np.asarray(actions, dtype=float).reshape(self.num_pursuers, 2)
        actions = clip_by_component(actions, self.max_acceleration)
        self.pursuer_vel = clip_by_component(self.pursuer_vel + actions * self.dt, self.max_velocity)
        self.pursuer_pos = self.pursuer_pos + self.pursuer_vel * self.dt

        target_acc = self._target_action()
        self.target_vel = clip_by_component(self.target_vel + target_acc * self.dt, self.target_max_velocity)
        self.target_pos = self.target_pos + self.target_vel * self.dt

        self.finished = self.check_finished_each_agent()
        collision_detail = self.collision_detail()
        collision = collision_detail["collision"]
        self.step_count += 1

        success = bool(np.all(self.finished))
        timeout = self.step_count >= self.max_steps
        done = success or collision or timeout
        rewards = self.compute_rewards(collision=collision, finished=self.finished)
        if timeout and not success and not collision:
            rewards = rewards + self.timeout_penalty
        distance_error, angle_error = self.encirclement_errors()
        info = {
            "success": success,
            "collision": collision,
            "timeout": timeout,
            "finished": self.finished.copy(),
            "collision_type": collision_detail["type"],
            "collision_distance": collision_detail["distance"],
            "collision_threshold": collision_detail["threshold"],
            "distance_error": distance_error,
            "angle_error": angle_error,
        }
        return self.compute_observations(), rewards, done, info

    def _target_action(self) -> np.ndarray:
        """Target acceleration from configured escape policy.

        PFP and GP are velocity control laws in Section V-D, Eqs. (20)-(21).
        This environment converts the desired velocity into acceleration for the
        second-order target dynamics described in Section II-A.
        """
        if self.target_policy == "pfp":
            desired_velocity = self._target_velocity_pfp()
            return clip_by_component((desired_velocity - self.target_vel) / max(self.dt, 1e-8), self.max_acceleration)
        if self.target_policy == "greedy":
            desired_velocity = self._target_velocity_greedy()
            return clip_by_component((desired_velocity - self.target_vel) / max(self.dt, 1e-8), self.max_acceleration)
        return np.zeros(2, dtype=float)

    def _target_velocity_greedy(self) -> np.ndarray:
        """Greedy policy from Section V-D, Eq. (21)."""
        distances = np.linalg.norm(self.target_pos[None, :] - self.pursuer_pos, axis=1)
        nearest = int(np.argmin(distances))
        return safe_unit(self.target_pos - self.pursuer_pos[nearest]) * self.target_max_velocity

    def _target_velocity_pfp(self) -> np.ndarray:
        """Potential field policy from Section V-D, Eq. (20)."""
        direction = np.zeros(2, dtype=float)
        for pos in self.pursuer_pos:
            rel = self.target_pos - pos
            distance = max(norm(rel), 1e-8)
            d_hat = distance - (self.target_radius + self.pursuer_radius)
            direction += (d_hat / distance) * rel
        return safe_unit(direction) * self.target_max_velocity

    def check_finished_each_agent(self) -> np.ndarray:
        """Check Definition 2 and Eq. (2) for every pursuer."""
        neighbors = neighbor_indices(self.pursuer_pos, self.target_pos)
        alpha_exp = 2.0 * math.pi / self.num_pursuers
        finished = np.zeros(self.num_pursuers, dtype=bool)
        for i in range(self.num_pursuers):
            left, right = neighbors[i]
            dist_ok = abs(norm(self.pursuer_pos[i] - self.target_pos) - self.capture_distance) < self.delta_distance
            left_ok = abs(included_angle(self.pursuer_pos[i], self.pursuer_pos[left], self.target_pos) - alpha_exp) < self.delta_alpha
            right_ok = abs(included_angle(self.pursuer_pos[i], self.pursuer_pos[right], self.target_pos) - alpha_exp) < self.delta_alpha
            finished[i] = dist_ok and left_ok and right_ok
        return finished

    def check_collision(self) -> bool:
        """Safety-buffer collision condition from Section II-A."""
        return bool(self.collision_detail()["collision"])

    def collision_detail(self) -> Dict[str, Any]:
        """Return the first safety-buffer collision and its source."""
        threshold_pp = 2 * self.pursuer_radius + self.safety_buffer_scale * (2 * self.safety_buffer)
        for i in range(self.num_pursuers):
            for j in range(i + 1, self.num_pursuers):
                distance = norm(self.pursuer_pos[i] - self.pursuer_pos[j])
                if distance < threshold_pp:
                    return {
                        "collision": True,
                        "type": "pursuer",
                        "entities": (i, j),
                        "distance": float(distance),
                        "threshold": float(threshold_pp),
                    }
        for i in range(self.num_pursuers):
            for obstacle in self.obstacles:
                threshold = self.pursuer_radius + obstacle.radius + self.safety_buffer_scale * (self.safety_buffer + obstacle.buffer)
                distance = norm(self.pursuer_pos[i] - obstacle.pos)
                if distance < threshold:
                    return {
                        "collision": True,
                        "type": "obstacle",
                        "entities": (i,),
                        "distance": float(distance),
                        "threshold": float(threshold),
                    }
        return {
            "collision": False,
            "type": "none",
            "entities": (),
            "distance": float("inf"),
            "threshold": 0.0,
        }

    def compute_rewards(self, collision: bool | None = None, finished: np.ndarray | None = None) -> np.ndarray:
        """GERD reward from Section IV-C.1, Eqs. (15)-(18)."""
        if collision is None:
            collision = self.check_collision()
        if finished is None:
            finished = self.check_finished_each_agent()

        aggregate = np.sum(self.target_pos[None, :] - self.pursuer_pos, axis=0)
        r_f = math.exp(-self.k1 * norm(aggregate)) - 1.0
        distances = np.linalg.norm(self.pursuer_pos - self.target_pos[None, :], axis=1)
        dist_error_sum = float(np.sum((distances - self.capture_distance) ** 2))
        r_d = math.exp(-self.k2 * dist_error_sum) - 1.0
        r_l = self.collision_penalty if collision else 0.0
        r_step = self.w1 * r_f + self.w2 * r_d + self.w3 * r_l
        if self.angle_shaping_weight > 0.0 or self.distance_shaping_weight > 0.0:
            # Engineering shaping for the simplified planar-circle environment:
            # the paper's GERD terms do not give a dense angular-spacing signal,
            # while the final success condition directly depends on Eq. (2)'s
            # angle and distance tolerances.
            distance_error, angle_error = self.encirclement_errors()
            r_step -= self.distance_shaping_weight * distance_error
            r_step -= self.angle_shaping_weight * angle_error

        if bool(np.all(finished)):
            return np.full(self.num_pursuers, self.global_reward, dtype=float)
        rewards = np.full(self.num_pursuers, r_step, dtype=float)
        rewards[finished] = self.local_reward
        return rewards

    def encirclement_errors(self) -> Tuple[float, float]:
        """Mean distance and angular errors for Eq. (2)'s encirclement target."""
        distances = np.linalg.norm(self.pursuer_pos - self.target_pos[None, :], axis=1)
        distance_error = float(np.mean(np.abs(distances - self.capture_distance)))
        neighbors = neighbor_indices(self.pursuer_pos, self.target_pos)
        alpha_exp = 2.0 * math.pi / self.num_pursuers
        errors = []
        for i in range(self.num_pursuers):
            _, right = neighbors[i]
            alpha = included_angle(self.pursuer_pos[i], self.pursuer_pos[right], self.target_pos)
            errors.append(abs(alpha - alpha_exp))
        return distance_error, float(np.mean(errors))

    def compute_observations(self) -> np.ndarray:
        neighbors = neighbor_indices(self.pursuer_pos, self.target_pos)
        observations = [self._observation_for(i, neighbors) for i in range(self.num_pursuers)]
        return np.vstack(observations)

    def _observation_for(self, i: int, neighbors: Dict[int, Tuple[int, int]]) -> np.ndarray:
        left, right = neighbors[i]
        obstacle_idx = self._closest_obstacle_index(i)
        parts = [
            self._neighbor_features(i, left),
            self._neighbor_features(i, right),
            self._target_features(i),
            self._obstacle_features(i, left, right, obstacle_idx),
        ]
        obs = np.concatenate(parts).astype(float)
        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"Expected observation dim {self.obs_dim}, got {obs.shape[0]}")
        return obs

    def _neighbor_features(self, i: int, j: int) -> np.ndarray:
        rel = self.pursuer_pos[j] - self.pursuer_pos[i]
        alpha = included_angle(self.pursuer_pos[i], self.pursuer_pos[j], self.target_pos)
        distance = norm(rel)
        rel_speed = norm(self.pursuer_vel[j] - self.pursuer_vel[i])
        q_ij = bearing_angle(self.pursuer_vel[i], rel)
        q_ji = bearing_angle(self.pursuer_vel[j], -rel)
        return np.array([alpha, distance, rel_speed, q_ij, q_ji], dtype=float)

    def _target_features(self, i: int) -> np.ndarray:
        rel = self.target_pos - self.pursuer_pos[i]
        distance = norm(rel)
        rel_vel = self.target_vel - self.pursuer_vel[i]
        dot_distance = float(np.dot(rel, rel_vel) / max(distance, 1e-8))
        q_it = bearing_angle(self.pursuer_vel[i], rel)
        # Engineering assumption: angular derivatives are zero in this minimal version.
        return np.array([distance, dot_distance, q_it, 0.0, 0.0, 0.0], dtype=float)

    def _obstacle_features(self, i: int, left: int, right: int, obstacle_idx: int) -> np.ndarray:
        obstacle = self.obstacles[obstacle_idx]
        rel = obstacle.pos - self.pursuer_pos[i]
        d_i = max(norm(rel) - obstacle.radius, 0.0)
        d_left = max(norm(obstacle.pos - self.pursuer_pos[left]) - obstacle.radius, 0.0)
        d_right = max(norm(obstacle.pos - self.pursuer_pos[right]) - obstacle.radius, 0.0)
        return np.array([rel[0], rel[1], d_i, d_left, d_right, obstacle.radius, self.pursuer_vel[i, 0], self.pursuer_vel[i, 1]], dtype=float)

    def _closest_obstacle_index(self, i: int) -> int:
        if not self.obstacles:
            self.obstacles = [Obstacle(1000.0, 1000.0, 0.0, 0.0)]
        distances = [norm(obstacle.pos - self.pursuer_pos[i]) - obstacle.radius for obstacle in self.obstacles]
        return int(np.argmin(distances))
