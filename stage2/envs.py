"""Stage-2 environments with reset randomization.

The original project files are left untouched. This module subclasses the
existing planar environment and adds optional target, obstacle, and pursuer
initial-state randomization for the next reproduction stage.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import numpy as np

from envs.encirclement_env import EncirclementEnv, Obstacle
from utils.geometry import norm


class Stage2EncirclementEnv(EncirclementEnv):
    """Planar EMOCA environment with feasible randomized resets.

    This keeps the paper-derived dynamics, observations, rewards, and terminal
    checks from ``EncirclementEnv``. Only reset-time sampling is extended.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(copy.deepcopy(config))
        self.stage2_cfg = self.config.get("stage2", {})
        self.last_reset_info: Dict[str, Any] = {}

    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        rng = np.random.default_rng(seed)
        if not self._randomization_enabled():
            obs, state = super().reset(seed=seed)
            self.last_reset_info = {
                "seed": seed,
                "attempt": 1,
                "randomized": False,
                "target_pos": self.target_pos.copy(),
                "pursuer_pos": self.pursuer_pos.copy(),
                "obstacles": np.array([[o.x, o.y, o.radius, o.buffer] for o in self.obstacles], dtype=float),
            }
            return obs, state

        max_attempts = int(self.stage2_cfg.get("feasibility", {}).get("max_reset_attempts", 200))
        reason = ""
        for attempt in range(1, max_attempts + 1):
            target_pos = self._sample_target(rng)
            obstacles = self._sample_obstacles(rng)
            pursuer_pos = self._sample_pursuers(rng, target_pos, obstacles)
            if pursuer_pos is None:
                continue
            feasible, reason = self._is_feasible(target_pos, pursuer_pos, obstacles)
            if not feasible:
                continue

            self.target_pos = target_pos.astype(float)
            self.target_vel = np.zeros(2, dtype=float)
            self.pursuer_pos = pursuer_pos.astype(float)
            self.pursuer_vel = np.zeros((self.num_pursuers, 2), dtype=float)
            self.obstacles = [Obstacle(o.x, o.y, o.radius, o.buffer) for o in obstacles]
            self.base_obstacles = [Obstacle(o.x, o.y, o.radius, o.buffer) for o in obstacles]
            self.finished = np.zeros(self.num_pursuers, dtype=bool)
            self.step_count = 0
            self.last_reset_info = {
                "seed": seed,
                "attempt": attempt,
                "randomized": True,
                "feasible": True,
                "rejection_reason": "",
                "target_pos": self.target_pos.copy(),
                "pursuer_pos": self.pursuer_pos.copy(),
                "obstacles": np.array([[o.x, o.y, o.radius, o.buffer] for o in self.obstacles], dtype=float),
            }
            return self.compute_observations(), self.get_state()

        raise RuntimeError(f"Failed to sample a feasible stage2 reset after {max_attempts} attempts; last reason={reason}")

    def _randomization_enabled(self) -> bool:
        return bool(
            self.stage2_cfg.get("target_randomization", {}).get("enabled", False)
            or self.stage2_cfg.get("pursuer_randomization", {}).get("enabled", False)
            or self.stage2_cfg.get("obstacle_randomization", {}).get("enabled", False)
        )

    def _sample_target(self, rng: np.random.Generator) -> np.ndarray:
        cfg = self.stage2_cfg.get("target_randomization", {})
        if not bool(cfg.get("enabled", False)):
            return self.target_initial_position.astype(float).copy()
        x_min, x_max = cfg.get("x_range", [-1.0, 1.0])
        y_min, y_max = cfg.get("y_range", [2.5, 4.5])
        return np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)

    def _sample_obstacles(self, rng: np.random.Generator) -> List[Obstacle]:
        cfg = self.stage2_cfg.get("obstacle_randomization", {})
        obstacles = [Obstacle(o.x, o.y, o.radius, o.buffer) for o in self.base_obstacles]
        if not bool(cfg.get("enabled", False)):
            return obstacles
        pos_jitter = float(cfg.get("position_jitter", 0.0))
        radius_jitter = float(cfg.get("radius_jitter", 0.0))
        min_radius = float(cfg.get("min_radius", 0.05))
        randomized = []
        for obstacle in obstacles:
            dx, dy = rng.uniform(-pos_jitter, pos_jitter, size=2)
            radius = max(min_radius, obstacle.radius + float(rng.uniform(-radius_jitter, radius_jitter)))
            randomized.append(Obstacle(obstacle.x + float(dx), obstacle.y + float(dy), radius, obstacle.buffer))
        return randomized

    def _sample_pursuers(
        self,
        rng: np.random.Generator,
        target_pos: np.ndarray,
        obstacles: List[Obstacle],
    ) -> np.ndarray | None:
        cfg = self.stage2_cfg.get("pursuer_randomization", {})
        if not bool(cfg.get("enabled", False)):
            offsets = (np.arange(self.num_pursuers) - (self.num_pursuers - 1) / 2.0) * self.initial_spacing
            return np.stack([offsets, np.full_like(offsets, self.initial_y)], axis=1).astype(float)

        mode = str(cfg.get("mode", "line_below_target"))
        max_attempts = int(cfg.get("max_attempts", 80))
        if mode == "box":
            x_min, x_max = cfg.get("x_range", [-2.0, 2.0])
            y_min, y_max = cfg.get("y_range", [-0.5, 1.0])
            for _ in range(max_attempts):
                positions = np.column_stack([
                    rng.uniform(x_min, x_max, size=self.num_pursuers),
                    rng.uniform(y_min, y_max, size=self.num_pursuers),
                ])
                feasible, _ = self._pursuers_feasible(positions, obstacles)
                if feasible:
                    return positions.astype(float)
            return None

        initial_distance = float(cfg.get("initial_distance", 3.5))
        spacing = float(cfg.get("spacing", self.initial_spacing))
        x_jitter = float(cfg.get("x_jitter", 0.25))
        y_jitter = float(cfg.get("y_jitter", 0.15))
        offsets = (np.arange(self.num_pursuers) - (self.num_pursuers - 1) / 2.0) * spacing
        for _ in range(max_attempts):
            positions = np.stack(
                [
                    target_pos[0] + offsets + rng.uniform(-x_jitter, x_jitter, size=self.num_pursuers),
                    np.full(self.num_pursuers, target_pos[1] - initial_distance)
                    + rng.uniform(-y_jitter, y_jitter, size=self.num_pursuers),
                ],
                axis=1,
            )
            feasible, _ = self._pursuers_feasible(positions, obstacles)
            if feasible:
                return positions.astype(float)
        return None

    def _is_feasible(
        self,
        target_pos: np.ndarray,
        pursuer_pos: np.ndarray,
        obstacles: List[Obstacle],
    ) -> Tuple[bool, str]:
        cfg = self.stage2_cfg.get("feasibility", {})
        margin = float(cfg.get("margin", 0.05))
        ring_margin = float(cfg.get("capture_ring_margin", 0.08))

        for obstacle in obstacles:
            center_distance = norm(target_pos - obstacle.pos)
            target_threshold = self.target_radius + obstacle.radius + self.safety_buffer + obstacle.buffer + margin
            if center_distance <= target_threshold:
                return False, "target_inside_obstacle_safety"
            ring_threshold = obstacle.radius + self.pursuer_radius + self.safety_buffer + obstacle.buffer + ring_margin
            if abs(center_distance - self.capture_distance) <= ring_threshold:
                return False, "capture_ring_intersects_obstacle_safety"

        pursuers_ok, reason = self._pursuers_feasible(pursuer_pos, obstacles)
        if not pursuers_ok:
            return False, reason
        return True, ""

    def _pursuers_feasible(self, pursuer_pos: np.ndarray, obstacles: List[Obstacle]) -> Tuple[bool, str]:
        cfg = self.stage2_cfg.get("feasibility", {})
        margin = float(cfg.get("margin", 0.05))
        threshold_pp = 2 * self.pursuer_radius + self.safety_buffer_scale * (2 * self.safety_buffer) + margin
        for i in range(self.num_pursuers):
            for j in range(i + 1, self.num_pursuers):
                if norm(pursuer_pos[i] - pursuer_pos[j]) <= threshold_pp:
                    return False, "pursuer_pursuer_initial_collision"
        for i in range(self.num_pursuers):
            for obstacle in obstacles:
                threshold = (
                    self.pursuer_radius
                    + obstacle.radius
                    + self.safety_buffer_scale * (self.safety_buffer + obstacle.buffer)
                    + margin
                )
                if norm(pursuer_pos[i] - obstacle.pos) <= threshold:
                    return False, "pursuer_obstacle_initial_collision"
        return True, ""


class Stage2VectorizedEncirclementEnv:
    """Synchronous vector wrapper that uses Stage2EncirclementEnv."""

    def __init__(self, config: Dict[str, Any], num_envs: int):
        self.config = config
        self.num_envs = int(num_envs)
        self.envs = [Stage2EncirclementEnv(copy.deepcopy(config)) for _ in range(self.num_envs)]
        self.num_pursuers = self.envs[0].num_pursuers
        self.obs_dim = self.envs[0].obs_dim
        self.state_dim = self.envs[0].flat_state().shape[0]
        self.max_steps = self.envs[0].max_steps
        self.step_in_episode = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_counts = np.zeros(self.num_envs, dtype=np.int64)

    def reset(self, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        observations = []
        states = []
        for idx, env in enumerate(self.envs):
            obs, _ = env.reset(seed=seed + idx)
            observations.append(obs)
            states.append(env.flat_state())
        self.step_in_episode[:] = 0
        self.episode_counts[:] = 0
        return np.stack(observations).astype(np.float32), np.stack(states).astype(np.float32)

    def flat_states(self) -> np.ndarray:
        return np.stack([env.flat_state() for env in self.envs]).astype(np.float32)

    def observations(self) -> np.ndarray:
        return np.stack([env.compute_observations() for env in self.envs]).astype(np.float32)

    def step(self, actions: np.ndarray, seed_base: int = 0):
        actions = np.asarray(actions, dtype=np.float32)
        observations = []
        rewards = []
        dones = []
        infos = []
        states = []
        for idx, env in enumerate(self.envs):
            obs, reward, done, info = env.step(actions[idx])
            self.step_in_episode[idx] += 1
            if done:
                self.episode_counts[idx] += 1
                info = dict(info)
                info["terminal_step_count"] = int(self.step_in_episode[idx])
                info["reset_info"] = env.last_reset_info
                obs, _ = env.reset(seed=seed_base + int(self.episode_counts[idx]) * self.num_envs + idx)
                self.step_in_episode[idx] = 0
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            states.append(env.flat_state())
        return (
            np.stack(observations).astype(np.float32),
            np.stack(rewards).astype(np.float32),
            np.asarray(dones, dtype=bool),
            infos,
            np.stack(states).astype(np.float32),
        )

    def guide_actions(self, guide_policy) -> np.ndarray:
        return np.stack([guide_policy.act(env) for env in self.envs]).astype(np.float32)

    def apply_curriculum(self, delta_alpha: float, delta_distance: float, obstacles) -> None:
        for env in self.envs:
            env.delta_alpha = float(delta_alpha)
            env.delta_distance = float(delta_distance)
            env.base_obstacles = [copy.copy(o) for o in obstacles]
            if not env.stage2_cfg.get("obstacle_randomization", {}).get("enabled", False):
                env.obstacles = [copy.copy(o) for o in obstacles]

    def set_target_max_velocity(self, value: float) -> None:
        for env in self.envs:
            env.target_max_velocity = float(value)
