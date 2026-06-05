"""APF guide policy from Section IV-A, Eqs. (7)-(10).

The optional ``environment_py`` mode mirrors the user-provided ``environment.py``
``policy_u`` implementation while keeping this project independent from its MPE
and onpolicy dependencies.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from envs.encirclement_env import EncirclementEnv
from utils.geometry import clip_by_component, norm


def limit_action_inf_norm(action: np.ndarray, max_limit: float) -> np.ndarray:
    """Infinity-norm action limiter copied from the user-provided environment.py."""
    action = np.asarray(action, dtype=np.float32).copy()
    max_limit = float(max_limit)
    if abs(action[0]) > abs(action[1]):
        if abs(action[0]) > max_limit:
            action[1] = max_limit * action[1] / max(abs(action[0]), 1e-8)
            action[0] = max_limit if action[0] > 0 else -max_limit
    else:
        if abs(action[1]) > max_limit:
            action[0] = max_limit * action[0] / max(abs(action[1]), 1e-8)
            action[1] = max_limit if action[1] > 0 else -max_limit
    return action.astype(float)


class APFGuidePolicy:
    def __init__(self, config: Dict[str, Any]):
        cfg = config["guide_policy"]
        env_cfg = config["environment"]
        self.mode = str(cfg.get("mode", cfg.get("variant", "paper")))
        self.kr = float(cfg["kr"])
        self.ko = float(cfg["ko"])
        self.ka = float(cfg["ka"])
        self.kb = float(cfg["kb"])
        self.obstacle_threshold = float(cfg["obstacle_threshold"])
        self.capture_distance = float(env_cfg["capture_distance"])
        self.max_acceleration = float(env_cfg["max_acceleration"])
        self.target_gain = float(cfg.get("target_gain", self.ka))
        self.target_velocity_gain = float(cfg.get("target_velocity_gain", 1.5))
        self.inner_target_gain_scale = float(cfg.get("inner_target_gain_scale", 20.0))
        self.far_target_distance = float(cfg.get("far_target_distance", 1.5))
        self.obstacle_extra_margin = float(cfg.get("obstacle_extra_margin", 0.3))
        self.cancel_backward_repulsion = bool(cfg.get("cancel_backward_repulsion", True))
        self.safety_filter = bool(cfg.get("safety_filter", self.mode == "environment_py"))
        self.safety_gain = float(cfg.get("safety_gain", 6.0))
        self.safety_margin = float(cfg.get("safety_margin", 0.35))
        self.safety_damping = float(cfg.get("safety_damping", 1.0))
        self.formation_correction = bool(cfg.get("formation_correction", False))
        self.formation_gain = float(cfg.get("formation_gain", 2.0))
        self.formation_damping = float(cfg.get("formation_damping", 1.0))
        self.formation_weight = float(cfg.get("formation_weight", 1.0))
        self.formation_start_step = int(cfg.get("formation_start_step", 0))
        self.formation_ramp_steps = max(1, int(cfg.get("formation_ramp_steps", 1)))
        self.formation_activate_distance_error = float(cfg.get("formation_activate_distance_error", 1.0))
        self.formation_obstacle_gate_margin = float(cfg.get("formation_obstacle_gate_margin", 0.4))
        self.formation_slot_clearance_margin = float(cfg.get("formation_slot_clearance_margin", 0.2))
        self.formation_slot_obstacle_weight = float(cfg.get("formation_slot_obstacle_weight", 20.0))
        self.formation_path_obstacle_weight = float(cfg.get("formation_path_obstacle_weight", 8.0))
        self.formation_candidate_count = max(3, int(cfg.get("formation_candidate_count", 15)))
        self._formation_cache_key: tuple[Any, ...] | None = None
        self._formation_cache_actions: np.ndarray | None = None

    def act(self, env: EncirclementEnv) -> np.ndarray:
        actions = np.vstack([self._agent_action(env, i) for i in range(env.num_pursuers)])
        if self.mode == "environment_py":
            if self.safety_filter:
                actions = self._safety_filter(env, actions)
            return np.vstack([limit_action_inf_norm(action, self.max_acceleration) for action in actions])
        return clip_by_component(actions, self.max_acceleration)

    def _agent_action(self, env: EncirclementEnv, i: int) -> np.ndarray:
        if self.mode == "environment_py":
            return self._environment_py_agent_action(env, i)

        alpha_exp = 2.0 * math.pi / env.num_pursuers
        desired_neighbor_distance = 2.0 * self.capture_distance * math.sin(alpha_exp / 2.0)
        total = np.zeros(2, dtype=float)
        p_i = env.pursuer_pos[i]

        # Eq. (7): repelling force between pursuers.
        for j in range(env.num_pursuers):
            if j == i:
                continue
            diff = p_i - env.pursuer_pos[j]
            d_ij = max(norm(diff), 1e-8)
            if d_ij <= desired_neighbor_distance:
                total += self.kr * (desired_neighbor_distance - d_ij) / d_ij * diff

        # Eq. (8): repelling force between pursuers and obstacles.
        for obstacle in env.obstacles:
            diff = p_i - obstacle.pos
            d_io = max(norm(diff) - obstacle.radius, 1e-8)
            if d_io <= self.obstacle_threshold:
                total += self.ko * (self.obstacle_threshold - d_io) / d_io * diff

        # Eq. (9): target attraction.
        to_target = env.target_pos - p_i
        d_it = max(norm(to_target), 1e-8)
        total += self.ka * (d_it - self.capture_distance) / d_it * to_target

        # Eq. (10): damping.
        total -= self.kb * env.pursuer_vel[i]
        return total

    def _environment_py_agent_action(self, env: EncirclementEnv, i: int) -> np.ndarray:
        """APF guide translated from user-provided environment.py::policy_u.

        This takes priority for the current reproduction request. It keeps the
        same ingredients as Eqs. (7)-(10), but follows the concrete force law and
        constants used in environment.py.
        """
        p_i = env.pursuer_pos[i]
        v_i = env.pursuer_vel[i]
        d_cap = self.capture_distance
        desired_neighbor_distance = 2.0 * d_cap * math.sin(math.pi / env.num_pursuers)

        r_ic = env.target_pos - p_i
        norm_r_ic = max(norm(r_ic), 1e-8)
        vel_vec = env.target_vel - v_i
        distance_error = norm_r_ic - d_cap
        if distance_error > 0.0:
            if distance_error > self.far_target_distance:
                f_c = self.far_target_distance / norm_r_ic * r_ic + self.target_velocity_gain * vel_vec
            else:
                f_c = self.target_gain * distance_error / norm_r_ic * r_ic + self.target_velocity_gain * vel_vec
        else:
            f_c = (
                self.inner_target_gain_scale
                * self.target_gain
                * distance_error
                / norm_r_ic
                * r_ic
                + self.target_velocity_gain * vel_vec
            )

        f_r = np.zeros(2, dtype=float)
        for j in range(env.num_pursuers):
            if j == i:
                continue
            r_ij = p_i - env.pursuer_pos[j]
            norm_r_ij = max(norm(r_ij), 1e-8)
            if norm_r_ij < desired_neighbor_distance:
                force = self.kr * (desired_neighbor_distance - norm_r_ij) / norm_r_ij * r_ij
                if self.cancel_backward_repulsion and np.dot(force, r_ic) < 0.0 and norm_r_ij > 2.0 * desired_neighbor_distance / 3.0:
                    force = force - np.dot(force, r_ic) / max(np.dot(r_ic, r_ic), 1e-8) * r_ic
                f_r += force

        f_obs = np.zeros(2, dtype=float)
        for obstacle in env.obstacles:
            d_io = p_i - obstacle.pos
            norm_d_io = max(norm(d_io), 1e-8)
            min_distance = env.pursuer_radius + env.safety_buffer + obstacle.radius + obstacle.buffer
            active_distance = min_distance + self.obstacle_extra_margin
            if norm_d_io < active_distance:
                f_obs += self.ko * (active_distance - norm_d_io) / norm_d_io * d_io

        total = f_c + f_r + f_obs - self.kb * v_i
        if self.formation_correction:
            gate = self._formation_gate(env, i)
            total += self.formation_weight * gate * self._formation_slot_action(env, i)
        return total

    def _formation_slot_action(self, env: EncirclementEnv, i: int) -> np.ndarray:
        """Dense ring-formation correction for Eq. (2)'s angular condition.

        The uploaded environment.py guide regulates target distance, pursuer
        repulsion, obstacle repulsion and damping, but it has no explicit term
        that assigns pursuers to evenly spaced slots around the target. In the
        simplified planar-circle environment this leaves a persistent angular
        error and all final evaluations time out. This term preserves the
        current cyclic order of agents and applies a PD correction toward an
        evenly spaced capture ring.
        """
        actions = self._formation_slot_actions(env)
        return actions[i]

    def _formation_slot_actions(self, env: EncirclementEnv) -> np.ndarray:
        """Compute all ring-slot corrections once per environment state."""
        cache_key = (
            id(env),
            int(env.step_count),
            env.pursuer_pos.tobytes(),
            env.pursuer_vel.tobytes(),
            env.target_pos.tobytes(),
            env.target_vel.tobytes(),
            tuple((o.x, o.y, o.radius, o.buffer) for o in env.obstacles),
        )
        if self._formation_cache_key == cache_key and self._formation_cache_actions is not None:
            return self._formation_cache_actions

        rel = env.pursuer_pos - env.target_pos[None, :]
        current_angles = np.arctan2(rel[:, 1], rel[:, 0])
        order = np.argsort(current_angles)
        sorted_angles = np.unwrap(current_angles[order])
        spacing = 2.0 * math.pi / env.num_pursuers
        ranks = np.arange(env.num_pursuers, dtype=float)
        base_angle = self._safe_formation_base_angle(env, order, sorted_angles, spacing, ranks)
        desired_angles = base_angle + spacing * ranks

        desired_pos = np.zeros_like(env.pursuer_pos)
        for rank, pursuer_idx in enumerate(order):
            angle = desired_angles[rank]
            desired_pos[pursuer_idx] = env.target_pos + self.capture_distance * np.array(
                [math.cos(angle), math.sin(angle)],
                dtype=float,
            )

        target_velocity = env.target_vel
        actions = self.formation_gain * (desired_pos - env.pursuer_pos) - self.formation_damping * (
            env.pursuer_vel - target_velocity[None, :]
        )
        self._formation_cache_key = cache_key
        self._formation_cache_actions = actions
        return actions

    def _formation_gate(self, env: EncirclementEnv, i: int) -> float:
        """Delay ring-slot correction until approach and obstacle clearance are safe."""
        if env.step_count < self.formation_start_step:
            return 0.0
        time_gate = min(
            max((env.step_count - self.formation_start_step) / float(self.formation_ramp_steps), 0.0),
            1.0,
        )
        distances = np.linalg.norm(env.pursuer_pos - env.target_pos[None, :], axis=1)
        radial_error = float(np.mean(np.abs(distances - self.capture_distance)))
        radial_gate = 1.0 - radial_error / max(self.formation_activate_distance_error, 1e-8)
        radial_gate = min(max(radial_gate, 0.0), 1.0)

        clearance = self._agent_obstacle_clearance(env, i)
        obstacle_gate = min(max(clearance / max(self.formation_obstacle_gate_margin, 1e-8), 0.0), 1.0)
        return float(time_gate * radial_gate * obstacle_gate)

    def _safe_formation_base_angle(
        self,
        env: EncirclementEnv,
        order: np.ndarray,
        sorted_angles: np.ndarray,
        spacing: float,
        ranks: np.ndarray,
    ) -> float:
        """Choose a ring orientation whose slots and straight-line approach avoid obstacles."""
        nominal_base = float(np.mean(sorted_angles - spacing * ranks))
        candidate_offsets = np.linspace(-spacing, spacing, self.formation_candidate_count)
        best_base = nominal_base
        best_score = float("inf")
        for offset in candidate_offsets:
            base = nominal_base + float(offset)
            desired = np.zeros_like(env.pursuer_pos)
            for rank, pursuer_idx in enumerate(order):
                angle = base + spacing * rank
                desired[pursuer_idx] = env.target_pos + self.capture_distance * np.array(
                    [math.cos(angle), math.sin(angle)],
                    dtype=float,
                )
            score = self._formation_slot_score(env, desired)
            if score < best_score:
                best_score = score
                best_base = base
        return best_base

    def _formation_slot_score(self, env: EncirclementEnv, desired: np.ndarray) -> float:
        movement_score = float(np.mean(np.sum((desired - env.pursuer_pos) ** 2, axis=1)))
        obstacle_score = 0.0
        for i in range(env.num_pursuers):
            for obstacle in env.obstacles:
                threshold = self._obstacle_threshold(env, obstacle) + self.formation_slot_clearance_margin
                slot_clearance = norm(desired[i] - obstacle.pos) - threshold
                if slot_clearance < 0.0:
                    obstacle_score += self.formation_slot_obstacle_weight * slot_clearance * slot_clearance
                path_clearance = self._segment_point_distance(env.pursuer_pos[i], desired[i], obstacle.pos) - threshold
                if path_clearance < 0.0:
                    obstacle_score += self.formation_path_obstacle_weight * path_clearance * path_clearance
        return movement_score + obstacle_score

    def _agent_obstacle_clearance(self, env: EncirclementEnv, i: int) -> float:
        if not env.obstacles:
            return float("inf")
        return float(min(norm(env.pursuer_pos[i] - obstacle.pos) - self._obstacle_threshold(env, obstacle) for obstacle in env.obstacles))

    def _obstacle_threshold(self, env: EncirclementEnv, obstacle: Any) -> float:
        return (
            env.pursuer_radius
            + obstacle.radius
            + env.safety_buffer_scale * (env.safety_buffer + obstacle.buffer)
        )

    @staticmethod
    def _segment_point_distance(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            return norm(p - a)
        t = min(max(float(np.dot(p - a, ab) / denom), 0.0), 1.0)
        closest = a + t * ab
        return norm(p - closest)

    def _safety_filter(self, env: EncirclementEnv, actions: np.ndarray) -> np.ndarray:
        """Conservative safety layer around environment.py::policy_u.

        The uploaded environment.py relied on a different MPE dynamics stack.
        In this planar second-order model, the translated guide can collide
        before the learned actor receives useful samples. This filter adds only
        short-range collision-avoidance accelerations and keeps the original
        guide as the nominal command.
        """
        adjusted = np.asarray(actions, dtype=float).copy()
        dt = max(float(env.dt), 1e-8)
        pred_vel = clip_by_component(env.pursuer_vel + adjusted * dt, env.max_velocity)
        pred_pos = env.pursuer_pos + pred_vel * dt

        pair_threshold = 2.0 * env.pursuer_radius + env.safety_buffer_scale * (2.0 * env.safety_buffer)
        pair_active = pair_threshold + self.safety_margin
        for i in range(env.num_pursuers):
            for j in range(i + 1, env.num_pursuers):
                current_diff = env.pursuer_pos[i] - env.pursuer_pos[j]
                predicted_diff = pred_pos[i] - pred_pos[j]
                current_distance = max(norm(current_diff), 1e-8)
                predicted_distance = max(norm(predicted_diff), 1e-8)
                distance = min(current_distance, predicted_distance)
                if distance >= pair_active:
                    continue
                direction = predicted_diff / predicted_distance
                rel_vel = env.pursuer_vel[i] - env.pursuer_vel[j]
                closing_speed = max(0.0, -float(np.dot(rel_vel, direction)))
                force = (
                    self.safety_gain * (pair_active - distance) / pair_active
                    + self.safety_damping * closing_speed
                ) * direction
                adjusted[i] += force
                adjusted[j] -= force

        for i in range(env.num_pursuers):
            for obstacle in env.obstacles:
                threshold = env.pursuer_radius + obstacle.radius + env.safety_buffer_scale * (env.safety_buffer + obstacle.buffer)
                active = threshold + self.safety_margin
                current_diff = env.pursuer_pos[i] - obstacle.pos
                predicted_diff = pred_pos[i] - obstacle.pos
                current_distance = max(norm(current_diff), 1e-8)
                predicted_distance = max(norm(predicted_diff), 1e-8)
                distance = min(current_distance, predicted_distance)
                if distance >= active:
                    continue
                direction = predicted_diff / predicted_distance
                closing_speed = max(0.0, -float(np.dot(env.pursuer_vel[i], direction)))
                force = (
                    self.safety_gain * (active - distance) / max(active, 1e-8)
                    + self.safety_damping * closing_speed
                ) * direction
                adjusted[i] += force

        return adjusted
