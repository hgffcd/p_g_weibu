"""Metrics used in the paper's Section V experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from envs.encirclement_env import EncirclementEnv
from utils.geometry import included_angle, neighbor_indices, norm


@dataclass
class StepMetric:
    reward: float
    distance_error: float
    angle_error: float
    min_obstacle_clearance: float
    min_pursuer_clearance: float
    danger: bool


@dataclass
class EpisodeMetric:
    success: bool
    collision: bool
    timeout: bool
    steps: int
    encirclement_time: float | None
    average_step_reward: float
    danger_rate: float
    final_distance_error: float
    final_angle_error: float
    min_obstacle_clearance: float
    min_pursuer_clearance: float
    step_metrics: List[StepMetric] = field(default_factory=list)


def distance_error(env: EncirclementEnv) -> float:
    """Fig. 14(a): e_d = mean_i |d_iT - d_c|."""
    distances = np.linalg.norm(env.pursuer_pos - env.target_pos[None, :], axis=1)
    return float(np.mean(np.abs(distances - env.capture_distance)))


def angle_error(env: EncirclementEnv) -> float:
    """Fig. 14(b): e_alpha = mean_i |alpha_i,i+1 - alpha_exp|."""
    neighbors = neighbor_indices(env.pursuer_pos, env.target_pos)
    alpha_exp = 2.0 * math.pi / env.num_pursuers
    errors = []
    for i in range(env.num_pursuers):
        _, right = neighbors[i]
        alpha = included_angle(env.pursuer_pos[i], env.pursuer_pos[right], env.target_pos)
        errors.append(abs(alpha - alpha_exp))
    return float(np.mean(errors))


def min_obstacle_clearance(env: EncirclementEnv) -> float:
    """Fig. 13(a): minimum pursuer-obstacle clearance."""
    if not env.obstacles:
        return float("inf")
    clearances = []
    for i in range(env.num_pursuers):
        for obstacle in env.obstacles:
            clearances.append(norm(env.pursuer_pos[i] - obstacle.pos) - obstacle.radius)
    return float(min(clearances))


def min_pursuer_clearance(env: EncirclementEnv) -> float:
    """Fig. 13(b): minimum clearance among pursuers."""
    clearances = []
    for i in range(env.num_pursuers):
        for j in range(i + 1, env.num_pursuers):
            clearances.append(norm(env.pursuer_pos[i] - env.pursuer_pos[j]) - env.pursuer_radius)
    return float(min(clearances)) if clearances else float("inf")


def danger_flag(env: EncirclementEnv) -> bool:
    """Danger event from Section V-D: entities are closer than safety-buffer threshold."""
    return env.check_collision()


def step_metric(env: EncirclementEnv, reward: np.ndarray) -> StepMetric:
    return StepMetric(
        reward=float(np.mean(reward)),
        distance_error=distance_error(env),
        angle_error=angle_error(env),
        min_obstacle_clearance=min_obstacle_clearance(env),
        min_pursuer_clearance=min_pursuer_clearance(env),
        danger=danger_flag(env),
    )


def summarize_episode(step_metrics: List[StepMetric], info: Dict[str, Any], steps: int, dt: float) -> EpisodeMetric:
    if not step_metrics:
        raise ValueError("Cannot summarize an empty episode")
    rewards = [m.reward for m in step_metrics]
    danger_count = sum(1 for m in step_metrics if m.danger)
    success = bool(info.get("success", False))
    return EpisodeMetric(
        success=success,
        collision=bool(info.get("collision", False)),
        timeout=bool(info.get("timeout", False)),
        steps=int(steps),
        encirclement_time=float(steps * dt) if success else None,
        average_step_reward=float(np.mean(rewards)),
        danger_rate=float(danger_count / len(step_metrics)),
        final_distance_error=step_metrics[-1].distance_error,
        final_angle_error=step_metrics[-1].angle_error,
        min_obstacle_clearance=float(min(m.min_obstacle_clearance for m in step_metrics)),
        min_pursuer_clearance=float(min(m.min_pursuer_clearance for m in step_metrics)),
        step_metrics=step_metrics,
    )


def aggregate_episodes(episodes: List[EpisodeMetric]) -> Dict[str, float]:
    """Aggregate Table IV/V style metrics over Monte Carlo episodes."""
    if not episodes:
        raise ValueError("No episodes to aggregate")
    successes = [e.success for e in episodes]
    enc_times = [e.encirclement_time for e in episodes if e.encirclement_time is not None]
    return {
        "episodes": float(len(episodes)),
        "average_step_reward": float(np.mean([e.average_step_reward for e in episodes])),
        "encirclement_time": float(np.mean(enc_times)) if enc_times else float("nan"),
        "danger_rate": float(np.mean([e.danger_rate for e in episodes])),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean([e.collision for e in episodes])),
        "timeout_rate": float(np.mean([e.timeout for e in episodes])),
        "final_distance_error": float(np.mean([e.final_distance_error for e in episodes])),
        "final_angle_error": float(np.mean([e.final_angle_error for e in episodes])),
        "min_obstacle_clearance": float(np.min([e.min_obstacle_clearance for e in episodes])),
        "min_pursuer_clearance": float(np.min([e.min_pursuer_clearance for e in episodes])),
    }


def communication_meta_messages(num_pursuers: int, pursuer_dim: int = 5, target_dim: int = 6, obstacle_dim: int = 8) -> Dict[str, float]:
    """Communication load from Section V-C.

    Bidirectional protocol: (2*Np + NT + No) * N, O(N).
    Fully connected: ((N-1)*Np + NT + No) * N, O(N^2).
    """
    n = int(num_pursuers)
    bidirectional = (2 * pursuer_dim + target_dim + obstacle_dim) * n
    fully_connected = ((n - 1) * pursuer_dim + target_dim + obstacle_dim) * n
    return {
        "bidirectional_meta_messages": float(bidirectional),
        "fully_connected_meta_messages": float(fully_connected),
        "communication_reduction": float(1.0 - bidirectional / fully_connected) if fully_connected else 0.0,
    }
