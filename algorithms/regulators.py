"""Curriculum regulators from Section IV-A/B, Eqs. (11)-(14)."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple


def jsrl_regulator(x: float, k_s: float) -> float:
    """Eq. (11): S-shaped guide policy phase-out regulator."""
    x = min(max(float(x), 0.0), 1.0)
    x_mapped = 2.0 * x - 1.0
    epsilon = 1.0 + math.tanh(-float(k_s))
    value = 0.5 * (-math.tanh(float(k_s) * x_mapped) - epsilon * x_mapped**3 + 1.0)
    return min(max(value, 0.0), 1.0)


def environment_py_jsrl_regulator(x: float) -> float:
    """JS curriculum from user-provided environment.py::set_JS_curriculum."""
    x = min(max(float(x), 0.0), 1.0)
    return 1.0 - x


def guide_steps(current_episode: int, total_episodes: int, max_steps: int, config: Dict[str, Any]) -> int:
    """Compute M^g using Eq. (11) and the text below it."""
    cfg = config["regulators"]
    e_g = float(cfg["cg"]) * total_episodes
    if current_episode >= e_g or e_g <= 0:
        return 0
    ratio = current_episode / e_g
    mode = str(cfg.get("jsrl_mode", "paper"))
    if mode == "environment_py":
        value = environment_py_jsrl_regulator(ratio)
    else:
        value = jsrl_regulator(ratio, float(cfg["ks"]))
    return int(max_steps * float(cfg["rho0"]) * value)


def angle_distance_bias(current_episode: int, total_episodes: int, config: Dict[str, Any]) -> Tuple[float, float]:
    """Eqs. (12)-(13): linear angle and distance tolerance regulators."""
    cfg = config["regulators"]
    e_e = max(float(cfg["ce"]) * total_episodes, 1.0)
    ratio = min(current_episode / e_e, 1.0)
    delta_alpha = float(cfg["delta_alpha_start"]) - (
        float(cfg["delta_alpha_start"]) - float(cfg["delta_alpha_expected"])
    ) * ratio
    delta_distance = float(cfg["delta_distance_start"]) - (
        float(cfg["delta_distance_start"]) - float(cfg["delta_distance_expected"])
    ) * ratio
    return delta_alpha, delta_distance


def obstacle_radius(current_episode: int, total_episodes: int, original_radius: float, config: Dict[str, Any]) -> float:
    """Eq. (14): obstacle radius growth regulator."""
    cfg = config["regulators"]
    lo = float(cfg["clo"]) * total_episodes
    hi = float(cfg["chi"]) * total_episodes
    if current_episode < lo:
        return 0.0
    if current_episode > hi or hi <= lo:
        return float(original_radius)
    return float(original_radius) * (current_episode - lo) / (hi - lo)


def obstacle_buffer(
    current_episode: int,
    total_episodes: int,
    original_radius: float,
    original_buffer: float,
    config: Dict[str, Any],
) -> float:
    """Scale obstacle safety buffer with Eq. (14)'s obstacle radius curriculum.

    The paper grows obstacle radius from zero. In this codebase obstacle buffer
    is an engineering addition; keeping a nonzero buffer when radius is zero
    leaves a point obstacle active and defeats the early obstacle curriculum.
    """
    radius = obstacle_radius(current_episode, total_episodes, original_radius, config)
    if float(original_radius) <= 0.0:
        return 0.0
    return float(original_buffer) * radius / float(original_radius)
