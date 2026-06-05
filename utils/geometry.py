"""Geometry helpers for the EMOCA environment."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


EPS = 1e-8


def norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def safe_unit(vec: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vec)
    if length < EPS:
        return np.zeros_like(vec, dtype=float)
    return vec / length


def clip_by_component(values: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(values, -limit, limit)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def included_angle(a: np.ndarray, b: np.ndarray, center: np.ndarray) -> float:
    """Included angle around target, used by Definition 2 and Eq. (2)."""
    va = a - center
    vb = b - center
    denom = max(np.linalg.norm(va) * np.linalg.norm(vb), EPS)
    cos_value = float(np.clip(np.dot(va, vb) / denom, -1.0, 1.0))
    return float(math.acos(cos_value))


def bearing_angle(velocity: np.ndarray, relative_position: np.ndarray) -> float:
    """Angle from velocity direction to a relative position vector, Fig. 5."""
    if np.linalg.norm(velocity) < EPS or np.linalg.norm(relative_position) < EPS:
        return 0.0
    vel_angle = math.atan2(float(velocity[1]), float(velocity[0]))
    rel_angle = math.atan2(float(relative_position[1]), float(relative_position[0]))
    return wrap_angle(rel_angle - vel_angle)


def neighbor_indices(pursuer_pos: np.ndarray, target_pos: np.ndarray) -> Dict[int, Tuple[int, int]]:
    """Return left and right neighbors via target-centered polar ordering.

    This maps the paper's Definition 1 to an implementation-friendly circular sort.
    """
    rel = pursuer_pos - target_pos[None, :]
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    order = np.argsort(angles)
    result: Dict[int, Tuple[int, int]] = {}
    n = len(order)
    for rank, idx in enumerate(order):
        left = int(order[(rank - 1) % n])
        right = int(order[(rank + 1) % n])
        result[int(idx)] = (left, right)
    return result


def flatten_state(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(part, dtype=float).reshape(-1) for part in parts])
