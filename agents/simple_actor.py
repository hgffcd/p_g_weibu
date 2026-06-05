"""Small NumPy actor/critic used for runnable smoke tests.

The paper uses recurrent MAPPO networks (Section IV-C.3 and Table II). This file
does not claim to reproduce rMAPPO; it verifies model I/O contracts until a full
PyTorch implementation is added.
"""

from __future__ import annotations

import numpy as np


class LinearGaussianActor:
    def __init__(self, obs_dim: int, action_dim: int, action_limit: float, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.05, size=(obs_dim, action_dim))
        self.bias = np.zeros(action_dim, dtype=float)
        self.action_limit = float(action_limit)

    def act(self, observations: np.ndarray) -> np.ndarray:
        actions = np.tanh(np.asarray(observations, dtype=float) @ self.weights + self.bias)
        return np.clip(actions * self.action_limit, -self.action_limit, self.action_limit)


class LinearValueCritic:
    def __init__(self, state_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.05, size=(state_dim,))
        self.bias = 0.0

    def value(self, flat_state: np.ndarray) -> float:
        state = np.asarray(flat_state, dtype=float).reshape(-1)
        return float(state @ self.weights + self.bias)
