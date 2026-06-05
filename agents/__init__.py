"""Agent package exports with lazy PyTorch imports.

Guide-policy diagnostics should not require importing torch. RecurrentActor and
RecurrentCritic are therefore imported only when requested.
"""

from agents.apf_guide import APFGuidePolicy
from agents.encirclement_controller import PlanarEncirclementController
from agents.simple_actor import LinearGaussianActor, LinearValueCritic

__all__ = [
    "APFGuidePolicy",
    "PlanarEncirclementController",
    "LinearGaussianActor",
    "LinearValueCritic",
    "RecurrentActor",
    "RecurrentCritic",
]


def __getattr__(name: str):
    if name in {"RecurrentActor", "RecurrentCritic"}:
        from agents.torch_networks import RecurrentActor, RecurrentCritic

        return {"RecurrentActor": RecurrentActor, "RecurrentCritic": RecurrentCritic}[name]
    raise AttributeError(f"module 'agents' has no attribute {name!r}")
