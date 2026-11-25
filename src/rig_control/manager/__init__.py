"""Rig manager: placement, observed status and reconciling; the lease record lives on the rig (docs/MANAGER.md)."""

from .core import Manager
from .errors import ManagerError

__all__ = ["Manager", "ManagerError"]
