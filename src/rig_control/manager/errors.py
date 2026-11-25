from __future__ import annotations

from typing import Any

# Rig error kinds -> Gateway HTTP status.
KIND_STATUS = {"invalid": 400, "ownership": 403, "not_found": 404, "conflict": 409,
               "not_controller": 409, "unfit": 422, "capacity": 503, "state": 500}


class ManagerError(Exception):
    def __init__(self, status: int, message: str, **extra: Any):
        super().__init__(message)
        self.status = status
        self.extra = extra
