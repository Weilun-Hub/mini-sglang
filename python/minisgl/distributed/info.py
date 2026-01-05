from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

class Role(Enum):
    TARGET = "target"
    DRAFT = "draft"

@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    rank: int
    size: int

    role: Role
    local_rank: int
    local_size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size
        assert 0 <= self.local_rank < self.local_size

    def is_primary(self) -> bool:
        return self.local_rank == 0


_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int, role: Role, local_rank: int, local_size: int) -> None:
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size, role, local_rank, local_size)
def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO


__all__ = ["DistributedInfo", "set_tp_info", "get_tp_info", "try_get_tp_info"]
