from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

class Role(Enum):
    TARGET = "target"
    DRAFT = "draft"

@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    global_rank: int
    global_size: int

    tp_rank: int
    tp_size: int

    dp_rank: int
    dp_size: int

    role: Role

    def __post_init__(self):
        assert 0 <= self.global_rank < self.global_size
        assert 0 <= self.tp_rank < self.tp_size
        assert 0 <= self.dp_rank < self.dp_size
        assert self.global_rank == self.tp_size * self.dp_rank + self.tp_rank + (self.role != Role.Target) * (self.global_size - self.tp_size * self.dp_size)

    def is_primary(self) -> bool:
        return (self.tp_rank == 0 and self.dp_rank == 0)


_TP_INFO: DistributedInfo | None = None

def set_tp_info(
    global_rank: int,
    global_size: int,
    tp_rank: int,
    tp_size: int,
    dp_rank: int,
    dp_size: int, 
    role: Role
) -> None:
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(
        global_rank=global_rank,
        global_size=global_size,
        tp_rank=tp_rank,
        tp_size=tp_size,
        tp_rank=dp_rank,
        dp_size=dp_size,
        role=role
    )
def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO


__all__ = ["DistributedInfo", "set_tp_info", "get_tp_info", "try_get_tp_info"]
