from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set, Optional

from minisgl.core import Batch, Req
import torch


@dataclass
class DecodeManager:
    running_reqs: Set[Req] = field(default_factory=set)
    # verify_done: Optional[torch.cuda.Event] = torch.get_device_module(torch.device(f"cuda:{torch.cuda.current_device()}")).Event()

    def add_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs.update(req for req in reqs if req.can_decode())

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    @property
    def inflight_tokens(self) -> int:
        return sum(req.remain_len for req in self.running_reqs)

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        return Batch(reqs=list(self.running_reqs), phase="decode")

    @property
    def runnable(self) -> bool:
        return bool(self.running_reqs)
