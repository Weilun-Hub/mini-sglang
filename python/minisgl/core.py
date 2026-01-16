from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle
    pre_verify: bool
    cur_acc_tokens: int
    num_acc_tokens: list[int]

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.pre_verify = True
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        self.input_ids = torch.cat([self.input_ids, next_token])

    def can_decode(self) -> bool:
        return self.remain_len > 0
    
    def rollback(self, n: int) -> None:
        self.device_len -= n
        self.cached_len -= n
        self.input_ids = self.input_ids[:-n]

    def __repr__(self) -> str:
        return (
            f"uid={self.uid}, "
            f"input_ids={self.input_ids}, pre_verfiy={self.pre_verify}, "
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    padded_reqs: List[Req] = field(init=False)  # may contain some dummy reqs for padding
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)

    @property
    def is_prefill(self) -> bool:
        if len(self.reqs) > 0:
            flag = False
            for req in self.reqs:
                cur_num_new_tokens = req.device_len - req.cached_len
                if cur_num_new_tokens > 1:
                    flag = True
                    break
            return flag
        else:
            return self.phase == "prefill"
    
    @property
    def is_verify(self) -> bool:
        if len(self.reqs) > 0:
            flag = False
            for req in self.reqs:
                if req.cached_len > 0:
                    flag = True
                    break
            return flag
        else:
            return False

    @property
    def is_decode(self) -> bool:
        if len(self.reqs) > 0:
            flag = True
            for req in self.reqs:
                cur_num_new_tokens = req.device_len - req.cached_len
                if cur_num_new_tokens > 1:
                    flag = False
                    break
            return flag
        else:
            return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)


@dataclass
class Context:
    page_size: int
    attn_backend: BaseAttnBackend
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
