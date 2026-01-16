from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from .utils import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@lru_cache(maxsize=None)
def _load_radix_module() -> Module:
    return load_aot("radix", cpp_files=["radix.cpp"])


def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    # compare 2 1-D int cpu tensors for equality

    print(
        "[fast_compare_key] x:",
        x.shape, x.dtype, x.device,
        "| y:",
        y.shape, y.dtype, y.device,
        flush=True,
    )
     
    assert x.dim() == 1 and y.dim() == 1, "x/y must be 1-D"
    x = x.to(device="cpu", dtype=torch.int32)
    y = y.to(device="cpu", dtype=torch.int32)

    return _load_radix_module().fast_compare_key(x, y)
