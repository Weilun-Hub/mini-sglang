from __future__ import annotations

from datetime import timedelta
from typing import Dict, NamedTuple, Tuple

from copy import deepcopy

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info, Role
from minisgl.kvcache import create_kvcache
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_hf_weight
from minisgl.utils import divide_even, init_logger, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


def create_page_table(shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.int32, device=device)


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


class Engine:
    def __init__(self, config: EngineConfig):
        self.model_config = config.model_config
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size, role=config.tp_info.role, local_rank=config.tp_info.local_rank, local_size=config.tp_info.local_size)

        assert not torch.cuda.is_initialized()
        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype

        self.role = config.tp_info.role

        self.tp_cpu_group, self.sd_group, self.sd_cpu_group, self.verify_group = self._init_communication(config)
        logger.info(f"Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 'unknown'}: About to sync memory")
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"{self.role.value} Free memory before loading model: {mem_GB(init_free_memory)}")

        # load model and determine number of pages
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            if self.role == Role.TARGET:
                logger.info_rank0(f"Creating {self.role.value} model on meta device")
                self.model = create_model(config.target_model_path, config.model_config, group=self.sd_group)
            else:
                logger.info_rank0(f"Creating {self.role.value} model on meta device")
                self.model = create_model(config.draft_model_path, config.model_config, group=self.sd_group)
        self.model.load_state_dict(self._load_weight_state_dict(config))
        logger.info(f"Rank {torch.distributed.get_rank()}: About to barrier after model loading")
        try:
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        except RuntimeError as e:
            logger.error(f"Distributed barrier failed after model loading: {e}")
            raise
        logger.info(f"Rank {torch.distributed.get_rank()}: Passed barrier after model loading")
        self.num_pages = self.dummy_page = self._determine_num_pages(init_free_memory, config)
        self.kv_cache = create_kvcache(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            device=self.device,
            dtype=self.dtype,
        )
        # NOTE: make page table 128 aligned (32 * sizeof(int32) == 128 bytes)
        self.max_seq_len = _align_up_32(min(config.max_seq_len, self.num_pages))
        self.page_table = create_page_table(  # + 1 for dummy request
            (config.max_running_req + 1, self.max_seq_len),
            device=self.device,
        )
        self.attn_backend = create_attention_backend(
            config.attention_backend,
            config.model_config,
            self.kv_cache,
            self.page_table,
        )
        self.ctx = Context(page_size=1, attn_backend=self.attn_backend)
        set_global_ctx(self.ctx)
        self.sampler = Sampler(self.device, self.model_config.vocab_size)

        logger.info(f"Rank {torch.distributed.get_rank()}: About to sync memory after initialization")
        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"{self.role.value} {config.tp_info.local_rank} Free memory after initialization: {mem_GB(post_free_memory)}")

        # cuda graph related
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
            pre_verify=True,
        )
        self.page_table[self.dummy_req.table_idx].fill_(self.dummy_page)
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=self.max_seq_len,
            vocab_size=self.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

        logger.info(f"Rank {torch.distributed.get_rank()}: About to final barrier after engine initialization")
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:

        torch.distributed.init_process_group(
            backend="nccl",
            rank=config.tp_info.rank,
            world_size=config.tp_info.size,
            timeout=timedelta(seconds=config.distributed_timeout),
            init_method=config.distributed_addr,
        )
        # Ensure all ranks have initialized the process group
        torch.cuda.synchronize(self.device)
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        tp_cpu_group = torch.distributed.new_group(backend="gloo")

        target_devices = list(range(0, config.target_tp_size))
        draft_devices = list(range(config.target_tp_size, config.tp_info.size))
        verify_devices = [target_devices[0]] + [draft_devices[0]]

        target_cpu_group = torch.distributed.new_group(target_devices, backend="gloo")
        draft_cpu_group = torch.distributed.new_group(draft_devices, backend="gloo")

        target_group = torch.distributed.new_group(target_devices)
        draft_group = torch.distributed.new_group(draft_devices)
        verify_group = torch.distributed.new_group(verify_devices)
        if config.tp_info.role == Role.TARGET:
            sd_group = target_group
            sd_cpu_group = target_cpu_group
        else:
            sd_group = draft_group
            sd_cpu_group = draft_cpu_group

        
        assert tp_cpu_group is not None
        assert sd_group is not None
        assert sd_cpu_group is not None
        assert verify_group is not None

        return tp_cpu_group, sd_group, sd_cpu_group, verify_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            return {
                k: v.to(self.dtype)
                for k, v in load_hf_weight(config.target_model_path if config.tp_info.role == Role.TARGET else config.draft_model_path, self.device).items()
            }

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2  # key + value
            * self.model_config.head_dim
            * divide_even(self.model_config.num_kv_heads, config.tp_info.local_size)
            * config.page_size
            * self.dtype.itemsize
            * self.model_config.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-tokens"
        real_kv_size = num_pages * cache_per_page
        logger.info(f"{self.role.value} {config.tp_info.local_rank} Allocating {num_pages} pages for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)

        rank = torch.distributed.get_rank()
        logger.info(f"Rank {rank}: About to all_reduce memory info")
        try:
            torch.distributed.all_reduce(
                free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.sd_cpu_group
            )
            logger.info(f"Rank {rank}: Completed all_reduce memory info")
        except RuntimeError as e:
            logger.error(f"all_reduce failed in _sync_get_memory: {e}")
            # Try to get individual rank memory info for debugging
            rank_info = f"rank={torch.distributed.get_rank() if torch.distributed.is_initialized() else 'unknown'}"
            logger.error(f"Rank info: {rank_info}, free_memory={mem_GB(free_memory)}")
            raise
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()

        for req in batch.reqs:
            req.complete_one()

        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record()
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


class DraftEngine(Engine):
    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self.gamma = None
        self.token_pool = None

        logger.info(f"world rank: {torch.distributed.get_rank()}, local rank: {config.tp_info.local_rank}, Initialized {config.tp_info.role.value} Engine")

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        if batch.phase == "prefill":
            assert torch.cuda.current_stream() == self.stream
            with self.ctx.forward_batch(batch):
                if self.graph_runner.can_use_cuda_graph(batch):
                    logits = self.graph_runner.replay(batch)
                else:
                    logits = self.model.forward()

            for req in batch.reqs:
                req.complete_one()
                # req.cached_len = req.device_len
                # req.device_len += self.gamma

            next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
            next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
            copy_done_event = torch.cuda.Event()
            copy_done_event.record()

            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
            return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
        elif batch.phase == "decode":
            return super().forward_batch(batch, args)
            # assert torch.cuda.current_stream() == self.stream
            # logger.info(f"{torch.distributed.get_rank()} self.gamma: {self.gamma}")
            # for req in batch.reqs:
            #     req.device_len -= self.gamma - 1
            # cur_batch = deepcopy(batch)
            # next_tokens_gpu = torch.empty((batch.size, self.gamma), dtype=torch.int32, device=self.device)
            # for i in range(0, self.gamma):
            #     cur_batch.out_loc = batch.out_loc[i::self.gamma]
            #     cur_batch.input_ids = batch.input_ids[i::self.gamma]

            #     self.attn_backend.prepare_metadata(cur_batch)

            #     with self.ctx.forward_batch(cur_batch):
            #         if self.graph_runner.can_use_cuda_graph(cur_batch):
            #             logits = self.graph_runner.replay(cur_batch)
            #         else:
            #             logits = self.model.forward()

            #     for req in cur_batch.reqs:
            #         req.complete_one()

            #     next_tokens_gpu[:, i] = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
            #     logger.info(f"{torch.distributed.get_rank()} decode step {i} completed: {next_tokens_gpu[:, i]}")
            
            # next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
            # copy_done_event = torch.cuda.Event()
            # copy_done_event.record()
            # return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
        # else:
        #     raise ValueError(f"Unknown batch phase: {batch.phase}")


class TargetEngine(Engine):
    def __init__(self, config: EngineConfig):
        super().__init__(config)
        logger.info(f"world rank: {torch.distributed.get_rank()}, local rank: {config.tp_info.local_rank}, Initialized {config.tp_info.role.value} Engine")

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        if batch.phase == "prefill":
            forward_output = super().forward_batch(batch, args)
            logger.info(f"{torch.distributed.get_rank()} TargetEngine forward_batch prefill completed")
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
            return forward_output
        elif batch.phase == "decode":
            assert torch.cuda.current_stream() == self.stream
            with self.ctx.forward_batch(batch):
                if self.graph_runner.can_use_cuda_graph(batch):
                    logits = self.graph_runner.replay(batch)
                else:
                    logits = self.model.forward()

            for req in batch.reqs:
                req.complete_one()

            next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
            next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
            copy_done_event = torch.cuda.Event()
            copy_done_event.record()
            return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
        else:
            raise ValueError(f"Unknown batch phase: {batch.phase}")
        
