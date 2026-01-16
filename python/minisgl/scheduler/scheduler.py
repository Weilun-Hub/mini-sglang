from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
import torch.nn.functional as F
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    BaseBackendMsg,
    BatchBackendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger, is_sm90_supported
from transformers import AutoTokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

from minisgl.distributed.info import Role, get_tp_info

from minisgl.core import Req

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput

from minisgl.engine import VerifyOutput

import flashinfer.sampling as sampling

torch.manual_seed(42)
torch.cuda.manual_seed(42)

logger = init_logger(__name__)


def _make_2d_indices(table_2d: torch.Tensor, ranges: List[Tuple[int, int, int]]) -> torch.Tensor:
    """
    Return the 1D indices for the given 2D table and ranges.

    Example: The underlying indices of a 2D table (3, 4) are:
        [[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]]
    For ranges [(0, 1, 3), (2, 0, 2)], the returned indices are [1, 2, 8, 9].

    Args:
        table_2d (torch.Tensor): The 2D table tensor.
        ranges (List[Tuple[int, int, int]]): A list of tuples (entry, begin, end),
            where `entry` is the row index in the 2D table, and `begin` and `end`
            specify the range of column indices to include.
    Returns:
        torch.Tensor: A 1D tensor of indices.
    """
    assert table_2d.dim() == 2 and table_2d.is_contiguous()
    STRIDE = table_2d.stride(0)
    needed_size = sum(end - begin for _, begin, end in ranges)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for entry, begin, end in ranges:
        length = end - begin
        offset += length
        torch.arange(
            begin + entry * STRIDE,
            end + entry * STRIDE,
            dtype=torch.int32,
            out=indices_host[offset - length : offset],
        )
    return indices_host.to(table_2d.device, non_blocking=True)


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    load_indices: torch.Tensor
    write_indices: torch.Tensor


ForwardData: TypeAlias = "Tuple[ForwardInput, Union[ForwardOutput, VerifyOutput]]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine, DraftEngine, TargetEngine
        if config.tp_info.role == Role.TARGET:
            self.engine = TargetEngine(config)
        else:
            self.engine = DraftEngine(config)
        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(self.device, self.engine.num_pages, config.cache_type)
        self.decode_manager = DecodeManager()
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        self.tp_info = config.tp_info
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = AutoTokenizer.from_pretrained(config.target_model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.page_table = self.engine.page_table
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens

    def _process_last_data(
        self, last_data: ForwardData | None, ongoing_data: ForwardData | None
    ) -> None:
        if last_data is None:
            return
        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        # logger.info(f"{torch.distributed.get_rank()} _process_last_data before copy_done.synchronize()")
        copy_done.synchronize()
        # logger.info(f"{torch.distributed.get_rank()} _process_last_data after copy_done.synchronize()")
        reply = BatchTokenizerMsg(data=[])

        max_seq_len = self.engine.max_seq_len
        for i, req in enumerate(batch.reqs):
            if req in self.finished_reqs or isinstance(req, ChunkedReq):
                continue

            # logger.info(f"{torch.distributed.get_rank()} Processing results for batch with req {i}: next_token_id {next_tokens_cpu[i]}")

            next_token_id = next_tokens_cpu[i]
            # logger.info(f"{torch.distributed.get_rank()} before req.append_host req[0]: {req}")
            req.append_host(next_token_id.unsqueeze(0))
            # logger.info(f"{torch.distributed.get_rank()} after req.append_host req[0]: {req}")
            next_token = int(next_token_id.item())
            finished = req.remain_len <= 0
            if not req.sampling_params.ignore_eos:
                finished |= next_token == self.eos_token_id
            if req.device_len >= max_seq_len - 1:
                finished = True
                logger.warning_rank0(f"Request {req.uid} reached {max_seq_len = }, dropped.")
            reply.data.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

            logger.info(f"{torch.distributed.get_rank()} appended next_token {next_token}, finished: {finished}")
            # free resources if the req is finished and not ongoing
            if finished:
                self.finished_reqs.add(req)
                self.decode_manager.remove_req(req)
                logger.debug_rank0("Request %s is finished", req)

        # free resources for finished but not ongoing reqs
        ongoing_reqs = ongoing_data[0].batch.reqs if ongoing_data else []
        for req in self.finished_reqs.difference(ongoing_reqs):
            self.table_manager.free(req.table_idx)
            self.cache_manager.free_and_cache_finished_req(
                req.cache_handle,
                req.input_ids[: req.cached_len],
                self.page_table[req.table_idx, : req.cached_len],
            )

        # keep only ongoing reqs in the finished set
        self.finished_reqs.intersection_update(ongoing_reqs)
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            if input_len >= max_seq_len:
                return logger.warning_rank0(
                    f"Input sequence len {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            max_output_len = max_seq_len - input_len
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        # logger.info(f"{torch.distributed.get_rank()} _prepare_batch {batch.reqs[0]}, slot {self.token_pool[batch.reqs[0].table_idx][:30]}")
        needed_size = sum(r.extend_len for r in batch.reqs)
        batch.out_loc = self.cache_manager.allocate(needed_size)
        # NOTE: Pad the batch if needed
        if padding_size := self.engine.graph_runner.pad_batch(batch):
            batch.out_loc = F.pad(batch.out_loc, (0, padding_size), value=self.engine.dummy_page)
        # NOTE: prepare 2d indices for token ids loading and writing
        load_indices = _make_2d_indices(
            self.token_pool, [(r.table_idx, r.cached_len, r.device_len) for r in batch.padded_reqs]
        )
        write_indices = _make_2d_indices(
            self.token_pool, [(r.table_idx, r.device_len, r.device_len + 1) for r in batch.reqs]
        )
        # NOTE: write out_loc to page_table before `prepare_metadata`
        self.page_table.view(-1)[load_indices] = batch.out_loc
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            load_indices=load_indices,
            write_indices=write_indices,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )

        # if batch is None:
        #     return None
        
        # with torch.cuda.stream(self.stream):
        #     if  batch.phase == "decode":
        #         self.stream.wait_event(self.decode_manager.verify_done)
        #     return self._prepare_batch(batch)

        if batch.phase == "decode":
            self.decode_manager.verify_done.synchronize()
        
        return self._prepare_batch(batch) if batch else None

    def _load_token_ids(self, input: ForwardInput) -> None:
        input.batch.input_ids = self.token_pool.view(-1)[input.load_indices]

    def _write_token_ids(self, input: ForwardInput, output: ForwardOutput) -> None:
        self.token_pool.view(-1)[input.write_indices] = output.next_tokens_gpu

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        self._load_token_ids(forward_input)
        batch, sample_args = forward_input.batch, forward_input.sample_args
        logger.info(f"{torch.distributed.get_rank()} Starting forward for {batch.phase}: {batch.input_ids}")
        forward_output = self.engine.forward_batch(batch, sample_args)
        self._write_token_ids(forward_input, forward_output)
        self.decode_manager.add_reqs(forward_input.batch.reqs)
        logger.info(f"{torch.distributed.get_rank()} forward_batch {forward_input.batch.phase} completed")
        return forward_output

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data, ongoing_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data, None)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                # while True:
                #     self.normal_loop()

                # while True:
                #     torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
                #     self.normal_loop()

                for i in range(10):
                    logger.info(f"{torch.distributed.get_rank()} ========================= step {i} =========================")
                    # logger.info(f"{torch.distributed.get_rank()} ========================= step {i} =========================")
                    # logger.info(f"{torch.distributed.get_rank()} ========================= step {i} =========================")
                    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
                    self.normal_loop()
            # import pdb; pdb.set_trace()
            
            import time; time.sleep(3600)

        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
                data = self.overlap_loop(data)
            
            # for i in range(13):
            #     logger.info(f"{torch.distributed.get_rank()} +------------------------- step {i} -------------------------+")
            #     torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
            #     data = self.overlap_loop(data)

            # import time; time.sleep(3600)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def rollback(self, req: Req, n: int):
        self.cache_manager.rollback(
            self.page_table[req.table_idx, req.cached_len - n : req.cached_len]
        )
        # logger.info(f"{torch.distributed.get_rank()} original page table: {self.page_table[req.table_idx, :30]}")
        # logger.info(f"{torch.distributed.get_rank()} page table to be freeed: {self.page_table[req.table_idx, req.cached_len - n : req.cached_len]}")
        req.rollback(n)


class TargetScheduler(Scheduler):
    def __init__(self, config: SchedulerConfig):
        assert config.tp_info.role == Role.TARGET
        super().__init__(config)
        self.gamma = 3

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        if forward_input.batch.phase == "prefill":
            # return super()._forward(forward_input)
            # torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
            # return output
            # return super()._forward(forward_input)
            output = super()._forward(forward_input)
            # torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
            _, next_tokens_cpu, copy_done = output
            copy_done.synchronize()
            # max_seq_len = self.engine.max_seq_len
            for i, req in enumerate(forward_input.batch.reqs):
                if req in self.finished_reqs or isinstance(req, ChunkedReq):
                    continue

                next_token_id = next_tokens_cpu[i]
                # logger.info(f"{torch.distributed.get_rank()} before req.append_host req[0]: {req}")
                req.append_host(next_token_id.unsqueeze(0))
                # logger.info(f"{torch.distributed.get_rank()} after req.append_host req[0]: {req}")
                # next_token = int(next_token_id.item())
            return output
        elif forward_input.batch.phase == "decode":
            self._load_token_ids(forward_input)
            batch, sample_args = forward_input.batch, forward_input.sample_args
            # logger.info(f"{torch.distributed.get_rank()} Starting forward for {batch.phase}: {batch.input_ids}")
            assert torch.cuda.current_stream() == self.engine.stream
            with self.engine.ctx.forward_batch(batch):
                if self.engine.graph_runner.can_use_cuda_graph(batch):
                    logits = self.engine.graph_runner.replay(batch)
                else:
                    logits = self.engine.model.forward()

            for req in batch.reqs:
                req.cached_len = req.device_len

            # +--------------------------------------- verify ---------------------------------------+
            local_rank = self.tp_info.local_rank
            rank = self.tp_info.rank
            verify_res = torch.zeros((4, len(batch.reqs)), dtype=torch.int64, device="cuda")
            if local_rank == 0:
                
                num_to_be_verified_tokens = sum([1 if req.pre_verify else self.gamma for req in batch.reqs])
                num_next_round_input = self.gamma * len(batch.reqs)
                msg = torch.zeros(num_to_be_verified_tokens + num_next_round_input, dtype=torch.int64, device="cuda")
                src_rank = self.tp_info.size - self.tp_info.local_size # draft rank 0
                torch.distributed.broadcast(msg, src=src_rank, group=self.engine.verify_group)
                # logger.info(f"{torch.distributed.get_rank()} verify group finish receive msg")
                to_be_verified_tokens = msg[:num_to_be_verified_tokens].cpu().numpy().tolist()
                next_round_input = msg[num_to_be_verified_tokens:].cpu().numpy().tolist()

                logger.info(f"{torch.distributed.get_rank()} to_be_verified_tokens {to_be_verified_tokens}")
                logger.info(f"{torch.distributed.get_rank()} next_round_input {next_round_input}")
                logger.info(f"{torch.distributed.get_rank()} logits.shape {target_logits.shape}")

                r = torch.rand(num_to_be_verified_tokens, device="cuda")
                
                target_logits = torch.zeros(logits.shape, device=logits.device, dtype=logits.dtype)
                for i in range(logits.shape[0]):
                    target_logits[i : i + 1] = sampling.softmax(logits[i : i + 1], sample_args.temperatures, enable_pdl=is_sm90_supported())

                target_prob = target_logits.gather(dim=1, index=msg[:num_to_be_verified_tokens].unsqueeze(1)).squeeze(1)
                judge = (r <= target_prob).tolist()

                logger.info(f"{torch.distributed.get_rank()} r {r}, target_prob {target_prob}, judge {judge}")

                original_tokens = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.int32)
                for i in range(len(original_tokens)):
                    original_tokens[i] = self.engine.sampler.sample(logits[i : i + 1], sample_args)

                logits.scatter_(1, msg[:num_to_be_verified_tokens].unsqueeze(1), float('-inf'))

                # TODO: directly using logits for [n, vocab_size] returns wrong results
                # revised_tokens = self.engine.sampler.sample(logits, sample_args)
                revised_tokens = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.int32)
                for i in range(len(revised_tokens)):
                    revised_tokens[i] = self.engine.sampler.sample(logits[i : i + 1], sample_args)

                acc, rollout, revise_token, finish = [], [], [], []
                v_idx = 0
                for i, req in enumerate(batch.reqs):
                    if req.pre_verify:
                        acc.append(judge[v_idx])
                        rollout.append(0 if judge[v_idx] else self.gamma)
                        revise_token.append(revised_tokens[v_idx])

                        if judge[v_idx]:
                            req.cur_acc_tokens += 1
                            is_finished = not req.sampling_params.ignore_eos
                            is_finished &= to_be_verified_tokens[v_idx] == self.eos_token_id
                            is_finished |= req.device_len >= self.engine.max_seq_len - 1
                            finish.append(is_finished)
                        else:
                            req.num_acc_tokens.append(req.cur_acc_tokens + 1)
                            req.cur_acc_tokens = 0

                            is_finished = not req.sampling_params.ignore_eos
                            is_finished &= revise_token[-1] == self.eos_token_id
                            is_finished |= req.device_len >= self.engine.max_seq_len - 1
                            finish.append(is_finished)
                    else:
                        n = self.gamma
                        is_finished = False
                        for j in range(v_idx, v_idx + self.gamma):
                            if ((not req.sampling_params.ignore_eos) and judge[j]) and (to_be_verified_tokens[j] == self.eos_token_id):
                                is_finished = True
                            
                            if not judge[j]:
                                n = j - v_idx
                                break
                        
                        acc.append(n == self.gamma)
                        rollout.append(self.gamma - n)
                        revise_token.append(revised_tokens[v_idx + n] if n < self.gamma else -1)

                        is_finished |= req.device_len >= self.engine.max_seq_len - min(n + 1, self.gamma)
                        finish.append(is_finished)

                        if n == self.gamma:
                            req.cur_acc_tokens += n
                        else:
                            req.num_acc_tokens.append(req.cur_acc_tokens + n + 1)
                            req.cur_acc_tokens = 0
                    
                    v_idx += 1 if req.pre_verify else self.gamma
                verify_res = torch.tensor([acc, rollout, revise_token, finish], dtype=torch.int64, device="cuda")

            torch.distributed.broadcast(verify_res, src=0)
            # logger.info(f"{torch.distributed.get_rank()} verify group finish broadcast verify_res")
            acc, rollout, revise_token, finish = verify_res.tolist()

            for idx, req in enumerate(batch.reqs):
                if req.pre_verify:
                    if acc[idx]:
                        req.pre_verify = False
                        _tokens = torch.as_tensor(next_round_input[self.gamma * idx : self.gamma * (idx + 1)], dtype=self.token_pool.dtype, device=self.token_pool.device)

                        # print("[DEBUG] forward_input.write_indices",forward_input.write_indices, forward_input.write_indices.dtype)
                        self.token_pool.view(-1)[forward_input.write_indices[idx] : forward_input.write_indices[idx] + self.gamma] = _tokens
                        
                        req.append_host(torch.tensor(next_round_input[self.gamma * idx : self.gamma * (idx + 1)]))
                        req.device_len += self.gamma
                    else:
                        req.pre_verify = True
                        req.append_host(torch.tensor(revise_token[idx : idx + 1]))
                        _tokens = torch.as_tensor(revise_token[idx], dtype=self.token_pool.dtype, device=self.token_pool.device)
                        self.token_pool.view(-1)[forward_input.write_indices[idx]] = _tokens
                        req.device_len += 1
                else:

                    if acc[idx]:
                        req.pre_verify = False
                        req.append_host(torch.tensor(next_round_input[self.gamma * idx : self.gamma * (idx + 1)]))
                        _tokens = torch.as_tensor(next_round_input[self.gamma * idx : self.gamma * (idx + 1)], dtype=self.token_pool.dtype, device=self.token_pool.device)
                        self.token_pool.view(-1)[forward_input.write_indices[idx] : forward_input.write_indices[idx] + self.gamma] = _tokens
                        req.device_len += self.gamma
                    else:
                        req.pre_verify = True
                        if rollout[idx] > 1:
                            self.rollback(req, rollout[idx] - 1)

                        req.append_host(torch.tensor(revise_token[idx : idx + 1]))
                        self.token_pool.view(-1)[forward_input.write_indices[idx] - rollout[idx] + 1: forward_input.write_indices[idx] - rollout[idx] + 2] = torch.as_tensor(revise_token[idx : idx + 1], dtype=self.token_pool.dtype, device=self.token_pool.device)

                        req.device_len += 1

            verify_done_event = torch.cuda.Event()
            verify_done_event.record()
            return VerifyOutput(verify_res, next_round_input, verify_done_event)
        
            
    def _process_last_data(
        self, last_data: ForwardData | None, ongoing_data: ForwardData | None
    ) -> None:
        if last_data is None:
            return
        batch = last_data[0].batch
        sample_args = last_data[0].sample_args

        reply = BatchTokenizerMsg(data=[])

        if batch.phase == "prefill":
            _, next_tokens_cpu, copy_done = last_data[1]
            # logger.info(f"{torch.distributed.get_rank()} _process_last_data before copy_done.synchronize()")
            # copy_done.synchronize()
            # logger.info(f"{torch.distributed.get_rank()} _process_last_data after copy_done.synchronize()")

            max_seq_len = self.engine.max_seq_len
            for i, req in enumerate(batch.reqs):
                if req in self.finished_reqs or isinstance(req, ChunkedReq):
                    continue

                # logger.info(f"{torch.distributed.get_rank()} Processing results for batch with req {i}: next_token_id {next_tokens_cpu[i]}")

                next_token_id = next_tokens_cpu[i]
                # logger.info(f"{torch.distributed.get_rank()} before req.append_host req[0]: {req}")
                # req.append_host(next_token_id.unsqueeze(0))
                # logger.info(f"{torch.distributed.get_rank()} after req.append_host req[0]: {req}")
                next_token = int(next_token_id.item())
                finished = req.remain_len <= 0
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                if req.device_len >= max_seq_len - 1:
                    finished = True
                    logger.warning_rank0(f"Request {req.uid} reached {max_seq_len = }, dropped.")
                reply.data.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # logger.info(f"{torch.distributed.get_rank()} appended next_token {next_token}, finished: {finished}")
                # free resources if the req is finished and not ongoing
                if finished:
                    self.finished_reqs.add(req)
                    self.decode_manager.remove_req(req)
                    logger.debug_rank0("Request %s is finished", req)
        elif batch.phase == "decode":
            
            verify_res, next_round_input, verify_done = last_data[1]
            verify_done.synchronize()

            acc, rollout, revise_token, finish = verify_res.tolist()

            for idx, req in enumerate(batch.reqs):
                
                reply.data.append(DetokenizeMsg(uid=req.uid, next_token=int(req.input_ids[-1].item()), finished=finish[idx]))
                if finish[idx]:
                    logger.info(f"{torch.distributed.get_rank()} finished req: {req}")
                    self.finished_reqs.add(req)
                    self.decode_manager.remove_req(req)
                    logger.debug_rank0("Request %s is finished", req)

        for i in range(len(batch.reqs)):
            logger.info(f"{torch.distributed.get_rank()} after process_last_data req[{i}]: {batch.reqs[i]}  token pool {self.token_pool[req.table_idx][:30]}")

        # free resources for finished but not ongoing reqs
        ongoing_reqs = ongoing_data[0].batch.reqs if ongoing_data else []
        for req in self.finished_reqs.difference(ongoing_reqs):
            self.table_manager.free(req.table_idx)
            self.cache_manager.free_and_cache_finished_req(
                req.cache_handle,
                req.input_ids[: req.cached_len].to(torch.int32),
                self.page_table[req.table_idx, : req.cached_len],
            )

        # keep only ongoing reqs in the finished set
        self.finished_reqs.intersection_update(ongoing_reqs)
        
        self.decode_manager.verify_done.record(torch.cuda.current_stream())
        
        self.send_result(reply)

class DraftScheduler(Scheduler):
    def __init__(self, config: SchedulerConfig):
        assert config.tp_info.role == Role.DRAFT
        super().__init__(config)
        self.gamma = 3
        self.engine.gamma = self.gamma
        self.engine.token_pool = self.token_pool
    
    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            if input_len >= max_seq_len:
                return logger.warning_rank0(
                    f"Input sequence len {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            max_output_len = max_seq_len - input_len
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _process_last_data(
        self, last_data: ForwardData | None, ongoing_data: ForwardData | None
    ) -> None:
        if last_data is None:
            return
        batch = last_data[0].batch
        reply = BatchTokenizerMsg(data=[])

        if batch.phase == "prefill":
            _, next_tokens_cpu, copy_done = last_data[1]
            # copy_done.synchronize()
            max_seq_len = self.engine.max_seq_len
            for i, req in enumerate(batch.reqs):
                if req in self.finished_reqs or isinstance(req, ChunkedReq):
                    continue

                next_token_id = next_tokens_cpu[i]
                # logger.info(f"{torch.distributed.get_rank()} before req.append_host req[0]: {req}")
                # req.append_host(next_token_id.unsqueeze(0))
                # logger.info(f"{torch.distributed.get_rank()} after req.append_host req[0]: {req}")
                next_token = int(next_token_id.item())
                finished = req.remain_len <= 0
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                if req.device_len >= max_seq_len - 1:
                    finished = True
                    logger.warning_rank0(f"Request {req.uid} reached {max_seq_len = }, dropped.")
                reply.data.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # free resources if the req is finished and not ongoing
                if finished:
                    self.finished_reqs.add(req)
                    self.decode_manager.remove_req(req)
                    logger.debug_rank0("Request %s is finished", req)
        
        elif batch.phase == "decode": # verify
            verify_res, next_round_input, verify_done = last_data[1]
            verify_done.synchronize()
            
            acc, rollout, revise_token, finish = verify_res.tolist()
            # for idx, req in enumerate(batch.reqs):
            #     if req in self.finished_reqs or isinstance(req, ChunkedReq):
            #         continue
            #     if finish[idx]:
            #         continue
                
            #     if req.pre_verify:
            #         if acc[idx]:
            #             req.pre_verify = False
            #         else:
            #             req.pre_verify = True
            #             self.rollback(req, self.gamma - 1)
            #             req.input_ids[-1] = revise_token[idx]
            #             self.token_pool[req.table_idx][len(req.input_ids) - 1] = revise_token[idx]
            #     else:
            #         if acc[idx]:
            #             req.pre_verify = False
            #         else:
            #             req.pre_verify = True
            #             self.rollback(req, self.gamma - 1)
            #             if rollout[idx] > 1:
            #                 self.rollback(req, rollout[idx] - 1)
                        
            #             req.input_ids[-1] = revise_token[idx]
            #             self.token_pool[req.table_idx][len(req.input_ids) - 1] = revise_token[idx]
            for idx, req in enumerate(batch.reqs):
                if req in self.finished_reqs or isinstance(req, ChunkedReq):
                    continue
                reply.data.append(DetokenizeMsg(uid=req.uid, next_token=req.input_ids[-1], finished=finish[idx]))
                if finish[idx]:
                    logger.info(f"{torch.distributed.get_rank()} finished req: {req}")
                    self.finished_reqs.add(req)
                    self.decode_manager.remove_req(req)
                    continue
        
        for i in range(len(batch.reqs)):
            logger.info(f"{torch.distributed.get_rank()} after process_last_data req[{i}]: {batch.reqs[i]} token_pool {self.token_pool[req.table_idx][:30]}")

        ongoing_reqs = ongoing_data[0].batch.reqs if ongoing_data else []
        for req in self.finished_reqs.difference(ongoing_reqs):
            self.table_manager.free(req.table_idx)
            self.cache_manager.free_and_cache_finished_req(
                req.cache_handle,
                req.input_ids[: req.cached_len].to(torch.int32),
                self.page_table[req.table_idx, : req.cached_len],
            )

        self.finished_reqs.intersection_update(ongoing_reqs)
        self.send_result(reply)

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        if forward_input.batch.phase == "prefill":
            # return super()._forward(forward_input)
            output = super()._forward(forward_input)
            # torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
            _, next_tokens_cpu, copy_done = output
            copy_done.synchronize()
            # max_seq_len = self.engine.max_seq_len
            for i, req in enumerate(forward_input.batch.reqs):
                if req in self.finished_reqs or isinstance(req, ChunkedReq):
                    continue

                next_token_id = next_tokens_cpu[i]
                # logger.info(f"{torch.distributed.get_rank()} before req.append_host req[0]: {req}")
                req.append_host(next_token_id.unsqueeze(0))
                # logger.info(f"{torch.distributed.get_rank()} after req.append_host req[0]: {req}")
                # next_token = int(next_token_id.item())
            return output
        elif forward_input.batch.phase == "decode":
            for i in range(self.gamma):
                self._load_token_ids(forward_input)
                batch, sample_args = forward_input.batch, forward_input.sample_args
                # logger.info(f"{torch.distributed.get_rank()} Starting forward for {batch.phase}: {batch.input_ids}")
                forward_output = self.engine.forward_batch(batch, sample_args)
                self._write_token_ids(forward_input, forward_output)
                forward_output.copy_done_event.synchronize()
                next_tokens_cpu = forward_output.next_tokens_cpu
                for idx_req, req in enumerate(batch.reqs):
                    req.append_host(next_tokens_cpu[idx_req].unsqueeze(0))
                if i < self.gamma - 1:
                    forward_input = self._prepare_batch(batch)

            # for i in range(len(forward_input.batch.reqs)):
            #     logger.info(f"{torch.distributed.get_rank()} after draft req[{i}]: {forward_input.batch.reqs[i]}")
            

            local_rank = get_tp_info().local_rank
            rank = get_tp_info().rank
            if local_rank == 0:
                to_be_verified_tokens = []
                next_round_input = []
                for req in batch.reqs:
                    if req.pre_verify:
                        to_be_verified_tokens.append(req.input_ids[- self.gamma].numpy().tolist())
                    else:
                        to_be_verified_tokens.extend(req.input_ids[-2 * self.gamma + 1 : - self.gamma + 1].numpy().tolist())
                    next_round_input.extend(req.input_ids[- self.gamma :].numpy().tolist())
                # logger.info(f"{torch.distributed.get_rank()} to_be_verified_tokens: {to_be_verified_tokens}")
                # logger.info(f"{torch.distributed.get_rank()} next_round_input: {next_round_input}")
                msg = torch.tensor(to_be_verified_tokens + next_round_input, dtype=torch.int64, device="cuda")
                torch.distributed.broadcast(msg, src=rank, group=self.engine.verify_group)
            # logger.info(f"{torch.distributed.get_rank()} draft group finish broadcast msg")
            verify_res = torch.zeros((4, len(batch.reqs)), dtype=torch.int64, device="cuda")
            torch.distributed.broadcast(verify_res, src=0)
            # logger.info(f"{torch.distributed.get_rank()} draft group finish receive verify_res")

            acc, rollout, revise_token, finish = verify_res.tolist()
            for idx, req in enumerate(batch.reqs):
                if req in self.finished_reqs or isinstance(req, ChunkedReq):
                    continue
                if finish[idx]:
                    continue
                
                if req.pre_verify:
                    if acc[idx]:
                        req.pre_verify = False
                    else:
                        req.pre_verify = True
                        self.rollback(req, self.gamma - 1)
                        req.input_ids[-1] = revise_token[idx]
                        self.token_pool[req.table_idx][len(req.input_ids) - 1] = revise_token[idx]
                else:
                    if acc[idx]:
                        req.pre_verify = False
                    else:
                        req.pre_verify = True
                        self.rollback(req, self.gamma - 1)
                        if rollout[idx] > 1:
                            self.rollback(req, rollout[idx] - 1)
                        
                        req.input_ids[-1] = revise_token[idx]
                        self.token_pool[req.table_idx][len(req.input_ids) - 1] = revise_token[idx]

            verify_done_event = torch.cuda.Event()
            verify_done_event.record()
            return VerifyOutput(verify_res, next_round_input, verify_done_event)
