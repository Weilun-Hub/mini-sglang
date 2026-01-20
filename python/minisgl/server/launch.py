from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger
from minisgl.distributed.info import Role

if TYPE_CHECKING:
    from .args import ServerArgs


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    import torch
    from minisgl.scheduler import Scheduler, TargetScheduler, DraftScheduler

    with torch.inference_mode():
        if args.tp_info.role == Role.TARGET:
            scheduler = TargetScheduler(args)
        else:
            scheduler = DraftScheduler(args)
        scheduler.sync_all_ranks()

        if args.tp_info.is_primary():
            ack_queue.put(f"{args.tp_info.role.value} scheduler is ready")
        if args.silent_output:
            logging.disable(logging.INFO)

        # debug
        # logger = init_logger(__name__)
        # if scheduler.tp_info.is_primary():
        #     print()  # for a clean newline after ^C
        #     logger.info("Scheduler exiting gracefully...")
        # scheduler.shutdown()

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if scheduler.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")

    def start_subprocess() -> None:
        import multiprocessing as mp

        from minisgl.tokenizer import tokenize_worker

        mp.set_start_method("spawn", force=True)

        target_tp_size = server_args.target_tp_size
        draft_tp_size = server_args.draft_tp_size
        target_dp_size = 1 # target dp is disable for current version
        draft_dp_size = server_args.draft_tp_size
        target_size = target_tp_size * target_dp_size
        draft_size = draft_tp_size * draft_dp_size
        world_size = target_size + draft_size
        logger.info(f"Launching {world_size} = #target(tp * dp) + #draft(tp * dp) = {target_tp_size} * {target_dp_size} + {draft_tp_size} * {draft_dp_size} scheduler subprocesses")
        
        # a multiprocessing queue to receive ack from subprocesses
        # so that we can guarantee all subprocesses are ready
        ack_queue: mp.Queue[str] = mp.Queue()

        for i in range(world_size):

            role = Role.TARGET if i < target_size else Role.DRAFT
            isTarget = role == Role.TARGET
            local_size = target_size if isTarget else draft_size
            local_rank = i if isTarget else i - target_size
            local_dp_size = target_dp_size if isTarget else draft_dp_size
            local_tp_size = target_tp_size if isTarget else draft_tp_size

            local_tp_rank = local_rank % local_dp_size
            local_dp_rank = local_rank // local_tp_size


            new_args = replace(
                server_args,
                tp_info=DistributedInfo(
                    global_rank=i,
                    global_size=world_size,
                    tp_rank=local_tp_rank,
                    tp_size=local_tp_size,
                    dp_rank=local_dp_rank,
                    dp_size=local_dp_size
                    role=role
                )
            )

            # last developed here

            logger.info(f"Starting scheduler subprocess for TP rank {i} / {world_size} : role={role}, local_rank={local_rank}")
            mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                daemon=False,
                name=f"minisgl-TP{i}-{role.value}{local_rank}-scheduler",
            ).start()

        num_tokenizers = server_args.num_tokenizer
        # DeTokenizer, only 1
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.target_model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="minisgl-detokenizer-0",
        ).start()
        for i in range(num_tokenizers):
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.target_model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"minisgl-tokenizer-{i}",
            ).start()

        # Wait for acknowledgments from all worker processes:
        # - world_size schedulers (but only primary target and draft rank send ack)
        # - num_tokenizers tokenizers
        # - 1 detokenizer
        # Total acks expected: 2 + num_tokenizers + 1 = num_tokenizers + 3
        for _ in range(num_tokenizers + 3):
            logger.info(ack_queue.get())

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
