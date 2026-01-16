
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export CUDA_LAUNCH_BLOCKING=1

export CUDA_VISIBLE_DEVICES="4,5"

export MINISGL_DISABLE_OVERLAP_SCHEDULING=1

TARGET_PATH="/media/disk1/models/Qwen3-14B"
DRAFT_PATH="/home/relay/liujiacheng06/models/Qwen3-0.6B"

python3 -m minisgl \
    --target-model-path ${TARGET_PATH} \
    --draft-model-path ${DRAFT_PATH} \
    --target-tensor-parallel-size 1 \
    --draft-tensor-parallel-size 1 \
    --cuda-graph-max-bs 4 \
    --port 1996 \
