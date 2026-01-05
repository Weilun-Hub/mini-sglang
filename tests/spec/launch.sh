
export CUDA_VISIBLE_DEVICES="6,7"

TARGET_PATH="/media/disk1/models/Qwen3-14B"
DRAFT_PATH="/home/relay/liujiacheng06/models/Qwen3-0.6B"

python3 -m minisgl \
    --target-model-path ${TARGET_PATH} \
    --draft-model-path ${DRAFT_PATH} \
    --target-tensor-parallel-size 1 \
    --draft-tensor-parallel-size 1 \
    --cuda-graph-max-bs 4 \
    --port 1996 \
