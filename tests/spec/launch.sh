
export CUDA_VISIBLE_DEVICES="6,7"

TARGET_PATH=""
DRAFT_PATH=""

python3 -m minisgl \
    --target-model-path ${TARGET_PATH} \
    --draft-model-path ${DRAFT_PATH} \
    --target-tensor-parallel-size 1 \
    --draft-tensor-parallel-size 1 \
