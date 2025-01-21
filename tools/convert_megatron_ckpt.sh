TARGET_TENSOR_MODEL_PARALLEL_SIZE=1
ORIGIN_MODEL='damo/nlp_gpt3_text-generation_1.3B'
TARGET_DIR='./target'

torchrun --nproc_per_node $TARGET_TENSOR_MODEL_PARALLEL_SIZE tools/convert_megatron_ckpt.py \
    --model_dir $ORIGIN_MODEL \
    --target_dir $TARGET_DIR \
