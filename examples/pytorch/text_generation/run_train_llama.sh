DATA_PARALLEL_SIZE=2

torchrun --nproc_per_node $DATA_PARALLEL_SIZE examples/pytorch/text_generation/finetune_llama.py \
    --work_dir './tmp' \
    --model 'skyline2006/llama-7b' \
    --deepspeed 'default_offload_opt_param.json'
