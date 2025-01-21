DATA_PARALLEL_SIZE=4


export PYTHONPATH=$PYTHONPATH:./
torchrun --nproc_per_node $DATA_PARALLEL_SIZE examples/pytorch/llama/finetune_llama.py \
    --work_dir './tmp' \
    --model 'skyline2006/llama-7b' \
    --deepspeed 'default_offload_opt_param.json' \
    --eval_interval 100 \
    --max_epochs 3 \
