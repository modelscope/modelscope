export PYTHONPATH=$PYTHONPATH:./
torchrun examples/pytorch/llama/finetune_llama.py \
    --work_dir './tmp' \
    --model 'skyline2006/llama-7b' \
    --eval_interval 100  \
    --use_lora 1 \
    --device_map 'auto' \
