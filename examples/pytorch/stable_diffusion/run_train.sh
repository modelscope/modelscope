PYTHONPATH=. torchrun examples/pytorch/stable_diffusion/finetune_stable_diffusion.py \
    --model 'damo/multi-modal_efficient-diffusion-tuning-lora' \
    --max_epochs 1 \
    --dataset_name 'controlnet_dataset_condition_fill50k'
