PYTHONPATH=. torchrun examples/pytorch/stable_diffusion/dreambooth/finetune_stable_diffusion_dreambooth.py \
    --model 'AI-ModelScope/stable-diffusion-v2-1' \
    --model_revision 'v1.0.8' \
    --work_dir './tmp/dreambooth_diffusion' \
    --train_dataset_name 'buptwq/lora-stable-diffusion-finetune' \
    --with_prior_preservation false \
    --instance_prompt "a photo of sks dog" \
    --class_prompt "a photo of dog" \
    --class_data_dir "./tmp/class_data" \
    --num_class_images 200 \
    --resolution 512 \
    --prior_loss_weight 1.0 \
    --prompt "dog" \
    --max_epochs 150 \
    --save_ckpt_strategy 'by_epoch' \
    --logging_interval 1 \
    --train.dataloader.workers_per_gpu 0 \
    --evaluation.dataloader.workers_per_gpu 0 \
    --train.optimizer.lr 5e-6 \
    --use_model_config true
