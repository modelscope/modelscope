export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="../lora/training_scripts/pics"
export OUTPUT_DIR="../lora/training_scripts/output_dir"

PYTHONPATH=.:../diffusers/src:../lora accelerate launch main.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --train_text_encoder \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="naruto of boy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --learning_rate_text=1e-5 \
  --lr_scheduler="linear" \
  --lr_warmup_steps=0 \
  --max_train_steps=5000 \
  --save_steps=10 \
  --max_grad_norm=1

