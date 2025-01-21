CUDA_VISIBLE_DEVICES=0,1 \
python llm_infer.py \
    --model_type polylm-13b \
    --ckpt_path "runs/polylm-13b/v0-20230802-172425/output_best/pytorch_model.bin" \
    --eval_human true
