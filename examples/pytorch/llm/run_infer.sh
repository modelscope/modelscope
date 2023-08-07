CUDA_VISIBLE_DEVICES=0,1 \
python llm_infer.py \
    --model_type qwen-7b \
    --ckpt_path "runs/qwen-7b/vx_xxx/output_best/pytorch_model.bin" \
    --eval_human true
