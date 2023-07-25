#!/bin/bash

python llm_infer.py \
    --device 0 \
    --model_type llama2-7b \
    --ckpt_path "runs/llama2-7b/vx_xxx/output_best/pytorch_model.bin" \
    --eval_human true
