#!/bin/bash

python llm_infer.py \
    --device 0,1 \
    --model_type openbuddy-llama2-13b \
    --ckpt_path "runs/openbuddy-llama2-13b/vx_xxx/output_best/pytorch_model.bin" \
    --eval_human true
