#!/bin/bash

DATE=$(date +"%Y%m%d-%H%M%S")
nohup python llm_sft.py \
    --device 0 \
    --model_type llama2-7b \
    --data_sample 25000 \
&> train_$DATE.out &
