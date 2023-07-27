#!/bin/bash

DATE=$(date +"%Y%m%d-%H%M%S")
nohup python llm_sft.py \
    --device 0,1 \
    --model_type openbuddy-llama2-13b \
    --dataset alpaca-en,alpaca-zh \
    --dataset_sample 20000 \
&> train_$DATE.out &
