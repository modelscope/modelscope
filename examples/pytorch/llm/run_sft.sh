#!/bin/bash

DATE=$(date +"%Y%m%d-%H%M%S")
nohup python llm_sft.py \
    --device 0,1 \
    --seed 42 \
    --model_type baichuan-13B \
    --debug true \
    &> train_$DATE.out &
