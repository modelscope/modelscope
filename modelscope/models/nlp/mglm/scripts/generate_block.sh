#!/bin/bash
CHECKPOINT_PATH=/glm

source $1

MPSIZE=1
MAXSEQLEN=200
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=1
TOPP=0

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="$script_dir/ds_config.json"

python3 -m torch.distributed.launch --nproc_per_node=$MPSIZE --master_port $MASTER_PORT generate_samples.py \
       --DDP-impl none \
       --model-parallel-size $MPSIZE \
       $MODEL_ARGS \
       --fp16 \
       --cache-dir cache \
       --out-seq-length $MAXSEQLEN \
       --seq-length 200 \
       --temperature $TEMP \
       --top_k $TOPK \
       --top_p $TOPP
