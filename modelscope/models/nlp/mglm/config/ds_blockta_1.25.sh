#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

config_json="$script_dir/config_blockta_large.json"
gpt_options=" \
       --block-lm \
       --task-mask \
       --bert-prob 0.5 \
       --gap-sentence-prob 0.3 \
       --avg-block-length 3 \
       --gpt-min-ratio 0.25 \
       --block-mask-prob 0.1 \
       --short-seq-prob 0.02 \
       --non-sentence-start 0.02 \
       --experiment-name blocklm-roberta-1.25-blank \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 24 \
       --hidden-size 1152 \
       --num-attention-heads 18 \
       --seq-length 512 \
       --max-position-embeddings 1024 \
       --attention-scale 8.0 \
       --save /dataset/fd5061f6/english_data/checkpoints \
       --load /dataset/fd5061f6/english_data/checkpoints/blocklm-roberta-1.25-blank04-22-14-01 \
       --save-interval 2500 \
       --train-iters 500000 \
       --resume-dataloader \
       --filter-english \
       --train-data wikibook cc-news openwebtext stories \
       --loader-scatter 8 \
       --no-lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --tokenizer-model-type roberta \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style linear \
       --lr-decay-iters 500000 \
       --lr-decay-ratio 0.025 \
       --warmup .06 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --fp16 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"
