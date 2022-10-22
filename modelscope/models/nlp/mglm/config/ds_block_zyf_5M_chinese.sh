#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

config_json="$script_dir/config_block_zyfbase.json"
gpt_options=" \
       --block-lm \
       --task-mask \
       --bert-prob 0.5 \
       --gap-sentence-prob 0.3 \
       --avg-block-length 3 \
       --gpt-min-ratio 0.25 \
       --block-mask-prob 0.1 \
       --short-seq-prob 0.02 \
       --experiment-name zyf-50m-chinese \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 6 \
       --hidden-size 512 \
       --num-attention-heads 8 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --save /ckpts \
       --log-interval 50 \
       --eval-interval 1000 \
       --save-interval 5000 \
       --train-iters 150000 \
       --train-data wudao \
       --resume-dataloader \
       --no-lazy-loader \
       --tokenizer-type ChineseSPTokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --lr-decay-ratio 0.1 \
       --lr-decay-iters 120000 \
       --warmup 0.04 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --fp16 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"
