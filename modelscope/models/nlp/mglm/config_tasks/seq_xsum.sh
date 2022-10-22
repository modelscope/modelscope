EXPERIMENT_NAME=${MODEL_TYPE}-xsum
TASK_NAME=xsum
DATA_PATH="${DATA_ROOT}/bbc-summary-data"

TRAIN_ARGS="--epochs 6 \
            --batch-size 8 \
            --lr 1e-5 \
            --lr-decay-style linear \
            --warmup 0.06 \
            --weight-decay 1.0e-1 \
            --label-smoothing 0.1"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100 \
             --eval-epoch 2"

TASK_ARGS="--src-seq-length 608 \
           --tgt-seq-length 60 \
           --min-tgt-length 10 \
           --length-penalty 1.0 \
           --no-repeat-ngram-size 3 \
           --num-beams 6 \
           --select-topk \
           --eval-batch-size 1"
