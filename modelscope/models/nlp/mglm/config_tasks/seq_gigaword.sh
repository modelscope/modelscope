EXPERIMENT_NAME=${MODEL_TYPE}-ggw
TASK_NAME=gigaword
DATA_PATH="${DATA_ROOT}/gigaword/org_data"

TRAIN_ARGS="--epochs 10 \
            --batch-size 16 \
            --lr 3e-5 \
            --lr-decay-style linear \
            --warmup 0.06 \
            --weight-decay 1.0e-1 \
            --label-smoothing 0.1"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100"

TASK_ARGS="--src-seq-length 192 \
           --tgt-seq-length 32 \
           --min-tgt-length 0 \
           --length-penalty 0.6 \
           --no-repeat-ngram-size 3 \
           --num-beams 5 \
           --select-topk \
           --eval-batch-size 4"
