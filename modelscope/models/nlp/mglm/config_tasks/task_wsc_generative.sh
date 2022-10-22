TASK_NAME=wsc
EXPERIMENT_NAME=${MODEL_TYPE}-${TASK_NAME}_generative
DATA_PATH="${DATA_ROOT}/WSC"
MAX_SEQ_LEN=128

LR_SINGLE=1e-5
EPOCH_SINGLE=50
XXLARGE_EPOCH=100

TRAIN_ARGS="--lr-decay-style linear \
            --warmup 0.1 \
            --weight-decay 0.1"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100"

BATCH_SIZE=16
