EXPERIMENT_NAME=${MODEL_TYPE}-CB
TASK_NAME=cb
DATA_PATH="${DATA_ROOT}/CB"
MAX_SEQ_LEN=256

LR_SINGLE=1e-5
EPOCH_SINGLE=50
XXLARGE_EPOCH=100

TRAIN_ARGS="--lr-decay-style linear \
            --warmup 0.1 \
            --weight-decay 1.0e-1 \
            --pattern-id 3"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100"

PATTERN_IDS=(0 1 2 3)
PROMPT_IDS=(1 2 3)

BATCH_SIZE=16
