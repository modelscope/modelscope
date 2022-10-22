EXPERIMENT_NAME=${MODEL_TYPE}-MultiRC
TASK_NAME=multirc
DATA_PATH="${DATA_ROOT}/MultiRC"
MAX_SEQ_LEN=512

LR_SINGLE=1e-5
EPOCH_SINGLE=15
XXLARGE_EPOCH=12

TRAIN_ARGS="--batch-size 16 \
            --lr-decay-style linear \
            --warmup 0.1 \
            --weight-decay 1.0e-1 \
            --pattern-id 0"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 10000000 \
             --eval-iters 100"

PATTERN_IDS=(0 1 2)
PROMPT_IDS=(1 2 3)

BATCH_SIZE=64
