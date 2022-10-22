EXPERIMENT_NAME=${MODEL_TYPE}-record
TASK_NAME=ReCoRD
DATA_PATH="${DATA_ROOT}/ReCoRD"
MAX_SEQ_LEN=512

LR_SINGLE=1e-5
EPOCH_SINGLE=5
XXLARGE_EPOCH=3

TRAIN_ARGS="--lr-decay-style linear \
            --warmup 0.1 \
            --weight-decay 1.0e-1 \
            --pattern-id 0"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100 \
             --deepspeed_config config_tasks/config_blocklm_10B_record.json"

PATTERN_IDS=(0)

BATCH_SIZE=64
