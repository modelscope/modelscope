EXPERIMENT_NAME=${MODEL_TYPE}-lambda_uni
TASK_NAME=lambda
DATA_PATH="${DATA_ROOT}/lambada_test.jsonl"
EVALUATE_ARGS="--eval-batch-size 16 \
               --seq-length 512 \
               --unidirectional"
