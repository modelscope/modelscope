EXPERIMENT_NAME=${MODEL_TYPE}-lm
TASK_NAME=language_model
DATA_PATH=${DATA_ROOT}/bert-large-test.txt
EVALUATE_ARGS="--eval-batch-size 16 \
               --seq-length 512 \
               --overlapping-eval 256"
