EXPERIMENT_NAME=${MODEL_TYPE}-wikitext
TASK_NAME=wikitext
DATA_PATH=/dataset/c07bd62b/wikitext-103/wiki.test.tokens
EVALUATE_ARGS="--eval-batch-size 16 \
               --seq-length 1024 \
               --overlapping-eval 256"
