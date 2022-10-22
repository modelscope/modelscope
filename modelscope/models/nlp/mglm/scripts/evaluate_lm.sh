DATA_ROOT=/dataset/c07bd62b
CHECKPOINT_PATH="/dataset/c07bd62b/checkpoints"

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port $MASTER_PORT"
DATESTR=$(date +"%m-%d-%H-%M")

source $1    # Model
source $2    # Task

python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_glm.py \
       --deepspeed \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --valid-data ${DATA_PATH} \
       --save ${CHECKPOINT_PATH} \
       --checkpoint-activations \
       --fp16 \
       --overwrite \
       $MODEL_ARGS \
       $EVALUATE_ARGS \
       2>&1 | tee logs/log-${EXPERIMENT_NAME}_${DATESTR}.txt
