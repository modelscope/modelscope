DATA_ROOT=/dataset/c07bd62b/superglue
source config_tasks/model_blocklm_roberta_1.25.sh
source $1

CHECKPOINT_PATH="/dataset/c07bd62b/finetune_checkpoints"

if [ -z $N_GPU ];then
  N_GPU=2
fi
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DISTRIBUTED_ARGS="--nproc_per_node ${N_GPU} --nnodes 1 --node_rank 0 --master_addr localhost --master_port $MASTER_PORT"

DATESTR=$(date +"%m-%d-%H-%M")
GRID_LOG=logs/grid_${EXPERIMENT_NAME}_${DATESTR}.txt
mkdir logs
for lr in 6e-6 1e-5 2e-5
do
  for seed in 1234 5678 3456
  do
  HYPER=${lr}-${seed}
  PER_GPU_BS=$(($BATCH_SIZE/$N_GPU))
  if [ ! -f runs/${EXPERIMENT_NAME}/${HYPER}/test_results.json ]; then
    echo runs/${EXPERIMENT_NAME}/${HYPER}
    python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_gpt2.py \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME}/${HYPER} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${CHECKPOINT_PATH} \
       --seq-length ${MAX_SEQ_LEN} \
       --checkpoint-activations \
       --eval-batch-size 16 \
       --save-epoch 1000 \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       --fp16 \
       --attention-scale 8.0 \
       --batch-size ${PER_GPU_BS} \
       --epochs ${EPOCH_SINGLE} \
       --lr-decay-style linear \
       --lr ${lr} \
       --seed ${seed} \
       --overwrite \
       2>&1 | tee logs/log-${EXPERIMENT_NAME}-${HYPER}.txt
  fi
  echo $lr $seed >> $GRID_LOG
  cat runs/${EXPERIMENT_NAME}/${HYPER}/results.json >> $GRID_LOG
  done
done

echo $EXPERIMENT_NAME >> $GRID_LOG
