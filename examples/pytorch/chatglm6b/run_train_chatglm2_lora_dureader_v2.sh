LR=5e-5

PYTHONPATH=. python examples/pytorch/chatglm6b/finetune.py \
    --train_dataset_name modelscope/DuReader_robust-QG \
    --val_dataset_name modelscope/DuReader_robust-QG \
    --train_subset_name default \
    --val_subset_name default \
    --train_split train \
    --val_split validation \
    --prompt_column text1 \
    --response_column text2 \
    --model "ZhipuAI/chatglm2-6b" \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --train.optimizer.options.cumulative_iters 1 \
    --max_epochs 2 \
    --save_strategy 'by_step' \
    --save_interval 300 \
    --lr $LR \
    --eval_strategy "by_step" \
    --eval_interval 300 \
    --lr_strategy 'by_step' \
    --task 'chat' \
    --model.type 'chatglm2-6b' \
    --use_lora 1 \
    --work_dir lora_dureader_target \
