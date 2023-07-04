PRE_SEQ_LEN=128
LR=2e-2

PYTHONPATH=. python examples/pytorch/chatglm6b/finetune.py \
    --train_dataset_name AdvertiseGen/train.json \
    --val_dataset_name AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --model "ZhipuAI/ChatGLM-6B" \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --train.optimizer.options.cumulative_iters 1 \
    --max_epochs 1 \
    --save_strategy 'by_step' \
    --save_interval 1000 \
    --lr $LR \
    --eval_strategy "by_step" \
    --eval_interval 1000 \
    --lr_strategy 'by_step' \
    --task 'chat' \
    --model.type 'chatglm6b' \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4 \
    --work_dir ptuning_adv_target \
