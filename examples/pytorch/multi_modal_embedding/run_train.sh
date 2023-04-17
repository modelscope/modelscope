DATA_PARALLEL_SIZE=2


PYTHONPATH=. torchrun --nproc_per_node $DATA_PARALLEL_SIZE \
    examples/pytorch/multi_modal_embedding/finetune_multi_modal_embedding.py \
    --trainer 'clip-multi-modal-embedding' \
    --work_dir './workspace/ckpts/clip' \
    --model 'damo/multi-modal_clip-vit-base-patch16_zh' \
    --dataset_name 'muge' \
    --dataset_column_map 'img=image,text=query' \
    --max_epochs 1 \
    --use_fp16 true \
    --per_device_train_batch_size 180 \
    --train_shuffle true \
    --train_drop_last true \
    --per_device_eval_batch_size 128 \
    --eval_shuffle true \
    --eval_drop_last true \
    --save_ckpt_best true \
    --save_ckpt_best_strategy by_step \
    --ckpt_best_interval 200 \
    --metric_for_best_model inbatch_t2i_recall_at_1 \
    --logging_interval 1 \
    --eval_strategy by_step \
    --eval_interval 200 \
    --eval_metrics 'inbatch_recall' \
    --optimizer_lr 2.5e-05 \
    --optimizer 'AdamW' \
    --optimizer_hparams 'weight_decay=0.001,beta1=0.9,beta2=0.999,eps=1e-08' \
    --loss_aggregate true \
    --lr_warmup_proportion 0.1 \
    --lr_scheduler_hook 'type=LrSchedulerHook,by_epoch=false' \
    --optimizer_hook 'type=TorchAMPOptimizerHook,cumulative_iters=1,loss_keys=loss' \
    --clip_clamp true \
    --world_size $DATA_PARALLEL_SIZE \
