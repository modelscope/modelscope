PYTHONPATH=. torchrun examples/pytorch/text_generation/finetune_text_generation.py \
    --trainer 'text-generation-trainer' \
    --work_dir './tmp' \
    --model 'damo/nlp_palm2.0_pretrained_chinese-base' \
    --train_dataset_name 'DuReader_robust-QG' \
    --src_txt 'text1' \
    --tgt_txt 'text2' \
    --max_epochs 1 \
    --use_model_config True \
    --per_device_train_batch_size 8 \
    --lr 1e-3 \
    --lr_scheduler 'noam' \
