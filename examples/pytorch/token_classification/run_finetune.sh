PYTHONPATH=. torchrun examples/pytorch/token_classification/finetune_token_classification.py \
    --model 'damo/mgeo_backbone_chinese_base' \
    --dataset_name 'GeoGLUE' \
    --subset_name 'GeoETA' \
    --name_space 'damo' \
    --train_dataset_param 'first_sequence=tokens,label=ner_tags,sequence_length=128' \
    --preprocessor 'token-cls-tokenizer' \
    --padding 'max_length' \
    --per_device_train_batch_size 32 \
    --max_epochs 1 \
    --lr 3e-5 \
    --logging_interval 100 \
    --eval_strategy 'by_epoch' \
    --save_ckpt_strategy 'by_epoch' \
    --work_dir './tmp'













