PYTHONPATH=. torchrun examples/pytorch/text_generation/finetune_text_generation.py \
    --model 'damo/nlp_gpt3_text-generation_1.3B' \
    --dataset_name 'chinese-poetry-collection' \
    --preprocessor 'text-gen-jieba-tokenizer' \
    --src_txt 'text1' \
    --max_epochs 3 \
    --per_device_train_batch_size 16 \
    --lr 3e-4 \
    --lr_scheduler 'noam' \
    --eval_metrics 'ppl' \
