DATA_PARALLEL_SIZE=2
TENSOR_MODEL_PARALLEL_SIZE=2

WORLD_SIZE=$(($DATA_PARALLEL_SIZE * $TENSOR_MODEL_PARALLEL_SIZE))


PYTHONPATH=. torchrun --nproc_per_node $WORLD_SIZE examples/pytorch/text_generation/finetune_text_generation.py \
    --trainer 'nlp-gpt3-trainer' \
    --work_dir './tmp' \
    --model 'damo/nlp_gpt3_text-generation_1.3B' \
    --dataset_name 'chinese-poetry-collection' \
    --preprocessor 'text-gen-jieba-tokenizer' \
    --src_txt 'text1' \
    --tgt_txt 'text2' \
    --max_epochs 3 \
    --per_device_train_batch_size 16 \
    --lr 3e-4 \
    --lr_scheduler 'noam' \
    --eval_metrics 'ppl' \
    --world_size $WORLD_SIZE \
    --tensor_model_parallel_size $TENSOR_MODEL_PARALLEL_SIZE \
    # --dataset_name 'DuReader_robust-QG' \ # input&output
