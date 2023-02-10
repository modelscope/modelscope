PYTHONPATH=. python examples/pytorch/transformers/finetune_transformers_model.py \
    --model bert-base-uncased \
    --num_labels 15 \
    --dataset_name clue \
    --subset_name tnews
