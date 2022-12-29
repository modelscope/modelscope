PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=2 \
    examples/pytorch/finetune_image_classification.py \
    --num_classes 2 \
    --train_data 'tany0699/cats_and_dogs' \
    --validation_data 'tany0699/cats_and_dogs'
