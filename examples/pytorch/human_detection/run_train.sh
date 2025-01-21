PYTHONPATH=. python examples/pytorch/human_detection/finetune_human_detection.py \
    --dataset_name "person_detection_for_train" \
    --namespace "modelscope" \
    --model "damo/cv_tinynas_human-detection_damoyolo" \
    --num_classes 1 \
    --batch_size 2 \
    --max_epochs 3 \
    --base_lr_per_img 0.001
