PYTHONPATH=. python examples/pytorch/text_classification/finetune_text_classification.py \
          --model 'damo/nlp_structbert_backbone_base_std' \
          --dataset_name 'clue' \
          --subset_name 'tnews' \
          --first_sequence 'sentence' \
          --preprocessor.label label \
          --model.num_labels 15 \
          --labels '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14' \
          --preprocessor 'sen-cls-tokenizer' \
          --lr 2e-5 \
          --eval_interval 100 \
          --optimizer.type 'AdamW' \
          --optimizer.lr 2e-5 \
          --lr_scheduler.type 'LinearLR' \
          --lr_scheduler.start_factor 1.0 \
          --lr_scheduler.end_factor 0.0 \
          --by_epoch false \
          --epoch 5 \
          --batch_size 32\

