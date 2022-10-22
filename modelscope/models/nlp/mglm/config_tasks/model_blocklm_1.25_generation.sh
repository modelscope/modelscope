MODEL_TYPE="blocklm-1.25-generation"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --num-layers 30 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-position-embeddings 512 \
            --tokenizer-model-type bert-large-uncased \
            --tokenizer-type BertWordPieceTokenizer \
            --load-pretrained ${CHECKPOINT_PATH}/blocklm-1.25-generation"
