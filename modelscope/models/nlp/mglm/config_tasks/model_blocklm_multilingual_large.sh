MODEL_TYPE="blocklm-large-multilingual"
MODEL_ARGS="--block-lm \
            --task-mask \
            --cloze-eval \
            --num-layers 24 \
            --hidden-size 1536 \
            --num-attention-heads 16 \
	    --max-position-embeddings 1024 \
            --tokenizer-type ChineseSPTokenizer \
            --tokenizer-model-type /aigroup_bak/workspace/glm8/model/glm/MGLM模型/代码/Multilingual-GLM-main/tokenizer/mglm250k/mglm250k-uni.model \
            --load-pretrained /aigroup_bak/workspace/glm8/model/glm/MGLM模型/模型参数/0"
