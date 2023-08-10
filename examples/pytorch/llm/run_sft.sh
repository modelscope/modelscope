CUDA_VISIBLE_DEVICES=0,1 \
python llm_sft.py \
    --model_type polylm-13b \
    --output_dir runs \
    --dataset alpaca-en,alpaca-zh \
    --dataset_sample 20000
