PYTHONPATH=. torchrun --standalone --nnodes=1 --nproc_per_node=2 examples/pytorch/keyword_spotting/finetune_keyword_spotting.py \
--work_dir './test_kws_training' \
--model 'damo/speech_charctc_kws_phone-xiaoyun' \
--train_scp './example_kws/train_wav.scp' \
--cv_scp './example_kws/cv_wav.scp' \
--merge_trans './example_kws/merge_trans.txt' \
--keywords '小云小云' \
--test_scp './example_kws/test_wav.scp' \
--test_trans './example_kws/test_trans.txt'
