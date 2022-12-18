export TOKENIZERS_PARALLELISM=false
export BC6PM_dir=../BC6PM
export pretrain_dir=../BioBERT-Models
export TMPDIR='./tmp'
export CUDA_VISIBLE_DEVICES='0'
python -u train.py \
--task BC6PM \
--do_train \
--do_eval \
--do_rc \
--loggerComment RC-Only \
--do_cross_valid \
--fineTune \
--train_withGNP